"""Tests for the verifier itself.

These are the tests that had to be written first.  A bug in the verifier makes
every pass rate that follows untrustworthy, and it looks like a problem with
the agent, so the search for it starts in the wrong place and stays there.

Run: pytest tests/test_verify.py -v
"""
import pytest

from sql_agent.paths import ROOT

from sql_agent.verify import BEHAVIOR, EXACT, FAIL, LOOSE, verify_behavior, verify_sql

DB = str(ROOT / "data" / "shop.db")


# ==================== basic correctness ====================

def test_identical_sql_passes():
    r = verify_sql("SELECT COUNT(*) FROM users", "SELECT COUNT(*) FROM users", DB)
    assert r.passed and r.level == EXACT


def test_different_query_fails():
    r = verify_sql("SELECT COUNT(*) FROM users", "SELECT COUNT(*) FROM orders", DB)
    assert not r.passed and r.level == FAIL


# ============ logically equivalent, written differently (the important class) ============

def test_equivalent_sql_passes():
    """Different wording, same result: this has to pass, or the verifier manufactures false negatives."""
    r = verify_sql(
        "SELECT COUNT(*) FROM users WHERE city = 'Beijing'",
        "SELECT COUNT(*) FROM users WHERE city IN ('Beijing')",
        DB,
    )
    assert r.passed


def test_alias_does_not_matter():
    """A different column alias must not change the verdict."""
    r = verify_sql(
        "SELECT COUNT(*) AS total FROM users",
        "SELECT COUNT(*) AS n FROM users",
        DB,
    )
    assert r.passed


# ==================== column order: two levels of verdict ====================

def test_column_order_falls_back_to_loose():
    """A different column order passes, but is marked loose for a human to spot-check."""
    r = verify_sql(
        "SELECT COUNT(*), city FROM users GROUP BY city",
        "SELECT city, COUNT(*) FROM users GROUP BY city",
        DB,
    )
    assert r.passed
    assert r.level == LOOSE          # not EXACT, so the difference is on record


def test_same_column_order_is_exact():
    r = verify_sql(
        "SELECT city, COUNT(*) FROM users GROUP BY city",
        "SELECT city, COUNT(*) FROM users GROUP BY city",
        DB,
    )
    assert r.level == EXACT


# ==================== row order ====================

def test_row_order_ignored_when_not_ordered():
    r = verify_sql(
        "SELECT name FROM products ORDER BY name ASC",
        "SELECT name FROM products ORDER BY name DESC",
        DB, ordered=False,
    )
    assert r.passed


def test_row_order_matters_when_ordered():
    r = verify_sql(
        "SELECT name FROM products ORDER BY name ASC",
        "SELECT name FROM products ORDER BY name DESC",
        DB, ordered=True,
    )
    assert not r.passed


# ==================== error handling ====================

def test_syntax_error_fails_with_reason():
    r = verify_sql("SELCT * FROM users", "SELECT * FROM users", DB)
    assert not r.passed
    assert "agent sql failed" in r.reason


def test_unknown_table_fails():
    r = verify_sql("SELECT * FROM nonexistent", "SELECT * FROM users", DB)
    assert not r.passed


def test_bad_gold_sql_is_flagged_separately():
    """A gold answer that does not run must be distinguishable from the agent being wrong."""
    r = verify_sql("SELECT * FROM users", "SELCT * FROM users", DB)
    assert not r.passed
    assert "GOLD SQL FAILED" in r.reason      # says at a glance that the task is the broken one


def test_empty_sql_fails():
    r = verify_sql("", "SELECT COUNT(*) FROM users", DB)
    assert not r.passed


# ==================== empty result vs error ====================

def test_both_empty_passes():
    """Both sides ran and both returned zero rows: that is a correct answer, not a failure."""
    r = verify_sql(
        "SELECT * FROM users WHERE city = 'Nowhere'",
        "SELECT * FROM users WHERE city = 'Atlantis'",
        DB,
    )
    assert r.passed


def test_empty_vs_nonempty_fails():
    r = verify_sql(
        "SELECT * FROM users WHERE city = 'Nowhere'",
        "SELECT * FROM users WHERE city = 'Beijing'",
        DB,
    )
    assert not r.passed


# ==================== writes have to be refused ====================

@pytest.mark.parametrize("sql", [
    "DELETE FROM users",
    "DROP TABLE users",
    "UPDATE users SET city = 'X'",
    "INSERT INTO users VALUES (9999, 'x', 'x', 'x', '2025-01-01')",
])
def test_write_operations_rejected(sql):
    """A read-only connection has to make every write fail."""
    r = verify_sql(sql, "SELECT COUNT(*) FROM users", DB)
    assert not r.passed


# ==================== self-check of the task set ====================

def test_all_gold_sql_runs():
    """Every gold_sql compared against itself has to come out EXACT.

    This catches a mistyped gold answer up front, which is far cheaper than
    discovering it after a full eval run.
    """
    import json

    tasks = json.loads(
        (ROOT / "tasks" / "dev.json").read_text(encoding="utf-8")
    )
    assert tasks, "tasks/dev.json is empty; run python build_tasks.py first"

    broken = []
    for t in tasks:
        if t.get("check") == "behavior":
            continue                      # behaviour tasks have no gold_sql: not querying is the correct behaviour
        r = verify_sql(t["gold_sql"], t["gold_sql"], DB, t["ordered"])
        if not r.passed or r.level != EXACT:
            broken.append((t["id"], r.reason))

    assert not broken, f"the gold_sql of these tasks is broken: {broken}"


def test_every_behavior_task_has_expect():
    """A behaviour task without `expect` makes verify_behavior check nothing and pass forever."""
    import json

    tasks = json.loads(
        (ROOT / "tasks" / "dev.json").read_text(encoding="utf-8")
    )
    bad = [t["id"] for t in tasks if t.get("check") == "behavior" and not t.get("expect")]
    assert not bad, f"these behaviour tasks have no expect: {bad}"


# ==================== the behaviour verifier (the second kind of check) ====================
# "test the verifier first" holds for every tool that renders a verdict, not
# just verify_sql.  check_tasks.py went untested and ran a whole round while
# reporting false positives.


def test_behavior_no_submit_expected_and_none_submitted():
    r = verify_behavior(None, "Top by what — revenue or units sold?",
                        {"submitted": False, "asks_question": True})
    assert r.passed and r.level == BEHAVIOR


def test_behavior_fails_when_agent_answers_an_ambiguous_question():
    """What failing an ambiguous task looks like: it did not ask, it just queried."""
    r = verify_behavior("SELECT * FROM products LIMIT 5", "Here are the top products.",
                        {"submitted": False, "asks_question": True})
    assert not r.passed
    assert "submitted=False" in r.reason
    assert "?" in r.reason                # both problems get reported, not just the first


def test_behavior_fails_when_it_queries_but_should_not():
    """Out-of-scope task: only the submission is graded, no question mark required."""
    r = verify_behavior("SELECT 1", "Our margin is 30%.", {"submitted": False})
    assert not r.passed


def test_behavior_passes_when_it_refuses_without_asking():
    r = verify_behavior(None, "The schema has no cost column, so margin cannot be computed.",
                        {"submitted": False})
    assert r.passed


def test_behavior_expects_a_submission():
    assert verify_behavior("SELECT COUNT(*) FROM users", "200 users.",
                           {"submitted": True}).passed
    assert not verify_behavior(None, "I am not sure.", {"submitted": True}).passed


def test_behavior_empty_expect_passes_everything():
    """Pinning the behaviour down: an empty `expect` checks nothing.

    That is why check_tasks has to require `expect` on behaviour tasks --
    without it a task passes silently, forever.
    """
    assert verify_behavior("SELECT 1", "whatever", {}).passed


# ==================== hygiene checks on the system prompt ====================
# What happened: a block of explanatory text starting with # was written inside
# the triple-quoted string, so it was not a Python comment but part of the
# prompt, and the model read it on every call.  It only surfaced when the prompt
# length jumped from 2,199 to 3,192 characters.


def test_system_prompt_has_no_chinese():
    """The system prompt has to be all English; stray non-English text usually means a misplaced comment."""
    from sql_agent.agent import SYSTEM_PROMPT

    cjk = [c for c in SYSTEM_PROMPT if "\u4e00" <= c <= "\u9fff"]
    assert not cjk, (
        f"SYSTEM_PROMPT holds {len(cjk)} CJK characters -- "
        f"is a comment sitting inside the triple quotes? first 40: {''.join(cjk[:40])}"
    )


def test_system_prompt_has_no_python_comments():
    """Inside triple quotes a line starting with # is text, not a comment."""
    from sql_agent.agent import SYSTEM_PROMPT

    bad = [ln for ln in SYSTEM_PROMPT.splitlines() if ln.lstrip().startswith("#")
           and not ln.lstrip().startswith("##")]      # ## is a markdown heading, allowed
    assert not bad, f"SYSTEM_PROMPT holds comment-looking lines, and they are sent to the model: {bad[:3]}"


# ==================== a bug inside check_tasks itself ====================
# What happened: with a gold answer ending in "... LIMIT 3;", stripping the
# LIMIT left "... ;", and appending the probe made it two statements, so sqlite
# raised.  It looked like a broken task; the check was the broken thing.  It
# threw out 2 candidate tasks before anyone noticed.


def test_limit_tie_handles_trailing_semicolon():
    """A trailing semicolon must not break the probe."""
    import sqlite3

    from sql_agent.check_tasks import check_limit_tie

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    task = {"question": "three most expensive products",
            "gold_sql": "SELECT name, price FROM products ORDER BY price DESC LIMIT 3;"}
    problem = check_limit_tie(conn, task, [])
    assert problem is None or "tie probe failed" not in problem, problem


# ============ two checks on ambiguous and truncated questions ============
# What happened: running the whole task set through a second model as a sanity
# check left 6 tasks it also got wrong.  Triage found 3 of them were broken
# tasks that all 7 linters had passed:
#   sql_013  "best-selling / units sold" -- the word list had neither sold nor selling
#   sql_016  the gold answer uses order_items without joining orders, so the
#            "orders appears in gold" premise was too narrow
#   test_517 the question stopped at "...products that have been." -- truncated
#            by max_tokens while it was being generated
#
# Each check is tested in both directions.  Testing only that a fixed task stays
# quiet would also pass if the check were deleted outright.


def _gold_check(question: str, gold_sql: str) -> list[str]:
    """Run every check against one throwaway task and return the problems found."""
    import sqlite3

    from sql_agent.check_tasks import check_one

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    return check_one(conn, {"id": "tmp", "question": question, "check": "result",
                            "gold_sql": gold_sql, "ordered": False})


SALES_GOLD = ("SELECT p.name, SUM(oi.quantity) AS t FROM order_items oi "
              "JOIN products p ON oi.product_id = p.id GROUP BY p.id")


def test_status_ambiguity_caught_for_selling_words():
    """'units sold' carries the same status ambiguity as 'spent'; the first word list missed it."""
    problems = _gold_check("What are our best-selling products by units sold?", SALES_GOLD)
    assert any("order statuses" in p for p in problems), problems


def test_status_ambiguity_caught_when_gold_only_uses_order_items():
    """A gold answer that skips the orders join is not evidence of no ambiguity: the agent can join and filter on its own."""
    problems = _gold_check(
        "What share of our total revenue does each category account for?",
        "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS r FROM order_items oi "
        "JOIN products p ON oi.product_id = p.id GROUP BY p.category")
    assert any("order statuses" in p for p in problems), problems


def test_status_ambiguity_not_raised_when_question_says_so():
    """Once the question says so the check has to go quiet, or it keeps firing after a fix and stops being believed."""
    problems = _gold_check(
        "What are our best-selling products by units sold? Include orders of every status.",
        SALES_GOLD)
    assert not any("order statuses" in p for p in problems), problems


def test_status_ambiguity_not_raised_for_catalog_only_question():
    """'the electronics we sell' reads current prices from products and never touches
    transactions, so there is no order status to be ambiguous about.

    This is the false positive produced by over-correcting the premise from too
    narrow to too wide.
    """
    problems = _gold_check(
        "Show me the name and price of all the electronics we sell.",
        "SELECT name, price FROM products WHERE category = 'Electronics'")
    assert not any("order statuses" in p for p in problems), problems


def test_truncated_question_is_caught():
    """A question ending on a dangling word plus a full stop was truncated as it was generated."""
    problems = _gold_check(
        "Show the product name and the city count. Only include products that have been.",
        "SELECT p.name, COUNT(DISTINCT u.city) AS c FROM products p "
        "JOIN order_items oi ON p.id = oi.product_id "
        "JOIN orders o ON oi.order_id = o.id JOIN users u ON o.user_id = u.id "
        "GROUP BY p.id, p.name")
    assert any("truncated" in p for p in problems), problems


def test_question_ending_in_auxiliary_verb_is_not_truncated():
    """'How many users do we have?' is a complete question, not a truncation.

    The dangling-word rule only applies to statements ending in a full stop; a
    question ending on an auxiliary verb is ordinary English.
    """
    problems = _gold_check("How many users do we have?", "SELECT COUNT(*) FROM users")
    assert not any("truncated" in p for p in problems), problems


# ============ retrieval: exclude_id is the leakage line ============
# What happens without it: searching the dev set with a dev question returns the
# question itself at rank 1 with similarity 1.0, and it carries the gold SQL.
# Not excluding it shows the model the answer, which produces a very good score
# that means nothing.
#
# Both directions are tested.  Testing only "the result excludes itself" would
# also pass for a function that always returns an empty list.


def _first_task_with_gold() -> dict:
    import json
    tasks = json.load(open(ROOT / "tasks" / "dev.json", encoding="utf-8"))
    return next(t for t in tasks if t.get("gold_sql"))


def test_overlap_search_finds_itself_without_exclude():
    """Direction one: without the exclusion it must find itself, or the test below cannot discriminate."""
    from sql_agent.retrieval import search_overlap

    t = _first_task_with_gold()
    hits = [x["id"] for x in search_overlap(t["question"], k=3)]
    assert t["id"] in hits, f"searching with the original question did not find it; retrieval itself is broken: {hits}"


def test_overlap_search_excludes_itself():
    """Direction two: with exclude_id passed, it must not appear."""
    from sql_agent.retrieval import search_overlap

    t = _first_task_with_gold()
    hits = [x["id"] for x in search_overlap(t["question"], k=3, exclude_id=t["id"])]
    assert t["id"] not in hits, f"exclude_id had no effect, and this is the leak: {hits}"
    assert len(hits) == 3, f"3 hits expected after excluding one, got {len(hits)}"


def _ollama_up() -> bool:
    import httpx

    try:
        return httpx.get("http://127.0.0.1:11434/", timeout=1.0).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="needs a local Ollama; skipped when offline")
def test_vector_search_excludes_itself():
    """The same two directions for the vector version.

    It depends on Ollama, hence the skipif: without it the whole suite goes red
    when offline, and a red suite that gets ignored is worse than no suite.
    """
    from sql_agent.retrieval import search_vector

    t = _first_task_with_gold()
    assert t["id"] in [x["id"] for x in search_vector(t["question"], k=3)]
    hits = [x["id"] for x in search_vector(t["question"], k=3, exclude_id=t["id"])]
    assert t["id"] not in hits, f"exclude_id had no effect, and this is the leak: {hits}"


# ==================== entrypoints of the command line tools ====================
# What happened: editing the reporting section of a script with a replacement
# that ran "from this line to the end of the file" deleted main() and the
# if __name__ == "__main__" block along with it.  Running the script then
# printed nothing at all -- no error, no warning, simply nothing.  What was
# verified before committing was the content (the length of VARIANTS), not
# whether the script still started.
#
# The four commands the README documents are console_scripts now, so for them
# the question is no longer "does the file have a __main__ block" but "does the
# name declared in pyproject.toml still resolve to something callable".  A
# renamed or moved function leaves the declaration behind pointing at nothing,
# and nothing notices until someone runs the command.


def test_declared_console_scripts_resolve():
    """Every console_script in pyproject.toml imports and names a callable."""
    import importlib
    import tomllib

    with open(ROOT / "pyproject.toml", "rb") as fh:
        declared = tomllib.load(fh)["project"]["scripts"]
    assert declared, "no console_scripts declared, so the commands ship as nothing"

    broken = []
    for command, target in declared.items():
        module_name, _, attr = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            broken.append(f"{command}: cannot import {module_name} ({e})")
            continue
        fn = getattr(module, attr, None)
        if not callable(fn):
            broken.append(f"{command}: {target} is not callable")
    assert not broken, broken


def test_shipping_scripts_have_entrypoints():
    """The scripts under scripts/ still run as scripts, so they need a __main__ block.

    The list is the set that ships, so a new one belongs here too.  It went red
    the moment these files moved into scripts/, and again when the modules moved
    under src/ -- which is its own demonstration that moving files breaks the
    ways of reaching them.  review_probes.py is deliberately absent: it runs top
    to bottom on import rather than through a main().
    """
    scripts = ["scripts/mock_llm_server.py", "scripts/mutation_check.py",
               "scripts/check_hitl_server_restart.py",
               "scripts/check_breaker_over_http.py"]
    broken = []
    for name in scripts:
        path = ROOT / name
        if not path.exists():
            broken.append(f"{name}: the file is gone (moved?)")
            continue
        if 'if __name__ == "__main__":' not in path.read_text(encoding="utf-8"):
            broken.append(f"{name}: no __main__ block")
    assert not broken, broken
