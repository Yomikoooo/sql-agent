"""Task-set self-check. Run it after adding or editing any task.

Why it exists: the "tie at the LIMIT boundary" defect appeared twice. The first
time it was fixed by hand as a one-off; the second time it came back on another
task. The test for automating something is not how annoying it is, it is
whether it recurs as the set grows -- and anything that recurs has to be caught
by a script rather than by remembering.

    python check_tasks.py
    echo $?          # 0 = all clear, 1 = problems found
"""
import json
import re
import sqlite3
import sys
from pathlib import Path
from . import paths

DB = paths.DB_PATH
TASKS_FILE = paths.TASKS_DIR / "dev.json"

MAX_GOLD_ROWS = 100      # beyond this, comparing result sets tells you little


def _numeric_cells(row: tuple) -> tuple:
    """Keep only the numeric cells of a row.

    Whole rows are the wrong comparison: differing names are not a tie. What
    decides a tie is the sort key -- an order count, an amount -- and those are
    numeric. Two different customers with 10 orders each are tied, however
    different their names.
    """
    return tuple(v for v in row if isinstance(v, (int, float)))


def _order_by_keys(sql: str) -> list[str]:
    """The sort expressions in ORDER BY, or [] if there is none.

    The count is what matters: with a single sort key, equal keys mean the row
    order is undefined. With a second key (ORDER BY count DESC, name ASC) the
    order is already pinned, and reporting a tie there is a false positive.

    Known limitation: this splits on commas, so ORDER BY COALESCE(a, b) counts
    as two keys. No task is written that way today. If one were, this would
    miss a real tie rather than invent one.
    """
    m = re.search(r"ORDER\s+BY\s+(.+?)(?:\s+LIMIT\b|$)", sql, re.I | re.S)
    if not m:
        return []
    return [k.strip() for k in m.group(1).split(",") if k.strip()]


# ---------------------------------------------------------------- the checks
# Each returns None when the task passes, or one printable sentence when not.


def check_gold_runs(conn: sqlite3.Connection, task: dict) -> str | None:
    """The gold SQL runs at all. This gates every other check, hence a
    signature without rows.

    A task whose gold does not run fails forever, and it looks like the agent
    is at fault.
    """
    try:
        conn.execute(task["gold_sql"]).fetchall()
    except sqlite3.Error as e:
        return f"gold sql failed: {e}"
    return None


def check_not_empty(conn: sqlite3.Connection, task: dict, rows: list) -> str | None:
    """The gold result is not empty.

    On an empty-result task the agent can write nonsense and still "match",
    because both sides are empty. That is a free pass and a false positive.
    """
    # allow_empty=True declares that emptiness is the intended answer. Without
    # the flag such tasks would be reported as broken forever -- and they are
    # an important class, testing whether the agent says zero honestly or
    # invents a number.
    if not rows and not task.get("allow_empty"):
        return "gold returns 0 rows (set allow_empty=True if that is intended)"
    return None


def check_not_too_many_rows(conn: sqlite3.Connection, task: dict, rows: list) -> str | None:
    """The gold result is not larger than MAX_GOLD_ROWS.

    A task returning 200 rows separates nothing: SELECT * gets most of it right
    by accident.
    """
    if len(rows) > MAX_GOLD_ROWS:
        return f"gold returns {len(rows)} rows (> {MAX_GOLD_ROWS}), low signal"
    return None


# This word list has been widened twice; do not trim it back. The first version
# had only spend/buy/purchase/revenue/paid and missed "best-selling / units
# sold". Selling has a whole family of phrasings in English, and the model
# writing the tasks reaches for a different one each time.
SALES_WORDS = re.compile(
    r"\b(spen[dt]|spending|bought|buy|purchas\w*|revenue|paid|sold|sell\w*|sales)\b", re.I)
STATUS_WORDS = re.compile(r"\b(complet\w*|cancel\w*|pending|status|every status|all statuses)\b", re.I)


def check_order_status_specified(conn: sqlite3.Connection, task: dict, rows: list) -> str | None:
    """A task about sales or spending must say whether unfinished orders count.

    "Who has spent the most" -- do cancelled orders count as spending? Both
    readings are defensible. The agent sometimes adds WHERE status='completed'
    and sometimes does not, so the task turns flaky and the model gets blamed.
    The ambiguity is in the question.

    The precondition here was wrong twice, in opposite directions:

      - Too narrow: `if "orders" not in gold: return None`. A task about
        revenue share whose gold used only order_items, never joining orders,
        slipped through -- yet the agent could join it itself and filter on
        status, so the ambiguity was still there.
      - Too wide: dropping the precondition entirely, which immediately
        produced false positives. "The electronics we sell" queries current
        prices in the product catalogue; `sell` there describes the catalogue,
        not a measurement of sales, and status does not enter into it.

    Fixing a criterion that is too narrow, the easy mistake is deleting it into
    one that is too wide. The property actually wanted is "the answer rests on
    transaction data", and transaction data is what has a status -- so the test
    is whether the gold touches `orders` *or* `order_items`. Naming only the
    first is exactly why the narrow version missed what it missed.

    This is still a keyword approximation of a semantic judgement and it will
    miss rephrasings. It assists the person writing tasks; it does not replace
    thinking the question through.
    """
    gold = task.get("gold_sql", "").lower()
    if "orders" not in gold and "order_items" not in gold:
        return None                        # no transaction data, so no status
    if not SALES_WORDS.search(task["question"]):
        return None                        # not about sales, ambiguity unlikely
    if STATUS_WORDS.search(task["question"]):
        return None                        # the question already says

    return ("question is about sales/spending but never says which order statuses "
            "to count — the agent may or may not filter on status, which shows up "
            "as flakiness. Add 'Include orders of every status.' or "
            "'Only count completed orders.'")


SUPERLATIVE = re.compile(r"\b(most|top|best|highest|lowest|largest|greatest|biggest)\b", re.I)
HOW_MANY_ROWS = re.compile(r"\b(all|every|each|top \w+|first \w+|\d+)\b", re.I)


def check_row_count_specified(conn: sqlite3.Connection, task: dict, rows: list) -> str | None:
    """A superlative plus a gold that sorts without LIMIT must say how many
    rows are wanted.

    "Which cities generate the most revenue? ... highest first." -- the top
    few, or all of them ranked? Both readings are defensible, and the agent was
    observed adding its own LIMIT 5 and returning one row fewer than the gold.

    Another keyword approximation that will miss cases; an aid, not a
    substitute for writing the question clearly.
    """
    gold = task.get("gold_sql", "")
    if not re.search(r"ORDER\s+BY", gold, re.I):
        return None                        # unsorted, so it does not apply
    if re.search(r"LIMIT\s+\d+", gold, re.I):
        return None                        # the gold caps the rows itself
    if not SUPERLATIVE.search(task["question"]):
        return None                        # no superlative, not a ranking
    if HOW_MANY_ROWS.search(task["question"]):
        return None                        # the question states how many

    return ("question uses a superlative and the gold sorts without LIMIT, but never "
            "says how many rows to return — the agent may add its own LIMIT. "
            "Say 'Rank every X' or 'Show the top N'.")


SORT_INSTRUCTION = re.compile(
    r"\b(first|sorted|sort by|order(ed)? by|in .{0,12}order|descending|ascending|"
    r"alphabetical\w*|chronological\w*)\b", re.I)


def check_ordered_flag_justified(conn: sqlite3.Connection, task: dict, rows: list) -> str | None:
    """ordered=True has to be justified by the question.

    ordered=True means row order is compared too. But when it is not stated
    explicitly it gets inferred from "does the gold have an ORDER BY" -- and
    that ORDER BY may exist only to drive a window function, with the question
    asking for no particular order at all. Then ordered=True fails answers that
    are correct but differently sorted.

    The rule: the question must contain an explicit sorting instruction (sort
    by, first, descending, ...). "Top 3 products" does not count; that asks for
    a set.
    """
    if not task.get("ordered"):
        return None
    if SORT_INSTRUCTION.search(task["question"]):
        return None
    return ("ordered=True but the question never asks for a specific order — "
            "row order will be enforced and correct answers may be failed. "
            "Set ordered=False, or say 'sorted by ...' in the question.")


def check_limit_tie(conn: sqlite3.Connection, task: dict, rows: list) -> str | None:
    """On a task with LIMIT n, ranks n and n+1 must not be tied. The main event.

    Which of two tied rows comes first is decided by the query plan, which is
    undefined behaviour. The gold and the agent write different SQL, so they
    get different plans, so LIMIT cuts at a different row, so the task fails
    unreproducibly through no fault of the answer.

    Method: strip the LIMIT, take three extra rows, compare the numeric cells
    of rank n against rank n+1.

    This check ignores the rows it is handed, because it has to run a different
    query -- hence it still needs the connection. The uniform signature is what
    lets CHECKS call every check the same way.
    """
    m = re.search(r"LIMIT\s+(\d+)", task["gold_sql"], re.I)
    if not m:
        return None                        # no LIMIT, does not apply

    n = int(m.group(1))
    # Strip the trailing semicolon before building the probe. A gold written
    # as "... LIMIT 3;" leaves "... ;" once LIMIT is removed, and appending
    # " LIMIT 6" makes it two statements; sqlite then reports that it can only
    # execute one at a time. That reads as a broken task when it is a bug in
    # this check, and it did reject two good candidate tasks that way.
    base = re.sub(r"LIMIT\s+\d+", "", task["gold_sql"].strip().rstrip(";"), flags=re.I)
    try:
        probe = conn.execute(f"{base} LIMIT {n + 3}").fetchall()
    except sqlite3.Error as e:
        return f"tie probe failed: {e}"

    if len(probe) <= n:
        return None                        # no rank n+1, so nothing to tie with

    # Zero-based: rank n is probe[n-1] and rank n+1 is probe[n]
    rank_n, rank_n1 = probe[n - 1], probe[n]

    # Identical rows are tied no matter how many sort keys there are
    if rank_n == rank_n1:
        return (f"TIE at LIMIT boundary: rank{n} and rank{n + 1} are identical "
                f"rows {rank_n} — the top-{n} set is not well defined")

    if _numeric_cells(rank_n) != _numeric_cells(rank_n1):
        return None                        # different sort keys, order is fixed

    # The sort keys are tied. That is only a problem with a single key;
    # ORDER BY count DESC, name ASC already pins the order.
    if len(_order_by_keys(task["gold_sql"])) >= 2:
        return None

    return (f"TIE at LIMIT boundary: rank{n}={rank_n} rank{n + 1}={rank_n1} "
            f"— single ORDER BY key, add a tie-break column")


# check_gold_runs is deliberately absent from CHECKS: it is the precondition
# for all of them, and check_one uses it as a gate. Adding a check means adding
# a name here, with the fixed signature (conn, task, rows) -> str | None.
#
# A dangling word before a full stop means the sentence was cut off. One task
# arrived reading "... Only include products that have been." -- the condition
# was never spoken.
#
# Full stops only, never question marks: ending a question on an auxiliary is
# ordinary English ("How many users do we have?") and the first version flagged
# it. Truncation shows up in the statements, because a generated task is a
# question followed by clarifications, and it is the last clarification that
# the token limit cuts.
DANGLING_END = re.compile(
    r"\b(that|which|who|whose|have|has|had|been|being|is|are|was|were|with|without"
    r"|of|in|on|for|from|to|by|and|or|but|the|a|an|than|at least|more than|fewer than)"
    r"\s*\.\s*$", re.I)


def check_question_is_complete(task: dict) -> str | None:
    """The question has to be a complete sentence.

    Why this is needed: the questions are model-generated, and model output
    gets cut off by the token limit. Generating tasks in bulk, a question
    truncated mid-sentence raises nothing -- the JSON is still valid, the task
    still runs, the gold still returns rows.

    One task entered the set that way. Its question stopped at "Only include
    products that have been." while its gold carried a
    `HAVING COUNT(DISTINCT u.city) >= 2` -- a threshold the question never
    mentioned. Every automated check passed, manual spot-checking missed it,
    and what finally exposed it was running the set through a strong model,
    which could not produce SQL for it either.

    This is a keyword approximation catching one shape of truncation: ending on
    a dangling word. A cut that happens to land on a noun gets through. The
    real backstop is running the set against a strong model, not this function.
    """
    q = task.get("question", "").strip()
    if not q:
        return "question is empty"
    if DANGLING_END.search(q):
        return (f"question looks truncated — it ends on a dangling word: "
                f"...{q[-40:]!r}. Rewrite it as a complete sentence and make sure "
                f"every condition in the gold SQL is actually stated.")
    return None


CHECKS = [
    check_not_empty,
    check_not_too_many_rows,
    check_limit_tie,
    check_order_status_specified,
    check_row_count_specified,
    check_ordered_flag_justified,
]


# ---------------------------------------------------------------- driver


def check_one(conn: sqlite3.Connection, task: dict) -> list[str]:
    """Run every check for one task and return the problems (empty = passed).

    The gold SQL executes once here and the rows are handed to each check.
    Every check used to run it itself, so the same statement ran three times --
    not slow at this size, but it is the same work done three times, and the
    three results could in principle disagree if the data moved underneath.
    Hoisting shared computation into the caller beats merging the checks into
    one large function: adding a check means adding a function, not editing
    anyone else's.
    """
    # Completeness applies to every task, so it runs before the split: a
    # truncated behaviour task is exactly as bad as a truncated result task.
    incomplete = [p] if (p := check_question_is_complete(task)) else []

    # Behaviour tasks have no gold_sql: the correct behaviour is not to query,
    # so there is no result set to compare. They have their own checks.
    if task.get("check") == "behavior":
        return incomplete + check_behavior_task(task)

    if problem := check_gold_runs(conn, task):
        return incomplete + [problem]      # nothing else can run on a broken gold

    rows = conn.execute(task["gold_sql"]).fetchall()
    return incomplete + [p for check in CHECKS if (p := check(conn, task, rows))]


def check_behavior_task(task: dict) -> list[str]:
    """Behaviour tasks: expect must exist, its keys must be recognised, and
    there must be no gold_sql.

    The key check is not optional. Misspell a key in expect and the verifier
    ignores it silently, so the task degrades into checking only whether
    something was submitted -- no error, just less measured than you think.
    """
    problems = []
    expect = task.get("expect")

    if not expect:
        return ["behavior task has no 'expect' dict"]

    known = {"submitted", "asks_question", "min_tool_calls"}
    if unknown := set(expect) - known:
        problems.append(f"unknown expect keys {sorted(unknown)}; known: {sorted(known)}")

    if "gold_sql" in task:
        problems.append("behavior task should not carry a gold_sql")

    return problems


def check_file(path: Path, conn: sqlite3.Connection) -> int:
    """Check one task file and return how many tasks had problems."""
    tasks = json.loads(path.read_text(encoding="utf-8"))

    # Duplicate ids: valid JSON, but they quietly corrupt per-task grouping
    ids = [t["id"] for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        print(f"DUPLICATE ids in {path.name}: {duplicates}")

    bad = 0
    for task in tasks:
        problems = check_one(conn, task)
        if problems:
            bad += 1
            print(f"\n[{path.name}] {task['id']}  {task['question'][:56]}")
            for p in problems:
                print(f"    - {p}")

    print(f"{len(tasks) - bad}/{len(tasks)} ok ({path.name})")
    return bad + len(duplicates)


def main() -> int:
    # This script takes no arguments; it always checks every file. But taking
    # no arguments and silently ignoring them are different things:
    # `check_tasks.py --tasks test.json` used to run happily and print results
    # for both files, leaving the impression that the flag had worked. Honour a
    # flag or reject it -- accepting and ignoring it is the worst of the three.
    if sys.argv[1:]:
        print(f"check_tasks.py takes no arguments (got {sys.argv[1:]}). "
              f"It always checks every file in tasks/.")
        return 2

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)      # read-only

    # The held-out set is checked too. It is never run, and that is exactly why
    # a fault in it would go unnoticed until the day the final number is
    # reported, which is far too late to find out.
    total = 0
    for name in ("dev.json", "test.json"):
        path = TASKS_FILE.parent / name
        if path.exists():
            total += check_file(path, conn)
    conn.close()
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
