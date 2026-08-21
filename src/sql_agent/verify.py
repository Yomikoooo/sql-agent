"""Verifier: run the agent's SQL and the gold SQL, compare the result sets.

Compares *results*, not SQL text -- one question has unbounded many correct
queries. Two levels: strict (column positions must match) then loose (column
order tolerated), and the level is reported so loose verdicts can be sampled
by hand. The failure reason is raw material for failure classification, so it
names the dimension that differs rather than dumping both result sets.
"""
import sqlite3
from dataclasses import dataclass

EXACT = "exact"         # strict match, trustworthy
LOOSE = "loose"         # only column order differs -- needs a spot check
FAIL = "fail"
BEHAVIOR = "behavior"   # behavioural check passed (no result set compared)


@dataclass
class VerifyResult:
    passed: bool
    level: str          # EXACT | LOOSE | FAIL | BEHAVIOR
    reason: str

    def __bool__(self) -> bool:
        return self.passed


def _round_floats(rows: list[tuple], digits: int = 2) -> list[tuple]:
    """Round floats so that a different summation order does not fail a match."""
    return [
        tuple(round(v, digits) if isinstance(v, float) else v for v in row)
        for row in rows
    ]


def _sort_rows(rows: list[tuple]) -> list:
    """Sort rows (for ordered=False). Keys are strings so mixed types compare."""
    return sorted(rows, key=lambda row: tuple(str(v) for v in row))


def _sort_within_row(rows: list[tuple]) -> list[tuple]:
    """Sort within each row, tolerating column order. This is what makes the
    loose level useful -- and it is also where its false positives come from:
    two swapped values of the same type are indistinguishable from a reorder."""
    return [tuple(sorted(str(v) for v in row)) for row in rows]


def _compare(got: list[tuple], want: list[tuple], ordered: bool) -> str:
    """Return the verdict level."""
    got = _round_floats(got)
    want = _round_floats(want)

    # Level 1: strict -- column positions must line up too.
    a, b = (got, want) if ordered else (_sort_rows(got), _sort_rows(want))
    if a == b:
        return EXACT

    # Level 2: loose -- tolerate column order.
    a2, b2 = _sort_within_row(got), _sort_within_row(want)
    if not ordered:
        a2, b2 = sorted(a2), sorted(b2)
    if a2 == b2:
        return LOOSE

    return FAIL


def _diagnose(got: list[tuple], want: list[tuple], ordered: bool) -> str:
    """On FAIL, name where the two differ: coarsest dimension first, and only
    the first one that actually disagrees.

    Must normalise exactly as _compare does, or the reported position is a
    fiction.
    """
    got = _round_floats(got)
    want = _round_floats(want)

    # Dimension 0: empty result set.
    # Must come before the column check. An empty set has no column count, so
    # gw/ww below would collapse to 0 and the message would blame the columns
    # while the real difference is that there are no rows.
    if not got and want:
        return f"empty result: got 0 rows, want {len(want)} rows"
    if got and not want:
        return f"expected empty: got {len(got)} rows, want 0 rows"

    # Dimension 1: column count.
    gw = len(got[0])
    ww = len(want[0])
    if gw != ww:
        # Report the count, not the row: the dimension named and the quantity
        # reported have to be the same thing.
        return f"column count differs: got {gw} cols, want {ww} cols"

    # Dimension 2: row count.
    if len(got) != len(want):
        # Not the whole result set: 100 rows of output buries the point.
        return f"row count differs: got {len(got)} rows, want {len(want)} rows"

    # Dimension 3: first differing cell.
    # Same normalisation as level 1 (sort rows when ordered=False), otherwise
    # the reported position describes an ordering the verdict never cared about.
    a, b = (got, want) if ordered else (_sort_rows(got), _sort_rows(want))
    for i, (ra, rb) in enumerate(zip(a, b)):
        for j, (va, vb) in enumerate(zip(ra, rb)):
            if va != vb:        # floats already rounded above
                where = f"row {i}, col {j}"
                if not ordered:
                    where += " (position after sorting rows)"
                return f"first diff at {where}: got {va!r}, want {vb!r}"

    # _compare said FAIL but no dimension disagrees. That is not "could not
    # find it" -- diagnosis and verdict are using different rules, and one of
    # them is wrong. Saying plain "mismatch" would disguise a bug as a normal
    # failure.
    return "mismatch (no dimension differs -- diagnosis and verdict disagree; this is a bug)"


def verify_sql(agent_sql: str, gold_sql: str, db_path: str,
               ordered: bool = False) -> VerifyResult:
    """Run both queries and compare."""
    if not agent_sql or not agent_sql.strip():
        return VerifyResult(False, FAIL, "agent produced no SQL")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)   # read-only
    try:
        try:
            got = conn.execute(agent_sql).fetchall()
        except sqlite3.Error as e:
            return VerifyResult(False, FAIL, f"agent sql failed: {e}")

        try:
            want = conn.execute(gold_sql).fetchall()
        except sqlite3.Error as e:
            # The gold query itself is broken: my task set is at fault here,
            # not the agent. Shouted so it is never read as a model failure.
            return VerifyResult(False, FAIL, f"GOLD SQL FAILED: {e}")
    finally:
        conn.close()

    level = _compare(got, want, ordered)
    if level == EXACT:
        return VerifyResult(True, EXACT, "ok")
    if level == LOOSE:
        return VerifyResult(True, LOOSE, "ok (column order differs — needs spot check)")
    return VerifyResult(
        False, FAIL,
        _diagnose(got, want, ordered),
    )


# ---------------------------------------------------------------- behaviour
# The second kind of check. An ambiguous question should be answered with a
# question, and an unanswerable one by saying what is missing -- neither has a
# gold SQL to compare against, so verify_sql cannot judge them at all. Those
# two are what separate a good agent from a model that writes fluent SQL.
#
# The criteria are structural only. "Did it submit an answer" is deterministic,
# free and reproducible; "was the wording appropriate" needs an LLM judge, and
# a judge has to be validated before it can be trusted. Only the former is
# checked here -- what cannot be judged is declared unjudged, not faked.


def verify_behavior(submitted_sql: str | None, answer: str,
                    expect: dict, tool_calls: int = 0) -> VerifyResult:
    """Check one run against its expected behaviour.

    `expect` takes three optional keys:
        submitted:      bool -- should it have called submit_answer
        asks_question:  bool -- should the answer contain a question mark
        min_tool_calls: int  -- how many tool calls it must have made

    min_tool_calls closes a hole that was hit for real: with only
    submitted=False, a weak model that never calls any tool passes every
    behavioural task for free. qwen2.5:1.5b scored 4/4 on them while making
    zero tool calls on three -- it had done nothing at all.

    The right behaviour for an unanswerable question is not "say no without
    looking", it is "read the schema, confirm the column is absent, then say
    no". min_tool_calls>=1 separates those two.

    Passing returns level=BEHAVIOR; failing lists every criterion that missed.
    """
    problems = []

    if (need := expect.get("min_tool_calls")) and tool_calls < need:
        problems.append(
            f"expected at least {need} tool call(s) but got {tool_calls} - "
            f"the agent must actually look at the schema before concluding")

    if "submitted" in expect:
        want = expect["submitted"]
        got = submitted_sql is not None
        if got != want:
            problems.append(
                f"expected submitted={want} but got {got}"
                + (f" (submitted: {submitted_sql[:80]!r})" if got else "")
            )

    # Only checked when true: writing asks_question=False silently checks
    # nothing. Kept as-is because tasks currently rely on it; the fix is to
    # test `"asks_question" in expect` and assert both directions.
    if expect.get("asks_question") and "?" not in answer:
        problems.append("expected the agent to ask a clarifying question, "
                        "but the answer contains no '?'")

    if problems:
        return VerifyResult(False, FAIL, "; ".join(problems))
    return VerifyResult(True, BEHAVIOR, "ok (behavior)")
