"""Review probes: one rerunnable assertion for every finding of a review pass.

Sections that have been fixed are deleted from this file, because a permanent
guard has taken over.  Three have gone that way so far: `--seed` being undone
by a second shuffle, now held by `runner.build_jobs` and tests/test_runner.py;
the breaker probe wedging in half_open, held by two tests in
tests/test_reliability.py and by bug 4 in scripts/mutation_check.py; and
export_csv running without authorization, held by the allowlist gate in
`agent._execute` and by tests/test_authz.py.

The lesson from the first of those: that section reproduced two lines of runner
inline instead of calling the real function, so it kept confirming after the
code was fixed, and its green meant nothing.  A probe has to call the function
under test, never a copy of its logic.

Run it with:
    python scripts/review_probes.py

How to read it: each section states its conclusion first, then prints the
measurements behind it.  Any FAIL exits 1, because these assertions describe
behaviour as it currently stands.  When one stops holding, the problem it
describes has been fixed -- come here and delete the section.
"""
import glob
import json
import random

from sql_agent.paths import ROOT  # noqa: F401

from sql_agent import verify as V

FAILURES: list[str] = []


CHECKS: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append(name)
    print(f"  [{'ok' if condition else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# -- 3. run_store: the status moves, the timestamp does not -----------------
print("\n3. run_store.py: ON CONFLICT updates only status and pending, "
      "and the table has no updated_at")

from sql_agent import run_store                              # noqa: E402

conn = run_store.connect()
try:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
    check("the table has no last-updated column", "updated_at" not in cols, str(cols))

    rid = "__review_probe__"
    conn.execute("DELETE FROM runs WHERE run_id=?", (rid,))
    run_store.save(conn, rid, "q", {}, "awaiting_approval", [{"name": "export_csv"}])
    before = run_store.load(conn, rid)
    run_store.save(conn, rid, "q", {}, "done", None)
    after = run_store.load(conn, rid)
    check("status changed while created_at stayed identical",
          before["status"] != after["status"] and before["created_at"] == after["created_at"],
          f"{before['status']} -> {after['status']} @ {after['created_at']}")
    conn.execute("DELETE FROM runs WHERE run_id=?", (rid,))
    conn.commit()
finally:
    conn.close()


# -- 4. the grader: random differential search for a diagnosis that names a
#       dimension which actually matches ------------------------------------
print("\n4. verify._diagnose: the dimension it names has to be one that really differs")

random.seed(0)
VALS = [0, 1, 2, "a", "b", None, 1.0, 3.14159]

def rnd_table() -> list[tuple]:
    r, c = random.randint(0, 3), random.randint(1, 3)
    return [tuple(random.choice(VALS) for _ in range(c)) for _ in range(r)]

n_fail = n_bad = 0
for _ in range(200_000):
    got, want = rnd_table(), rnd_table()
    ordered = random.choice([True, False])
    if V._compare(got, want, ordered) != V.FAIL:
        continue
    n_fail += 1
    reason = V._diagnose(got, want, ordered)
    g, w = V._round_floats(got), V._round_floats(want)
    bad = (
        (reason.startswith("column count") and (not g or not w or len(g[0]) == len(w[0])))
        or (reason.startswith("row count") and len(g) == len(w))
        or (reason.startswith("empty result") and bool(g))
        or reason.startswith("mismatch")          # catch-all for a diagnosis that disagrees with the verdict
    )
    n_bad += bad
check(f"none of the {n_fail} FAIL cases names a dimension that matches", n_bad == 0)

# also: asks_question=False inside expect is silently a no-op
r = V.verify_behavior("SELECT 1", "here is the answer.", {"asks_question": False})
check("expect={'asks_question': False} produces no check at all", r.passed is True)


# -- 5. the LOOSE verdict, a known source of false positives, has never fired
#       in the recorded results -------------------------------------------
print("\n5. the LOOSE verdict in verify (tolerates column order, and is a known "
      "false-positive channel)")

tot = loose = 0
for p in sorted(glob.glob(str(ROOT / "results" / "*.summary.json"))):
    s = json.loads(open(p, encoding="utf-8").read())
    tot += s.get("n_records", 0)
    loose += s.get("loose_count", 0)
check(f"LOOSE appears {loose} times across {tot} recorded verdicts", loose == 0)


print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} assertion(s) did not hold: {FAILURES}")
    raise SystemExit(1)
print(f"all {len(CHECKS)} assertions confirmed.")
