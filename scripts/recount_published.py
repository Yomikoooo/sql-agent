#!/usr/bin/env python3
"""Recompute the two published numbers under corrected grading.

Two grading defects were found in the task set after the numbers in the
README were published:

  1. Six questions ask for a specific order in their text and their gold
     query has ORDER BY, but carry `ordered: false`, so a prediction with the
     right rows in the wrong order was credited.
  2. `sql_030`'s gold query returns no rows, so any prediction that also
     returns nothing scores correct -- including a wrong one.

**Nothing is re-run.** The per-task records in results/published/ hold the
SQL each run actually submitted, so the corrections are applied to what
happened rather than to a fresh attempt by a model that has since changed.

This script deliberately does **not** re-implement the verifier. Re-grading
everything would mix two things that must stay separate: the effect of the
corrections, and any disagreement between a reimplementation and the grader
that produced the published number. It applies the two named corrections and
carries every other verdict through untouched.

    python scripts/recount_published.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/shop.db"
PUB = ROOT / "results/published"

RUNS = {
    # The starting point the README quotes as 29 %. ⚠️ It is a 31-task run,
    # not the 78-task dev set the 56.4 % comes from, so the two are not on
    # the same denominator. Recounted here so that at least both ends of the
    # claimed improvement use the same grading; the fact that they are
    # different task sets is a separate problem and is not fixed by this
    # script.
    "baseline-31": PUB / "2026-08-11T03-49-36.jsonl",
    "dev": PUB / "2026-08-11T19-08-40.jsonl",
    "test": PUB / "2026-08-11T21-37-22.jsonl",
}

# Question text asks for an order and the gold query has ORDER BY, but the
# task file says ordered: false.
SHOULD_BE_ORDERED = {"sql_110", "sql_117", "sql_146",
                     "test_513", "test_516", "test_521"}

# Gold returns no rows, so returning nothing scores correct. Carries no
# information either way; excluded from the denominator rather than deleted.
EMPTY_GOLD = {"sql_030"}


def rows(con, sql: str):
    try:
        return con.execute(sql).fetchall()
    except sqlite3.Error:
        return None


def main() -> None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    report = {}

    for split, path in RUNS.items():
        records = {r["task_id"]: r
                   for r in (json.loads(l) for l in path.read_text().splitlines()
                             if l.strip())}
        published = {t: bool(r.get("passed")) for t, r in records.items()}

        flipped, unchecked = [], []
        corrected = dict(published)
        for task in sorted(SHOULD_BE_ORDERED & set(records)):
            r = records[task]
            if not published[task]:
                continue                      # already wrong; order is moot
            sub = r.get("submitted_sql") or r.get("agent_sql")
            if not sub:
                unchecked.append((task, "no SQL stored"))
                continue
            got, gold = rows(con, sub), rows(con, r["gold_sql"])
            if got is None or gold is None:
                unchecked.append((task, "SQL no longer executes"))
                continue
            # Only flip when re-execution reproduces the published verdict
            # under the loose comparison. If it does not, this script and the
            # original grader disagree for some other reason, and that is not
            # something to silently fold into a corrected number.
            if sorted(map(str, got)) != sorted(map(str, gold)):
                unchecked.append((task, "re-execution does not reproduce the "
                                        "published pass"))
                continue
            if got != gold:
                corrected[task] = False
                flipped.append(task)

        excluded = sorted(EMPTY_GOLD & set(records))
        scored = [t for t in records if t not in excluded]

        n_pub, k_pub = len(records), sum(published.values())
        n_cor = len(scored)
        k_cor = sum(corrected[t] for t in scored)
        report[split] = {
            "file": path.name, "n_published": n_pub, "correct_published": k_pub,
            "rate_published": round(k_pub / n_pub, 4),
            "flipped_by_ordering": flipped,
            "excluded_empty_gold": excluded,
            "unchecked": unchecked,
            "n_corrected": n_cor, "correct_corrected": k_cor,
            "rate_corrected": round(k_cor / n_cor, 4),
        }

        print(f"=== {split} ===")
        print(f"  as published            {k_pub}/{n_pub} = {k_pub/n_pub:.1%}")
        if flipped:
            print(f"  credited on wrong order {flipped}")
        if excluded:
            print(f"  excluded, empty gold    {excluded}")
        for task, why in unchecked:
            print(f"  NOT corrected: {task} -- {why}")
        print(f"  corrected               {k_cor}/{n_cor} = {k_cor/n_cor:.1%}")
        print()

    out = ROOT / "results/published/recount.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"written: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
