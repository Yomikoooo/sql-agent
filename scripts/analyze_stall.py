"""Count stalling: how often the agent resubmits a query that already failed.

Why this exists: experiment I changed the wording of the SQL error hint. The
success rate answers "did it score better"; it does not answer "did it stop
going in circles", which is the mechanism the change was aimed at. That
mechanism is a directed count, and a directed count needs its own script --
a number produced in a throwaway shell cannot be re-derived later.

Reads the run files written by runner.py and reports, per run:

  stalled      runs where some run_sql query that errored was submitted again
  repeats      total number of such repeat submissions
  max_turns    runs that hit the turn budget
  no_sql       runs where no SQL was ever executed successfully

Counting only run_sql was the first version of this script and it was wrong:
submit_answer executes SQL too, and most SQL errors in this project come from
it, not from run_sql. That version reported no_sql = 70/76 on a run that passed
37/76 -- two numbers that cannot both be true, which is how the bug surfaced.

Usage:
    python scripts/analyze_stall.py results/2026-08-22T15-24-06.jsonl [...]
    python scripts/analyze_stall.py results/*.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path


def normalize(sql: str) -> str:
    """Whitespace-insensitive comparison.

    The model reformats the same query between turns often enough that a raw
    string match undercounts. Case is left alone: SELECT and select really are
    different tokens to the model, and collapsing them would hide a real
    difference in its behaviour.
    """
    return " ".join(sql.split())


def walk_calls(messages: list) -> list[tuple[str, str]]:
    """-> [(query, result_text)] for every SQL-executing call, in order.

    Both run_sql and submit_answer execute SQL, so both can fail and both can
    be resubmitted. Tracking only run_sql undercounts by roughly a factor of
    seven on this task set.

    A tool result is matched to its call by position: the assistant message
    carries the calls, and the tool messages that follow answer them in the
    same order. Matching on tool_call_id would be stricter, but local servers
    do not always echo the id back -- see PITFALLS on the missing-id crash.
    """
    out, pending = [], []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            pending = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                if fn.get("name") not in ("run_sql", "submit_answer"):
                    pending.append(None)
                    continue
                try:
                    pending.append(json.loads(fn.get("arguments", "{}")).get("query", ""))
                except json.JSONDecodeError:
                    pending.append("")          # malformed args: count the call, not the text
        elif m.get("role") == "tool" and pending:
            q = pending.pop(0)
            if q is not None:
                out.append((q, str(m.get("content", ""))))
    return out


def analyze(path: Path) -> dict:
    stats = Counter()
    variant = set()
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        variant.add(rec.get("error_hint_variant", "?"))
        stats["runs"] += 1
        if rec.get("stop_reason") == "max_turns":
            stats["max_turns"] += 1

        calls = walk_calls(rec.get("messages", []))
        failed, seen, repeats = set(), Counter(), 0
        any_ok = False
        for q, result in calls:
            key = normalize(q)
            seen[key] += 1
            if result.startswith("SQL error"):
                failed.add(key)
            else:
                any_ok = True
            # a resubmission counts only if this exact query had already failed
            if seen[key] > 1 and key in failed:
                repeats += 1
        if not any_ok:
            stats["no_sql"] += 1
        if repeats:
            stats["stalled"] += 1
            stats["repeats"] += repeats
    # Read every field through Counter.__getitem__, which returns 0 for a key
    # that was never incremented. `{**stats}` would drop those keys entirely,
    # and the caller would KeyError -- on the *cleanest* runs, the ones where
    # nothing stalled and nothing hit the turn budget. noise_a was the first
    # run with stop_reason {'done': 76} and it crashed the script.
    FIELDS = ("runs", "stalled", "repeats", "max_turns", "no_sql")
    return {"variant": "/".join(sorted(variant)), **{k: stats[k] for k in FIELDS}}


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    print(f"{'run':22}{'variant':9}{'runs':>6}{'stalled':>9}{'repeats':>9}"
          f"{'max_turns':>11}{'no_sql':>8}")
    for p in sorted(paths):
        s = analyze(p)
        print(f"{p.stem[:19]:22}{s['variant']:9}{s['runs']:>6}{s['stalled']:>9}"
              f"{s['repeats']:>9}{s['max_turns']:>11}{s['no_sql']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
