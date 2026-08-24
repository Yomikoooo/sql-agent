"""Compare wording arms task by task, and size the wording-noise band.

Why this exists: the per-arm success rate is the wrong resolution for this
experiment. base scores 37 and inline 33, but "-4" hides the fact that the two
arms disagree on more than four tasks -- some flip each way and the totals
partly cancel. The question is not "which total is higher" but "is this gap
bigger than the gap produced by rewording that carries no information at all",
and only a task-level view answers it.

Two numbers here are easy to get wrong, and both were got wrong once:

  1. The band is NOT the spread of the noise family's totals. All three noise
     arms scored 34 while passing three *different* sets of 34 tasks, so the
     totals agreed by cancellation. Measured on totals the band is 3 and
     inline's -4 looks significant; measured as pairwise disagreement the
     family spans 3..7 tasks and inline's 4 sits in the middle of it. Totals
     cancel bidirectional flips, which is the exact failure this script was
     written to avoid.

  2. The denominator is NOT the task count. The hint text only reaches the
     model when a SQL statement errors, which happened in the same 36 of 76
     tasks in every arm. The other 40 cannot be affected by any rewording, so
     quoting effects over 76 halves them.

Reads the run files written by runner.py and reports:

  per arm      passes, flips against the reference arm, and hint exposure
  unstable     every task that is not unanimous across the arms given
  pairwise     disagreement between every pair; the noise family's own range
               is the band, and any arm inside it is inside noise

One run per arm is enough at temperature 0 with --workers pinned: reruns of
inline 11 hours apart were bit-identical (2026-08-22T15-49-10 and
2026-08-23T02-34-16). Anything that moves here moved because of the wording.

Usage:
    python scripts/compare_arms.py results/*.jsonl
    python scripts/compare_arms.py --ref base results/*.jsonl
"""
import argparse
import itertools
import json
from pathlib import Path

# The arms whose wording is informationally identical to base: same
# instruction, same tool named, same order of operations, same length band.
# Any disagreement between them is wording noise by construction, so their
# pairwise range IS the band. See tools.py, SQL_ERROR_HINTS.
NOISE_FAMILY = ("base", "noise_a", "noise_b", "noise_c")

# What a failing SQL tool result starts with. The hint is appended to exactly
# these, so this prefix is how a run is known to have seen the hint at all.
SQL_ERROR_PREFIX = "SQL error"


def _tool_texts(messages: list[dict]):
    """Yield the text of every tool result in one run's message log."""
    for m in messages:
        if m.get("role") == "tool":
            yield str(m.get("content", ""))


def load(path: Path) -> tuple[str, dict[str, bool], set[str], dict]:
    """Return (arm, {task_id: passed}, tasks_that_saw_the_hint, meta).

    Trials are collapsed with all(): a task counts as a pass only if every
    trial passed, matching how runner.py builds always_pass. At temperature 0
    there is normally one trial, so this is a no-op there.
    """
    by_task: dict[str, list[bool]] = {}
    exposed: set[str] = set()
    arms, meta = set(), {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        arms.add(rec.get("error_hint_variant", "?"))
        meta = {k: rec.get(k) for k in ("workers", "temperature", "model",
                                        "prompt_hash", "error_hint_hash")}
        by_task.setdefault(rec["task_id"], []).append(bool(rec["passed"]))
        if any(t.startswith(SQL_ERROR_PREFIX)
               for t in _tool_texts(rec.get("messages", []))):
            exposed.add(rec["task_id"])
    if len(arms) != 1:
        raise SystemExit(f"{path.name}: mixed arms {sorted(arms)}; "
                         "a run file must be one arm")
    return arms.pop(), {t: all(v) for t, v in by_task.items()}, exposed, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--ref", default="base", help="arm to diff the others against")
    args = ap.parse_args()

    # Later files win, so re-running an arm replaces it rather than duplicating.
    arms: dict[str, dict[str, bool]] = {}
    seen: dict[str, set[str]] = {}
    metas: dict[str, dict] = {}
    for path in sorted(args.runs):
        arm, results, exposed, meta = load(path)
        if arm in arms and arms[arm] != results:
            print(f"note: {arm} rerun in {path.name} disagrees with the earlier "
                  f"run -- greedy decoding is supposed to be reproducible")
        arms[arm], seen[arm], metas[arm] = results, exposed, meta

    # Comparing across a different --workers or temperature is meaningless:
    # both change results on their own. Refuse rather than print a wrong band.
    knobs = {(m["workers"], m["temperature"], m["model"]) for m in metas.values()}
    if len(knobs) > 1:
        raise SystemExit(f"runs differ in workers/temperature/model: {sorted(knobs)}. "
                         "Those are independent variables; rerun to match them.")

    if args.ref not in arms:
        raise SystemExit(f"reference arm {args.ref!r} not among {sorted(arms)}")
    ref = arms[args.ref]
    tasks = sorted(ref)

    # A run file still being written parses fine and yields a smaller task set.
    # Folding it in would understate that arm's count with no error at all, so
    # compare the task sets rather than trusting the file to be finished.
    for arm, results in arms.items():
        if set(results) != set(ref):
            raise SystemExit(
                f"{arm} covers {len(results)} tasks, {args.ref} covers "
                f"{len(ref)} -- is that run still in progress?")

    order = [args.ref] + sorted(a for a in arms if a != args.ref)
    print(f"{len(tasks)} tasks   ref={args.ref}   "
          f"workers={metas[args.ref]['workers']} "
          f"temp={metas[args.ref]['temperature']} "
          f"model={metas[args.ref]['model']}\n")

    print(f"{'arm':10}{'pass':>6}{'vs ref':>8}{'lost':>6}{'gained':>8}"
          f"{'churn':>7}{'saw hint':>10}")
    for arm in order:
        cur = arms[arm]
        lost = [t for t in tasks if ref[t] and not cur[t]]
        gained = [t for t in tasks if cur[t] and not ref[t]]
        n = sum(cur.values())
        print(f"{arm:10}{n:>6}{n - sum(ref.values()):>+8}{len(lost):>6}"
              f"{len(gained):>8}{len(lost) + len(gained):>7}{len(seen[arm]):>10}")

    # The effective denominator. A task where no SQL ever errored never saw the
    # hint, so no rewording of it can change that task's outcome; counting it
    # in the rate dilutes every effect. If the arms disagree on this set, the
    # runs are not comparable and the band below is not meaningful.
    exposures = [seen[a] for a in order]
    union = set().union(*exposures)
    identical = all(s == exposures[0] for s in exposures)
    print(f"\nsaw the hint: {len(union)}/{len(tasks)} tasks"
          f"   identical across arms: {identical}")
    if not identical:
        print("  WARNING: the arms differ on which tasks ever saw the hint. "
              "Effects are not comparable across a moving denominator.")

    unstable = [t for t in tasks if len({arms[a][t] for a in order}) > 1]
    stray = sorted(t for t in unstable if t not in union)
    print(f"unstable: {len(unstable)}/{len(union)} of the tasks that saw it "
          f"({len(unstable) / len(union):.0%})" if union else "")
    if stray:
        # At temperature 0 the hint is the only thing that differs between
        # arms, so a task that flips without ever seeing it means something
        # else is nondeterministic and the whole comparison is void.
        print(f"  🔴 flipped WITHOUT ever seeing the hint: {stray}. "
              "Something other than the wording is moving; do not read the band.")

    print(f"\n{'task':10}" + "".join(f"{a[:8]:>9}" for a in order))
    for t in unstable:
        print(f"{t:10}" + "".join(f"{('P' if arms[a][t] else '.'):>9}" for a in order))

    def disagree(a: str, b: str) -> int:
        return sum(1 for t in tasks if arms[a][t] != arms[b][t])

    print("\npairwise disagreement (tasks where the two arms differ)")
    print(f"{'':10}" + "".join(f"{a[:8]:>9}" for a in order))
    for a in order:
        print(f"{a:10}" + "".join(f"{disagree(a, b):>9}" for b in order))

    family = [a for a in NOISE_FAMILY if a in arms]
    if len(family) < 2:
        print("\nnoise band: not enough of the noise family present to size it")
        return
    inner = [disagree(a, b) for a, b in itertools.combinations(family, 2)]
    lo, hi = min(inner), max(inner)
    print(f"\nnoise band over {family}: pairwise disagreement {lo}..{hi} tasks")
    if len(family) < len(NOISE_FAMILY):
        missing = [a for a in NOISE_FAMILY if a not in arms]
        print(f"  INCOMPLETE -- still to run: {', '.join(missing)}. "
              "The band can only widen, so any verdict below is provisional.")
    for arm in sorted(set(arms) - set(family)):
        d = disagree(args.ref, arm)
        verdict = f"inside noise" if d <= hi else f"above the band by {d - hi}"
        print(f"  {args.ref} vs {arm:10} disagreement={d:<3} {verdict}")


if __name__ == "__main__":
    main()
