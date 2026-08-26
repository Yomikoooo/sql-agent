#!/usr/bin/env python3
"""Copy the arm identifiers from each run's trajectory file onto its summary.

Why this is needed: `results/*.jsonl` is gitignored -- the messages carry
database rows -- so `*.summary.json` is the only artefact that ships. It has
carried `always_pass` / `always_fail` all along, which at one trial per task is
the complete per-task result, so every task-level conclusion in this project
can be recomputed from what ships. What it did not carry, until 2026-08-25,
was any statement of *which arm produced it*.

A reader could therefore recompute a comparison correctly and still not know
what was being compared. That is the same defect that made one arm of the
retrieval experiment permanently unreproducible: the data survived, the
configuration did not.

runner.py now writes the config at summary time. This script backfills the
runs that predate that change, reading the trajectory files while they are
still on disk. Fields absent from an old run stay null -- that is information,
not a gap to paper over: it marks exactly which runs cannot be pinned.

Usage:
    python scripts/backfill_summary_config.py                    # dry run
    python scripts/backfill_summary_config.py --write
    python scripts/backfill_summary_config.py --also DIR [DIR…]  # extra jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sql_agent import paths  # noqa: E402

FIELDS = ("model", "temperature", "workers", "retriever",
          "error_hint_variant", "error_hint_hash",
          "schema_sample_mode", "schema_hash",
          "prompt_hash", "tools_hash", "api_url")


def index_trajectories(dirs: list[Path]) -> dict[str, Path]:
    """run_id -> trajectory path. The first directory given wins on a tie."""
    found: dict[str, Path] = {}
    for d in dirs:
        for p in sorted(d.glob("*.jsonl")):
            found.setdefault(p.name[:19], p)
    return found


def config_of(path: Path) -> tuple[dict, list[str]] | None:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        return None
    cfg = {k: records[0].get(k) for k in FIELDS}
    mixed = sorted({k for k in FIELDS if len({r.get(k) for r in records}) > 1})
    return cfg, mixed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="actually rewrite the files")
    ap.add_argument("--also", nargs="*", type=Path, default=[],
                    help="extra directories to search for trajectory files")
    args = ap.parse_args()

    results = paths.RESULTS_DIR
    traj = index_trajectories([results, *args.also])

    done = skipped = missing = 0
    unpinned: list[str] = []
    for summary in sorted(results.glob("*.summary.json")):
        data = json.loads(summary.read_text(encoding="utf-8"))
        if "config" in data:
            skipped += 1
            continue
        run_id = summary.name[:19]
        src = traj.get(run_id)
        if src is None:
            missing += 1
            print(f"  🔴 {run_id}  no trajectory file; cannot be labelled")
            continue
        got = config_of(src)
        if got is None:
            missing += 1
            print(f"  🔴 {run_id}  trajectory file is empty")
            continue
        cfg, mixed = got
        # Rebuild the dict so config leads the file, matching runner.py's order.
        merged = {"config": cfg, "config_mixed_within_run": mixed, **data}
        if args.write:
            summary.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        done += 1
        if cfg["workers"] is None:
            unpinned.append(run_id)

    verb = "labelled" if args.write else "would label"
    print(f"\n  {verb} {done}   already had config {skipped}   unlabelled {missing}")
    if unpinned:
        print(f"  ⚠️  {len(unpinned)} of them record no `workers`: they predate the field, "
              f"so the concurrency they ran at is unknowable. That is now visible in the "
              f"file rather than only in the docs.")
    if not args.write:
        print("  (dry run — nothing written; add --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
