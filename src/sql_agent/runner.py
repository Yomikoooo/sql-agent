"""Batch eval: repeated trials, a thread pool, JSONL traces, stratified metrics.

This is also a data pipeline, not only a measurement tool: the complete message
lists in results/*.jsonl are the raw material for rejection sampling. Messages
are therefore stored in full even when nothing reads them yet -- regenerating
them costs real API spend.

    python runner.py                          # default model, 3 trials
    python runner.py --trials 1 --workers 3   # quick smoke test
    python runner.py --model qwen2.5:1.5b     # measure another model's baseline
"""
import argparse
import hashlib
import json
import logging
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from statistics import mean

from .agent import SYSTEM_PROMPT, run_agent
from .cache import CACHE, ENABLED as CACHE_ON
from .llm import resolve_url
from .tools import SQL_ERROR_HINT, SQL_ERROR_HINT_VARIANT, TOOLS_SCHEMA
from .redact import redact_record
from .tracing import rollup, trace
from .verify import verify_behavior, verify_sql

from . import retrieval
from . import paths

logger = logging.getLogger(__name__)

# Anything that can change the result has to end up in the record, or in three
# months results/ cannot say which version produced a number.
#
# Two hashes rather than one, because model-visible text lives in two places:
#   - SYSTEM_PROMPT, which embeds the schema and the prompt suffix, so changing
#     the schema sample mode moves this hash too, which is the intent
#   - the tool descriptions in TOOLS_SCHEMA, which is exactly the kind of text
#     an experiment changes next. Recording only the first would miss it.
def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


PROMPT_HASH = _hash(SYSTEM_PROMPT)
TOOLS_HASH = _hash(json.dumps(TOOLS_SCHEMA, sort_keys=True, ensure_ascii=False))

# Worker count is a hidden independent variable, not a performance knob.
# Measured: same code, same model, temperature 0, and moving --workers from 5
# to 1 turned one task from "exhausted its 8-turn budget" into "finished in 3".
# Each worker count is reproducible token for token *within itself*; across
# counts it differs. The likely cause is continuous batching in the server:
# different batch composition changes the order of floating-point reductions,
# which perturbs logits, which flips argmax on near-ties.
#
# Temperature 0 guarantees "take the most likely token". It does not guarantee
# that the probabilities are computed to the same bits every time.
#
# main() fills in the real value. Calling run_one directly leaves it None,
# meaning unknown -- do not paper over that with a default, or the record
# will state something untrue.
WORKERS: int | None = None

DB = str(paths.DB_PATH)
RESULTS_DIR = paths.RESULTS_DIR

RETRIEVERS = {
    "overlap": retrieval.suffix_overlap,     
    "vector":  retrieval.suffix_vector,
    "random":  retrieval.suffix_random,
}

# ================================================================ C1
def run_one(task: dict, trial: int, model: str | None,
            temperature: float = 0.0, retriever=None) -> dict:
    """Run one task once and return one complete record.

    Never raises. One task blowing up must not take the sweep with it: every
    task already completed was paid for.

    Two field names that are easy to confuse: AgentResult.total_input_tokens
    is input_tokens in the record, and AgentResult.text is answer.
    """
    base = {
        "task_id": task["id"],
        "trial": trial,
        "model": model or "default",
        "difficulty": task.get("difficulty", "unknown"),
        "tags": task.get("tags", []),
        "question": task["question"],
        # .get, because behaviour tasks have no gold_sql and would KeyError
        "check": task.get("check", "result"),
        "gold_sql": task.get("gold_sql"),
        "expect": task.get("expect"),
        # Which retriever ran belongs in the record. This is where a command
        # line flag beats an environment variable: the file says what ran,
        # instead of leaving you to guess what the shell had set that day.
        "retriever": None,
        # The endpoint goes in too: a renamed --model pointing at an unchanged
        # endpoint once produced a whole run of fake failures
        "api_url": resolve_url(model),
        # Fields are only ever added, never changed, so older results still read
        "prompt_hash": PROMPT_HASH,
        "tools_hash": TOOLS_HASH,
        # The error hint is not part of TOOLS_SCHEMA, so tools_hash does not
        # cover it. An independent variable has to be recorded explicitly
        # rather than hoping some other hash happens to include it.
        "error_hint_variant": SQL_ERROR_HINT_VARIANT,
        "error_hint_hash": _hash(SQL_ERROR_HINT),
        # See the WORKERS note above: an independent variable, not a knob
        "workers": WORKERS,
        "temperature": temperature,
    }

    tr = None            # pre-bound so the trace can still be written on a crash
    result = None        # same, with a harder reason: if scoring crashes, this
                         # run's API spend is already sunk and the messages,
                         # token counts and trace are in hand. Dropping them
                         # forces the task to be paid for twice.
    stage = "setup"      # assigned *before* each stage, never after. Assigned
                         # after, a crash would record the name of the previous
                         # stage -- precisely when accuracy matters most.
    try:
        # exclude_id is mandatory: retrieving a dev task against the dev pool
        # returns itself first, and it carries its own gold SQL -- that is
        # handing the model the answer. For test tasks the id is not in the
        # pool anyway, so passing it always is harmless and keeps one path.
        # Retrieval goes inside the trace: which examples were selected is the
        # thing a post-mortem needs and the thing most easily lost.
        stage = "retrieve"
        with trace("agent.run", task_id=task["id"], trial=trial,
                   model=model, temperature=temperature) as tr:      # noqa: F841 -- used in except
            suffix = retriever(task["question"], exclude_id=task["id"]) if retriever else ""
            base["retriever"] = getattr(retriever, "__name__", None)
            base["suffix_len"] = len(suffix)     # recorded in case the effect tracks length
            stage = "agent"
            result = run_agent(task["question"], model=model, temperature=temperature,
                               prompt_suffix=suffix)

        # Score the query the agent explicitly submitted, falling back to
        # last_sql only if it never did -- and record that the fallback
        # happened, because the fallback rate is itself a metric: a high one
        # means the model is not following the protocol, which is a signal to
        # change the prompt rather than something to swallow quietly.
        stage = "verify"
        answer_sql = result.submitted_sql or result.last_sql

        if task.get("check") == "behavior":
            # Behaviour tasks score submitted_sql, never the fallback: what
            # they check *is* whether it submitted, so substituting a fallback
            # would erase the criterion being tested.
            v = verify_behavior(result.submitted_sql, result.text,
                                task["expect"], result.tool_calls)
        else:
            v = verify_sql(answer_sql or "", task["gold_sql"], DB,
                           task.get("ordered", False))

        stage = "record"
        return {
            **base,
            # Added fields only; nothing existing moved, so old results read
            "trace": tr.to_dict(),
            "rollup_ms": rollup(tr),
            "passed": v.passed,
            "level": v.level,
            "reason": v.reason,
            "failed_stage": None,        # present on success too, so analysis has one shape
            "agent_sql": answer_sql,
            "submitted_sql": result.submitted_sql,
            "last_sql": result.last_sql,
            "used_fallback": result.submitted_sql is None,
            "answer": result.text,
            "stop_reason": result.stop_reason,
            "turns": result.turns,
            "tool_calls": result.tool_calls,
            "input_tokens": result.total_input_tokens,
            "output_tokens": result.total_output_tokens,
            "cache_hit_tokens": result.total_cache_hit_tokens,
            "elapsed": result.elapsed,
            "truncated_turns": result.truncated_turns,
            "messages": result.messages,     # requires the agent to return them
        }

    except Exception as e:
        # The isolation boundary has to reach all the way through building the
        # record, not stop at run_agent. It used to stop there, so an exception
        # from the verifier or from to_dict() went straight through
        # future.result() and aborted the whole sweep -- while this function's
        # contract says it never raises.
        logger.exception("run_one failed at stage %r on %s trial %d",
                         stage, task["id"], trial)
        return _failure_record(base, tr, result, stage, e)


def _failure_record(base: dict, tr, result, stage: str, exc: Exception) -> dict:
    """The record written when any stage crashes. Salvage everything possible.

    failed_stage has to be recorded, because "the agent crashed" and "the
    scorer crashed" look identical in the metrics and mean opposite things:
    the first is a fault in the thing being measured, the second a fault in the
    measuring instrument. Summing two events of opposite meaning into one
    bucket is how a metric stops meaning anything, and a broken instrument has
    to be fixable first, so it must be countable on its own.

    The trace is at its most useful here: it shows whether the failure was in
    retrieval, in a particular model turn, or in a tool.
    """
    r = result
    return {
        **base,
        "trace": tr.to_dict() if tr else None,
        "rollup_ms": rollup(tr) if tr else {},
        "passed": False,
        "level": "fail",
        "failed_stage": stage,
        "reason": f"{stage} crashed: {type(exc).__name__}: {exc}",
        "stop_reason": "crash",
        # These hold real values when the agent finished and a later stage
        # crashed. Zeroing them throws away a trajectory already paid for.
        "turns": r.turns if r else 0,
        "tool_calls": r.tool_calls if r else 0,
        "input_tokens": r.total_input_tokens if r else 0,
        "output_tokens": r.total_output_tokens if r else 0,
        "cache_hit_tokens": r.total_cache_hit_tokens if r else 0,
        "elapsed": r.elapsed if r else 0.0,
        "agent_sql": (r.submitted_sql or r.last_sql) if r else None,
        "submitted_sql": r.submitted_sql if r else None,
        "last_sql": r.last_sql if r else None,
        "used_fallback": (r.submitted_sql is None) if r else False,
        "truncated_turns": r.truncated_turns if r else 0,
        "answer": r.text if r else "",
        "messages": r.messages if r else [],
    }


# ================================================================ job order
def build_jobs(tasks: list[dict], trials: int, seed: int) -> list[tuple[dict, int]]:
    """Flatten tasks x trials into a job list and shuffle it.

    The shuffle is required. Flattened, the list is grouped by task
    (sql_001 x3, sql_002 x3, ...), and a failure that happens *in time* -- a
    server restart, a rate limit -- lands on a contiguous stretch of it. Tasks
    are ordered by difficulty, so one such episode put all 21 errors on hard
    tasks, which makes even "recompute the score excluding errors" unsafe.
    Shuffled, at least the damage is spread evenly.

    Shuffle exactly once, with a local RNG. There used to be a second,
    unseeded `random.shuffle(jobs)` after the seeded one, which voided it
    entirely: --seed was accepted and passed through and had no effect, without
    error. Two runs at one seed produced different orders, so an entire class
    of time-correlated failures could never be reproduced -- and job order is a
    hidden variable of the same kind as worker count, which is already known to
    change results.

    It is a separate function for exactly one reason: so it can be tested on
    its own. Buried in main(), checking "same seed, same order" would have
    meant running a full eval, which costs money.
    """
    jobs = [(task, trial) for task in tasks for trial in range(trials)]
    random.Random(seed).shuffle(jobs)
    return jobs


# ================================================================ C5
def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    # p is on a 0-100 scale, not 0-1. Written as int(len(s) * p), p=95 gives
    # len(s)*95, gets clamped to len(s)-1, and silently returns the maximum
    # every time.
    idx = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return s[idx]


def _rate(records: list[dict]) -> float:
    return sum(r["passed"] for r in records) / len(records) if records else 0.0


def summarize(records: list[dict]) -> dict:
    """Reduce a list of records to metrics.

    Every breakdown is computed from the same records. Re-running to get a
    different grouping costs n times as much and yields data with different
    randomness, which is then not comparable with the first pass.
    """
    by_difficulty = defaultdict(list)
    by_tag = defaultdict(list)
    by_task = defaultdict(list)
    stop_reasons = defaultdict(int)          # counts, so int rather than list

    for r in records:
        by_difficulty[r["difficulty"]].append(r)
        by_task[r["task_id"]].append(r)
        stop_reasons[r["stop_reason"]] += 1  # the field is singular
        for tag in r["tags"]:
            by_tag[tag].append(r)

    # error and crash are infrastructure faults, not wrong answers. Split out.
    completed = [r for r in records if r["stop_reason"] not in ("error", "crash")]
    n_error = len(records) - len(completed)

    always_pass, always_fail, flaky = [], [], []
    for task_id, rs in by_task.items():
        n_passed = sum(r["passed"] for r in rs)
        if n_passed == len(rs):
            always_pass.append(task_id)
        elif n_passed == 0:
            always_fail.append(task_id)
        else:
            flaky.append(task_id)

    return {
        "n_records": len(records),
        "n_tasks": len(by_task),
        "success_rate": _rate(records),
        "by_difficulty": {k: _rate(v) for k, v in by_difficulty.items()},
        "by_tag": {k: _rate(v) for k, v in by_tag.items()},
        "stop_reasons": dict(stop_reasons),
        "always_pass": sorted(always_pass),
        "always_fail": sorted(always_fail),
        "flaky": sorted(flaky),
        "flaky_rate": len(flaky) / len(by_task) if by_task else 0.0,
        "loose_count": sum(1 for r in records if r["level"] == "loose"),
        # Counted separately: mixing infrastructure faults into the score
        # contaminates the conclusion
        "error_count": n_error,
        "success_rate_completed": _rate(completed),
        # How often submit_answer was skipped: protocol compliance
        "fallback_count": sum(1 for r in records if r.get("used_fallback")),
        # Runs whose output hit max_tokens. A truncated turn may hold partial
        # SQL, which is measurement contamination and has to be visible rather
        # than buried in a log line.
        "truncated_runs": sum(1 for r in records if r.get("truncated_turns")),
        "turns_mean": mean([r["turns"] for r in records]) if records else 0,
        "tool_calls_mean": mean([r["tool_calls"] for r in records]) if records else 0,
        "in_tokens_mean": mean([r["input_tokens"] for r in records]) if records else 0,
        "in_tokens_p95": percentile([r["input_tokens"] for r in records], 95),
        "elapsed_mean": mean([r["elapsed"] for r in records]) if records else 0,
        "elapsed_p95": percentile([r["elapsed"] for r in records], 95),
    }


# ================================================================ C6
def print_report(s: dict, run_id: str) -> None:
    print(f"\n{'=' * 62}\nrun {run_id}   {s['n_tasks']} tasks x "
          f"{s['n_records'] // max(s['n_tasks'], 1)} trials = {s['n_records']} runs")
    print("=" * 62)
    print(f"success rate        {s['success_rate']:.1%}")
    print(f"flaky rate          {s['flaky_rate']:.1%}   "
          f"<- noise floor: ignore gains smaller than this")
    print(f"loose verdicts      {s['loose_count']}   <- spot-check these by hand")
    if s.get("truncated_runs"):
        print(f"truncated output   {s['truncated_runs']}   "
              f"<- hit max_tokens; those turns may have produced partial SQL")
    if s.get("error_count"):
        print(f"infra errors        {s['error_count']}   "
              f"<- NOT model failures. rate excluding them: "
              f"{s['success_rate_completed']:.1%}")
    print(f"no submit_answer    {s.get('fallback_count', 0)}   "
          f"<- fell back to last_sql; if this is high, fix the prompt")

    print("\nby difficulty")
    for k in ("easy", "medium", "hard"):
        if k in s["by_difficulty"]:
            print(f"  {k:<10} {s['by_difficulty'][k]:>6.0%}")

    print("\nby tag (worst first)")
    for tag, rate in sorted(s["by_tag"].items(), key=lambda kv: kv[1])[:10]:
        print(f"  {tag:<14} {rate:>6.0%}")

    print(f"\nstop_reason         {s['stop_reasons']}")
    print(f"turns    mean {s['turns_mean']:.1f}   tool_calls mean {s['tool_calls_mean']:.1f}")
    print(f"in_tok   mean {s['in_tokens_mean']:>8,.0f}   p95 {s['in_tokens_p95']:>8,.0f}")
    print(f"elapsed  mean {s['elapsed_mean']:>8.1f}s  p95 {s['elapsed_p95']:>8.1f}s")

    if s["always_fail"]:
        print(f"\nalways FAIL ({len(s['always_fail'])})  <- read these traces first")
        print("  " + ", ".join(s["always_fail"]))
    if s["flaky"]:
        print(f"\nflaky ({len(s['flaky'])})")
        print("  " + ", ".join(s["flaky"]))


# ================================================================ 


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="override the default model")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--workers", type=int, default=5,
                        help="keep it small: the API rate-limits you")
    parser.add_argument("--tasks", default="dev.json")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 for eval (measure ability). raise it only when sampling training data")
    parser.add_argument("--seed", type=int, default=0,
                        help="shuffle seed; same seed = same job order")
    parser.add_argument("--retriever", choices=sorted(RETRIEVERS), help="prompt")
    args = parser.parse_args()

    global WORKERS
    WORKERS = args.workers      # independent variable; see the note above

    tasks = json.loads((paths.TASKS_DIR / args.tasks).read_text(encoding="utf-8"))

    CACHE.load()

    if args.trials == 1:
        print("--trials 1 cannot measure noise: temperature 0 is greedy "
              "decoding, not a guarantee of run-to-run identity.\n"
              "    Measured: 2.6% of dev tasks are still flaky at temperature "
              "0. Concluding anything improved needs trials >= 3.")

    # Different worker counts give different results, reproducibly so within
    # each count. Comparing across runs means matching this first.
    print(f"workers={args.workers} is an independent variable: "
          f"match it before comparing runs")

    if args.trials > 1 and CACHE_ON:
        sys.exit("refuse: --trials > 1 is incompatible with LLM_CACHE=1. The "
                 "cache replays trial 1 for trials 2 and 3, which forces the "
                 "flaky rate to 0 no matter what the model does.")


    RESULTS_DIR.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")   # no ":", filenames forbid it
    out_path = RESULTS_DIR / f"{run_id}.jsonl"

    jobs = build_jobs(tasks, args.trials, args.seed)
    print(f"{len(tasks)} tasks x {args.trials} trials = {len(jobs)} runs "
          f"({args.workers} workers, temp={args.temperature}) -> {out_path.name}")

    retriever = RETRIEVERS[args.retriever] if args.retriever else None
    if retriever:
        retriever(jobs[0][0]["question"])

    records = []
    # Written as it goes, so a Ctrl+C halfway keeps everything before it
    with open(out_path, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        # submit returns a Future immediately and does not block
        futures = [pool.submit(run_one, task, trial, args.model, args.temperature, retriever)
                   for task, trial in jobs]
        # as_completed yields in completion order, not submission order
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()        # run_one absorbs its own exceptions
            record["run_id"] = run_id
            records.append(record)
            f.write(json.dumps(redact_record(record), ensure_ascii=False) + "\n")
            f.flush()                       # or a crash loses the buffer
            print(f"\r  {i}/{len(jobs)}", end="", flush=True)

    summary = summarize(records)
    print_report(summary, run_id)

    if CACHE_ON:
        print(f"cache               {CACHE.stats()}")

    (RESULTS_DIR / f"{run_id}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ntraces  -> {out_path}")
    print(f"summary -> {out_path.with_suffix('.summary.json').name}")

    CACHE.save()


if __name__ == "__main__":
    # WARNING, because INFO drowns in one httpx line per request.
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    main()
