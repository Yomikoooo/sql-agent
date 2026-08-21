"""Reproducibility tests for `runner.build_jobs`.

Why this file exists: `--seed` was present, parsed and passed through, and did
nothing, because further downstream a `random.shuffle()` call used the global
RNG.  Nothing in that chain raised; `--seed 0` and `--seed 7` both looked fine.

A bug like this is only caught by running the same input twice and comparing.
Reading the code does not find it -- taken one at a time, both lines are right.
"""
import random

from sql_agent import runner

TASKS = [{"id": f"sql_{i:03d}"} for i in range(1, 9)]


def test_same_seed_gives_the_same_order():
    assert runner.build_jobs(TASKS, 3, seed=0) == runner.build_jobs(TASKS, 3, seed=0)


def test_different_seeds_give_different_orders():
    """The necessary counterpart: with only the test above, an implementation
    that never shuffles passes too -- it also gives the same order twice.
    """
    assert runner.build_jobs(TASKS, 3, seed=0) != runner.build_jobs(TASKS, 3, seed=7)


def test_the_global_rng_cannot_reach_the_job_order():
    """This is the test that actually pins the bug down.

    With a local RNG the job order depends only on `seed`.  The earlier code
    made it depend on the state of the global RNG as well, and anything can
    move that state -- any imported library, any earlier `random.*` call -- so
    the same seed gave different orders in different contexts.

    The method: move the global RNG to two completely different positions on
    purpose, and require the order to come out identical.
    """
    random.seed(1)
    first = runner.build_jobs(TASKS, 3, seed=0)

    random.seed(999)
    for _ in range(10):          # advance it further so the state really differs
        random.random()
    second = runner.build_jobs(TASKS, 3, seed=0)

    assert first == second, "the global RNG reached the job order, so --seed does nothing"


def test_every_task_appears_exactly_trials_times():
    """Shuffling must not drop or duplicate jobs along the way.

    `shuffle` works in place, so `jobs = random.shuffle(jobs)` sets it to None
    -- but that raises.  The dangerous case is a count that comes out wrong
    without raising anything.
    """
    jobs = runner.build_jobs(TASKS, 3, seed=0)
    assert len(jobs) == len(TASKS) * 3
    for task in TASKS:
        assert sum(1 for t, _ in jobs if t["id"] == task["id"]) == 3
    assert sorted(trial for _, trial in jobs) == sorted([0, 1, 2] * len(TASKS))


# ---------------------------------------------------------------------------
# The isolation boundary of `run_one`.
#
# Its contract, stated in its own docstring, is that it never raises: one task
# blowing up must not take down the whole eval and waste the API spend of the
# dozens of tasks before it.  The original try block only wrapped `run_agent`,
# so a throw from grading or from building the record went straight through
# `future.result()` and ended the run.
#
# Faults have to be injected stage by stage.  Testing only "the agent crashed"
# leaves the path where the grader crashes unexercised -- and that is the more
# dangerous one, because the grader is the measuring instrument: when it breaks,
# every number it produced stops meaning anything.
# ---------------------------------------------------------------------------
import pytest

TASK = {"id": "sql_001", "question": "how many users?", "gold_sql": "SELECT 1",
        "difficulty": "easy", "tags": ["count"], "check": "result"}


class _FakeResult:
    """A stand-in for AgentResult, complete enough for run_one."""
    text = "42 users."
    stop_reason = "done"
    turns = 3
    tool_calls = 2
    submitted_sql = "SELECT COUNT(*) FROM users"
    last_sql = "SELECT COUNT(*) FROM users"
    total_input_tokens = 1234
    total_output_tokens = 56
    total_cache_hit_tokens = 0
    elapsed = 1.5
    truncated_turns = 0
    messages = [{"role": "user", "content": "how many users?"}]


def _boom(*_a, **_k):
    raise RuntimeError("boom")


def test_a_crashing_retriever_does_not_escape(monkeypatch):
    monkeypatch.setattr(runner, "run_agent", lambda *a, **k: _FakeResult())
    rec = runner.run_one(TASK, 0, None, retriever=_boom)
    assert rec["failed_stage"] == "retrieve"
    assert rec["stop_reason"] == "crash" and rec["passed"] is False


def test_a_crashing_agent_does_not_escape(monkeypatch):
    monkeypatch.setattr(runner, "run_agent", _boom)
    rec = runner.run_one(TASK, 0, None)
    assert rec["failed_stage"] == "agent"
    assert rec["messages"] == []            # the agent never ran, so there is no trajectory


def test_a_crashing_verifier_does_not_escape(monkeypatch):
    """The core of the fix: this used to end the whole eval run."""
    monkeypatch.setattr(runner, "run_agent", lambda *a, **k: _FakeResult())
    monkeypatch.setattr(runner, "verify_sql", _boom)
    rec = runner.run_one(TASK, 0, None)
    assert rec["failed_stage"] == "verify"
    assert "RuntimeError" in rec["reason"]


def test_a_crash_after_the_agent_keeps_the_trajectory(monkeypatch):
    """API spend already made must not be thrown away because something downstream crashed.

    By the time grading crashes the agent turn has run and been paid for, and
    the messages are the raw material for supervised fine-tuning (see the
    module docstring of `runner.py`).  The old crash record zeroed those fields
    out, which forced the task to be run again.
    """
    monkeypatch.setattr(runner, "run_agent", lambda *a, **k: _FakeResult())
    monkeypatch.setattr(runner, "verify_sql", _boom)
    rec = runner.run_one(TASK, 0, None)

    assert rec["messages"] == _FakeResult.messages
    assert rec["input_tokens"] == 1234
    assert rec["turns"] == 3
    assert rec["submitted_sql"] == "SELECT COUNT(*) FROM users"


def test_the_stage_name_is_the_stage_that_actually_failed(monkeypatch):
    """`stage` has to be assigned before the work, not after.

    Assigned after, a crash records the name of the previous stage, and an
    error pointing at the wrong place costs more time than no error at all.
    """
    monkeypatch.setattr(runner, "run_agent", lambda *a, **k: _FakeResult())
    monkeypatch.setattr(runner, "verify_sql", _boom)
    assert runner.run_one(TASK, 0, None)["failed_stage"] != "agent"


def test_the_happy_path_still_records_no_stage(monkeypatch):
    """The positive case is required: with only the crash paths, an
    implementation that always reports a failed_stage passes too.
    """
    monkeypatch.setattr(runner, "run_agent", lambda *a, **k: _FakeResult())
    rec = runner.run_one(TASK, 0, None)
    assert rec["failed_stage"] is None
    assert rec["stop_reason"] == "done"
