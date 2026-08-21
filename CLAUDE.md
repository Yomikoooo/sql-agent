# CLAUDE.md

A natural-language data analysis agent. **The hard part is not generating SQL, it is
knowing when the answer is wrong** -- a confident, well-formed, incorrect answer costs
far more than an obviously broken one. Most of the design below exists to make wrong
answers detectable rather than to make right ones more likely.

## Layout

Every module lives under `src/sql_agent/` and the project installs as a package
(`pip install -e .`). Four console scripts are the supported entry points and work
from any directory once installed:

| Command | Does |
|---|---|
| `sql-agent-seed` | Build `data/*.db` from a fixed seed |
| `sql-agent-check-tasks` | Self-check the task files |
| `sql-agent-eval` | Run the evaluation harness |
| `sql-agent-serve` | Start the HTTP service |

🔴 **The project root is resolved in exactly one place**, `src/sql_agent/paths.py`
(`SQLAGENT_HOME` → search upward for `pyproject.toml` → search upward from cwd → raise).

**New code must never write `Path(__file__).parent / "some_data_dir"`.** When a module
moves, that expression silently points somewhere new, creates an empty directory, and
reports nothing.

## Four things to know before changing anything

**1. `tools.get_schema()` is shared by every historical run.** Its output is spliced
into `SYSTEM_PROMPT` at `src/sql_agent/agent.py:96`. The default `SCHEMA_SAMPLE_MODE`
of `"raw"` **must keep behaviour byte-for-byte identical** -- changing it moves the
baseline under every number ever recorded, **and produces no symptom**. Verify any
change by diffing against the old output byte by byte.

**2. `SYSTEM_PROMPT` is assembled at import time.** After changing `tools.DB_PATH` or
`SCHEMA_SAMPLE_MODE` you **must** `importlib.reload(agent)`, or the run still uses the
schema captured at first import.

**3. The database connection is read-only (`mode=ro`), and that is a diagnostic, not
just a safeguard.** A read-only connection fails loudly when the file is missing. The
default connection **silently creates an empty database**, turning "the file is not
there" into "the table does not exist" and sending the search in the wrong direction.

**4. `data/*.db` is gitignored.** Rebuild it with `sql-agent-seed`; the seed is fixed,
so the rebuild is identical.

## Commands

```bash
pip install -e .
```

```bash
sql-agent-seed && sql-agent-check-tasks
```

```bash
sql-agent-eval --tasks dev.json --trials 3
```

```bash
python -m pytest -q -rs
```

⚠️ **`-rs` is not optional.** Tests skipped by `skipif` are green in CI. Without `-rs`,
"it passed" and "it never ran" are the same signal.

## Comments are English, and they explain choices

Comments belong in the code when they answer **"why is it not written the other way?"**
-- why this default, why this order, what was tried and rolled back and with which
numbers. Comments that merely restate what the code does are noise.

⚠️ **A comment may only point at something inside this repository**, preferably at a
test. Tests are the documentation that cannot go stale silently.

⚠️ Measure comment density in **lines and ratio, never characters**. Ratio moving is not
by itself good or bad: a module that opens with an explanation of a concept should come
down, a module that is almost entirely decision criteria may legitimately go up.

## Measurement discipline

The evaluation harness is the point of this project, so these rules outrank convenience.

**① Done means evidence, not "implemented".** Every claim needs a command that
reproduces it.

**② 🔴 The measuring tool outranks the thing being measured.** When scoring, metrics, or
an error message look suspicious, **stop and fix the tool first**. Optimising against a
broken instrument produces a clean, entirely meaningless improvement curve. Six of the
recorded failures in this project were the measurement being wrong, not the agent.

**③ Ask of every metric: does it add together two events with opposite meanings?** This
has happened five times here. `no submit_answer` merged "correctly declined to answer an
ambiguous question" with "stuck in a loop". The first injection detector compared
`base.text != r.text` and scored a difference in wording as a change in behaviour.

**④ Verify the change took effect before running anything.** Print the prompt after
editing it; assert the flag actually landed. 🔴 Without this step, "triggered 0 times"
and "the code never ran" look identical in the output.

**⑤ Confirm the N runs really happened N times.** Caching (`LLM_CACHE=1`), memoisation
and batch de-duplication all turn N into 1 with no symptom.

**⑥ 🔴 A conclusion must be reproducible from code.** Three things, none optional: a
script that reruns it, the raw record on disk (git hash, model, temperature, full input,
prompt hash, complete trajectory), and a pinned code version (commit recorded at the top
of the report). Printing to a terminal is not data. Storing only the summary table means
the question cannot be re-asked from a different angle later.

**⑦ The noise floor is ±1.3%**, measured with a placebo arm over 78 tasks. **An
improvement smaller than that may not be claimed.** When the largest available gain is
below the noise floor, switch to targeted counting plus side-effect checks -- a bigger
sample does not rescue it.

## Defaults

| | Value | Where |
|---|---|---|
| `temperature` | **0** -- evaluation measures capability, not luck | `src/sql_agent/llm.py:217` |
| Exact cache | **off**, enabled with `LLM_CACHE=1` | `src/sql_agent/cache.py:9` |
| `SCHEMA_SAMPLE_MODE` | `"raw"` | `src/sql_agent/tools.py` |
| Model routing | a name containing a colon goes to a local Ollama, otherwise the hosted API | `src/sql_agent/llm.py` |
| Prompt append hook | `SQLAGENT_PROMPT_SUFFIX` env var | `src/sql_agent/agent.py:100` |

⚠️ **`SQLAGENT_PROMPT_SUFFIX` is worth copying as a pattern**: an environment variable
plus a subprocess lets several prompt variants run isolated from each other, with no
file edits and no race.

## Decided against, with the reason

| Not doing | Why |
|---|---|
| Semantic cache | Measured, then dropped: the useful threshold sat at 0.9706 and the gain did not justify it |
| Embedding retrieval | Zero-dependency word overlap 60.5% vs vector 61.8% -- **it did not earn its dependency** |
| Constrained decoding / JSON mode | **0 of 30 recorded failures were syntax errors**, so the ceiling on the gain is 0 |
| Multi-tenant filtering | There are no tenants, so it cannot be verified |
| Replacing the business database | Every gold answer would need rewriting and all historical numbers would be void |
| K8s, external queues, multi-provider routing | No users and no load evidence |
