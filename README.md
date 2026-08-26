# SQL Data Analysis Agent

[![tests](https://github.com/Yomikoooo/sql-agent/actions/workflows/test.yml/badge.svg)](https://github.com/Yomikoooo/sql-agent/actions/workflows/test.yml)

A data analysis agent written from scratch: it takes a question in English,
writes SQL, runs it, and answers. No agent framework -- the loop, the tool
calls and the context management are all hand-written.

**The agent is not the point of this repository. The evaluation around it is,
along with the things that evaluation said that I did not want to hear.**

---

## The headline number, and why it is the lower one

| | Success rate | |
|---|---|---|
| `qwen2.5:3b` on **dev** (78 tasks, used over and over while tuning) | **55.8%** | not trustworthy, see below |
| `qwen2.5:3b` on **test** (24 tasks, never used for any decision) | **41.7%** | **the number worth quoting** |
| `deepseek-chat` on dev | ~100% | a ceiling; it measures nothing |

> **These numbers were corrected after publication.** Two grading defects
> were found in the task set: three questions say "ordered alphabetically"
> but were graded without an order, and `sql_030`'s gold query returns no
> rows, so any prediction returning nothing scored correct — including, in
> the run behind the published figure, a query with the wrong date bound.
>
> Nothing was re-run. `scripts/recount_published.py` applies the corrections
> to the SQL each run actually submitted, and `results/published/` holds
> those records, so the numbers can be checked rather than taken on trust.
>
> The dev figure fell from 56.4% to **55.8%** and its task count was 78, not
> 76. **The test figure did not move**: its three order-sensitive questions
> had failed for unrelated reasons, two of them by producing no SQL at all.
> A third suspected defect — `test_506`'s truncated question text and its
> gold filter on a condition the text never states — changes nothing either,
> because that filter removes no rows from this database.
>
> That asymmetry is not luck. The figure that moved is the one that was tuned
> against; the one that held is the one nothing was allowed to touch.

The starting point was 26.7%. Six rounds of prompt work took the dev set from
26.7% to 55.8%, and every one of those rounds chose what to keep by looking at
the dev number.

⚠️ Those two figures are **not on the same task set**: the starting point was
measured on 31 tasks and the end point on 78. Both are recounted under the
same grading, which the script shows, but a change of denominator sits inside
that comparison and no recount removes it.

On a held-out set of 24 tasks, locked from the day it was generated, the best
configuration scored **4 points below the baseline**.

**Those 14.1 points were overfitting.** It is measurable only because the test
set never took part in a single decision. Without it, the honest-sounding
claim would have been "I improved the success rate from 26.7% to 55.8%", which
reads far better and is not true.

This is the same phenomenon as overfitting in machine learning, with "training"
replaced by "tuning the prompt". Repeatedly choosing between options by looking
at the same tasks selects, in part, for whatever happens to work on those tasks.

---

## Retrieval-augmented few-shot: it works, and it does not improve what it looks like it improves

Four arms, dev set, `qwen2.5:3b`, temperature 0. The `random` arm is a placebo:
real tasks with real gold answers, the same format and the same number of
examples, differing only in being irrelevant to the question.

| Arm | Passed | Net change | Dependency |
|---|---|---|---|
| `base` | 36/76 = 47.4% | -- | -- |
| `vector` (embedding retrieval) | **47/76 = 61.8%** | **+11** | Ollama and a 270 MB model |
| `overlap` (word overlap) | 46/76 = 60.5% | +10 | **none, 20 lines** |
| `random` (placebo) | 40/76 = 52.6% | +4 | -- |
| `fixed` (one fixed pair of examples) | 44/76 = 57.9% | +8 | -- |

Two conclusions, and the second is the one that matters.

**The placebo is worth 4 of the 11 points.** Having examples at all does that
much. Retrieval is worth the remaining 7. Without the placebo arm, all 11
points would have been credited to retrieval.

**Word overlap ties with embeddings** (one task apart, inside the noise band)
while needing no dependency at all, so embeddings did not earn theirs on this
task. The two agree on only 49% of their top-2 picks and still tie, which says
the benefit comes from the coarse event of hitting one sufficiently similar
task, not from ranking well.

Stratifying by how similar each task is to its nearest neighbour in the pool
changes the conclusion again:

| Similarity to nearest neighbour | Tasks | `base` | `vector` | `random` |
|---|---|---|---|---|
| >= 0.93 | 12 | 4 | **11 (+7)** | 7 (+3) |
| 0.88 to 0.93 | 27 | 11 | **17 (+6)** | 12 (+1) |
| **< 0.88** | **33** | 20 | **18 (-2)** | 20 (+0) |

**All of the benefit sits on the tasks that have a near-twin in the pool. On
the largest group, the 33 tasks with no near-twin, it is negative.**

The most dramatic case: one task that four different prompt techniques had all
failed to fix flipped to a pass. It flipped because retrieval found another
task at similarity 0.927 whose question was nearly identical and whose gold SQL
had the same shape. The agent copied the structure and changed the aliases. It
did not learn anything.

**As a product feature this is correct behaviour** -- reusing the answer to a
question much like one already answered is exactly what retrieval is for.
**As a measurement of capability it is contamination.** The same number has to
be reported differently depending on which claim it is supporting.

---

## What was tried and did not work

| Tried | Result | |
|---|---|---|
| Numbered procedure, forcing verification before submission | net zero, reverted | The stated goal was reached (`run_sql` usage went 2/31 to 29/31) and the submission protocol collapsed (30/31 to 5/31) |
| Targeted SQL rules in the prompt | +2, inside the noise | The placebo showed that an equally long irrelevant addition moves it by ±1 too |
| Few-shot with correct examples | dev +7, test -1 | The overfitting above |
| Simple examples in the same format | **-4** | Refutes the hypothesis that examples work by demonstrating the format, and shows few-shot carries risk |

**Four different prompt techniques, aimed at one class of SQL error -- queries
needing two levels of aggregation -- all failed.** Three tasks fail under every
prompt configuration tried.

That is the strongest evidence that the ceiling here is not in the prompt, and
it is the starting point for the next stage.

The list of such tasks was revised twice, from five down to three, and the
revision is part of the conclusion: two of them were flipped by retrieval, and
in both cases the agent had copied the structure of a near-identical task
rather than learning the pattern. So the conclusion stands with a caveat --
prompting does not solve these, and retrieval can route around them when a
near-twin exists, which is reuse rather than capability.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # the code lives under src/, so install it
cp .env.example .env          # add an API key
sql-agent-seed                # build the database, fixed seed, reproducible
python -m pytest -q -rs       # 92 tests; -rs makes skips visible
sql-agent-check-tasks         # task set self-check: dev 76/76, test 24/24
```

The commands are console scripts declared in `pyproject.toml`, so they work
from any directory once the package is installed. Everything they read and
write is located relative to the checkout root, which `sql_agent/paths.py`
resolves once; `SQLAGENT_HOME` overrides it when the data lives elsewhere.

`-rs` is not optional: tests skipped by `skipif` are green in CI, so without it
"tested" and "not tested" look identical.

Run an evaluation against a hosted API:

```bash
sql-agent-eval --trials 3
```

Run it against a local small model:

```bash
OLLAMA_KEEP_ALIVE=30m ollama serve > /tmp/ollama.log 2>&1 &

DEEPSEEK_API_URL=http://127.0.0.1:11434/v1/chat/completions DEEPSEEK_API_KEY=ollama \
sql-agent-eval --model qwen2.5:3b --trials 1 --temperature 0 --workers 1
```

Every flag has a reason. `--workers 1` because Ollama has a single slot;
`--temperature 0` because an evaluation should measure capability, not luck;
`--trials 3` because whether temperature 0 is deterministic is an empirical
question, not a definitional one -- server-side batching, floating point and
concurrency all flip near-tied candidates.

Start the service:

```bash
sql-agent-serve                               # 0.0.0.0:8000 by default
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"How many products are there in total?"}'
```

Streaming over SSE, which emits tool events as well, so the user sees "querying
now" instead of thirty blank seconds:

```bash
curl -N -X POST localhost:8000/ask/stream -H 'Content-Type: application/json' \
     -d '{"question":"What is the price of the most expensive product?"}'
```

Events arrive as `run` (carrying a `run_id`), then `turn`, `tool_call`,
`tool_result`, and finally `final`. A run can be cancelled explicitly:

```bash
curl -s -X POST localhost:8000/runs/<run_id>/cancel
```

Dropping the client works too: about four seconds later the server records
`stream ended early` and the agent exits at the next turn boundary.

In Docker:

```bash
docker build -t sql-agent .
docker run --rm sql-agent sql-agent-check-tasks        # expects dev 76/76, test 24/24, exit 0
docker run --rm -p 8000:8000 --env-file .env sql-agent
```

Secrets are injected at run time and never enter the image (`.env` is in
`.dockerignore`; the built image contains neither the file nor the value). The
container runs as a non-root user, and the database is rebuilt at build time
from the fixed seed. A container cannot reach an Ollama running on the host --
`127.0.0.1` means the container itself -- so a local model needs its own
network configuration.

---

## Actions that need approval

`export_csv` overwrites a file of the same name and is the only irreversible
action in the project. It is hidden from the model by default; when it is
enabled, a run that wants it pauses instead of doing it:

```bash
curl -s -X POST localhost:8000/runs -H 'Content-Type: application/json' \
     -d '{"question":"export the user count"}'      # -> status: awaiting_approval
curl -s localhost:8000/runs/pending
curl -s -X POST localhost:8000/runs/<run_id>/approve
```

Three properties this has to have, each with a test behind it:

- **Nothing has happened at the moment it pauses.** Acting first and asking
  afterwards is the same as not asking.
- **A rejection is not a failure.** The action does not run and the run
  finishes normally, so declining does not throw away the work so far.
- **The pause survives the process.** Kill the server, start a new one, and the
  pending run is still there and still resumable. `scripts/check_hitl_server_restart.py`
  proves this across a real SIGKILL and a fresh process.

Being on the allowlist is not permission to run: what the model may see, what
it may call, and what a person approved are three separate gates.

---

## Code map

Layers, and the rule between them: each layer may call the ones below it and
never the ones above.

All of it lives in `src/sql_agent/`.

```
paths.py             resolves the checkout root once; everything else asks it
server.py            HTTP, SSE, cancellation, approvals
  run_store.py       persists a run's configuration next to its checkpoint
  agent.py           the hand-written loop
  agent_graph.py     the same loop as a LangGraph graph, kept field-identical
    tools.py         get_schema / run_sql / submit_answer / export_csv
    context.py       what to drop when the window fills
    llm.py           HTTP to the model, retries, error taxonomy
      reliability.py circuit breaker, guarded_call
      cache.py       exact-match response cache, off by default
      rate_limit.py  token bucket
runner.py            runs the task set concurrently, records trajectories
  verify.py          grades a run: result comparison and behaviour checks
  redact.py          strips PII on the way to disk
retrieval.py         word overlap and vector search over the task pool
  embedding.py       vectors, behind an interface the word-overlap arm ignores
tracing.py           span tree, cost attribution
```

Three more are entry points rather than layers -- nothing imports them, they are
run to produce the inputs everything else consumes:

```
seed_db.py           generates the database from a fixed seed
build_tasks.py       generates tasks/dev.json and tasks/test.json
check_tasks.py       the task-set self-check that guards them
```

🔴 **`build_tasks.py` is the largest file in the repository**, which is the right
shape for this project: the task set decides what every number here means, so it
gets more scrutiny than the agent does. `check_tasks.py` exists because a task set
that nothing checks is a measuring instrument nobody calibrated.

The files worth reading first, in order:

| File | What it holds |
|---|---|
| `verify.py` | How a run is graded. Read this before trusting any number in this README |
| `agent.py` | The loop itself, and the system prompt with the domain rules |
| `tools.py` | The four tools; the read-only database connection lives here |
| `runner.py` | Concurrency, per-task isolation, what gets written to `results/` |
| `agent_graph.py` | The same behaviour on LangGraph, held to the hand-written version field by field |

Two things about the code that are easy to get wrong:

**The database connection is read-only (`mode=ro`), and that is a diagnostic
measure as much as a safety one.** A read-only connection fails loudly when the
file is missing. A default connection silently creates an empty database, and
the error turns into "no such table", which points at entirely the wrong thing.

**`SYSTEM_PROMPT` is assembled at import time** from `tools.get_schema()`.
After changing `tools.DB_PATH`, reload `agent` -- otherwise the old schema is
still in the prompt. This is also why the package move was checked by comparing
the sha256 of the assembled prompt before and after: the schema reaching the
prompt differently would move the baseline under every number on this page,
without any symptom.

The two agent implementations are held together by
`tests/test_graph_equivalence.py`, which compares every field of the result and
every message under the same fake model. Its first run found three bugs, none
of which raised anything: the message dicts being converted to library objects,
counter fields being overwritten instead of accumulated, and the turn check
sitting on the wrong edge so the last turn's tool never ran.

---

## The dev and test split

| File | Tasks | Purpose | When to run |
|---|---|---|---|
| `tasks/dev.json` | 76 | tuning prompts, tools, context | freely |
| `tasks/test.json` | 24 | locked | only when announcing a result |

`test.json` was generated independently, not split off from dev.

It has been run five times, and every run is recorded with its purpose -- a
claim like "we only ran it twice" is worth nothing if nobody can check it. Two
runs were the announced results, two were a task-set audit with a stronger
model looking for broken tasks, and one was a re-measurement after one broken
task was fixed. **No prompt was ever changed on the basis of any of them.**

The rule: before running the test set, ask whether the goal is to announce a
result, to audit the tasks, or to satisfy curiosity. The first two are fine and
the third is not. Either way, finding a broken task allows fixing the task and
never allows tuning against it. What contaminates a held-out set is not running
it, it is choosing anything by looking at its numbers.

---

## How the tasks were built

```
candidates, generated by a model
 |- sanity checks (runs, non-empty, <= 100 rows, no SQL keywords in the question)
 |- cross-validation: a model re-solves it independently; keep only if the
 |  results match
 |- the checks in check_tasks.py
 `- a human reads the questions, which cannot be skipped
```

**Cross-validation is the core of it.** When one model writes both the question
and the gold answer, a wrong gold answer passes every automatic check -- it
runs, it returns rows, the row count is reasonable, and it answers a different
question than the one asked. Re-solving independently, without showing the
model its own gold answer, caught six of them.

**What is automated is the quality checking, not the authoring.** A batch that
passed every automatic check was still caught by a human noticing that the
questions read like someone reciting SQL.

Of the checks in `check_tasks.py`, most exist because something got through:
the gold answer runs, returns rows, does not return too many, ties at the
`LIMIT` boundary are resolved, unfinished orders are addressed explicitly, a
superlative says how many rows it wants, `ordered=True` is justified by the
question, and the question is not a truncated sentence.

**All of these are checks on form, and none of them can show that the task is
answerable.** Only running a strong model end to end does that, and it found
three broken tasks in a set where all the form checks were green.

A duplicate check runs separately, because it needs embeddings. Building the
retrieval arm turned up two pairs of duplicate tasks -- the generator worked in
batches and deduplicated within a batch but not across batches, so dev went
from 78 tasks to 76.

That flaw had been harmless right up until the moment it was not. As an
evaluation set, a duplicate task merely counts twice. **As a retrieval pool it
is a leak**, because retrieving examples for one task pulls up its twin with
nearly the same gold answer. The same data, used for something else, has to be
asked all over again whether it is good enough.

---

## The database

An e-commerce order database, four tables, `random.seed(42)`.

| Table | Rows | Key columns |
|---|---|---|
| `users` | 200 | id / name / email / city / signup_date |
| `products` | 20 | id / name / category / **price (current price)** |
| `orders` | 1,000 | id / user_id / order_date / status |
| `order_items` | 2,482 | id / order_id / product_id / quantity / **unit_price (price at order time)** |

Two traps are deliberate. **`unit_price` is not `products.price`**, which
separates agents that read the domain rules in the system prompt from agents
that guess. And **59 names are duplicated** -- one name is shared by seven
people -- so grouping by `name` is wrong.

---

## If this were scaled up a thousand times, what breaks first

The project is small on purpose: what it demonstrates is how to know whether
you are right, and adding distributed retrieval would only dilute that. But the
boundary should be a measurement rather than a guess, so it was measured.

| Pool size | Memory | One search, brute force numpy |
|---|---|---|
| 72 (today) | 0 MB | 0.2 ms |
| 7,200 | 22 MB | 35 ms |
| 72,000 | 221 MB | 40 ms |
| 720,000 | 2.2 GB | 221 ms |

**Brute force holds to the hundred-thousand range.** The common advice that a
thousand documents call for a vector database does not hold on this data.
Brute force is an exact answer and approximate nearest neighbour search is not:
**the criterion for adopting ANN is not "the data got big", it is "recall is
now worth trading for latency"** -- and at 40 ms for 72,000 items there is
nothing to trade it for.

What actually breaks first is the evaluation, not the retrieval:

| | 76 tasks | 760 | 7,600 |
|---|---|---|---|
| Building the index (one off, cacheable) | 2.7 s | ~30 s | ~5 min |
| One search | 0.2 ms | 2 ms | 35 ms |
| **One full evaluation run** | **37 min** | **6 h** | **60 h** |
| **Four arms of a comparison** | 2.5 h | **1 day** | **10 days** |

The evaluation falls over two orders of magnitude before the retrieval does,
and the evaluation is where this project's value sits. **Scaling up does not
make the hard part easier.**

What does not change with scale, and matters more: the placebo arm, because at
a million documents it is even less obvious where an improvement came from;
stratified attribution, because a single aggregate number is easier to be
fooled by, not harder; leakage checks, because overlap between pools gets
harder to notice as they grow; and a held-out set, because more iterations mean
more overfitting.

**The API of a vector library can be learned in a day. There is no library for
knowing whether your improvement was copied from the answer key.**

---

## Known limitations

- **The test set has only 24 tasks**, so its resolution is low. Minus one task
  is inside the noise: the honest statement is that the +7 seen on dev did not
  reproduce, not that few-shot is useless.
- **The verifier has a known false positive.** To tolerate different column
  orders, values within a row are sorted, so `(1, 2)` and `(2, 1)` compare
  equal. This is accepted because false negatives are far more common in SQL
  grading and they send you off optimising a problem that does not exist. The
  cost is counted separately as a `loose` verdict rather than hidden inside
  `passed=True`.
- **Why few-shot helps is still unknown.** Three candidate mechanisms were
  each refuted by the data, and a fourth was not invented to fill the gap.
- **The four behaviour tasks grade on structural signals only** -- whether an
  answer was submitted, whether a question was asked -- and not on wording.
  Grading wording needs an LLM judge, and the judge would have to be validated
  first. One of them fails for every model tried; a task that is always zero
  carries no information, the same way a ceiling does.
- **The task-set audit only catches one kind of broken task**: the kind a
  strong model also gets wrong. A question with a second reasonable reading
  that the strong model happens to share is invisible to this method, because
  the auditing model and the authoring model are the same family.
- **The historical dev numbers were measured before one task was fixed** and
  were not re-measured. All arms ran against the same wording, so the
  differences between arms are unaffected; only the absolute values move, by at
  most 2.6 points.

---

## About this repository's history

This repository was re-initialised from a private development repository, so the
commit history starts here rather than at the beginning of the work. The numbers
quoted above are not affected: every one of them traces to a run summary under
`results/`, which you can read without taking anything on trust.

---

## What this points to

Four prompt techniques failed against the same class of error, which locates the
ceiling: on this task, with this model, the remaining failures are not the kind a
better prompt fixes.

That conclusion is only worth as much as the harness that produced it -- and the
harness has a second use. It records the full `messages` trajectory of every run,
which is exactly the raw material a rejection-sampling pipeline consumes. **The
evaluation harness and the training-data pipeline are the same object**: the
records that decide whether a change helped are the records you would fine-tune
on.

*(Trajectory files are large and stay out of version control. `results/` ships the
per-run summaries that every number above is drawn from.)*
