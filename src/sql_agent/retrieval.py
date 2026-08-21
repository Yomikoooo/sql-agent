import json
import random
import re
import sys
from pathlib import Path

import numpy as np

from .embedding import embed_cached
from .tracing import span
from . import paths

TASKS_PATH = paths.TASKS_DIR / "dev.json"

STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
             "is", "are", "was", "were", "how", "what", "which", "who",
             "show", "list", "give", "me", "our", "us", "do", "we", "have",
             "each", "all", "by", "with", "that", "this", "their", "its"}

_POOL_CACHE: dict[str, list[dict]] = {}

_MATRIX_CACHE: dict[str, tuple[np.ndarray, list[dict]]] = {}

def tokenize(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9_]+", text.lower()))
    return words - STOPWORDS


def overlap_score(query: str, doc: str) -> float:
    a, b = tokenize(query), tokenize(doc)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_pool(path: Path | str = TASKS_PATH) -> list[dict]:
    key = str(path)
    if key not in _POOL_CACHE:
        with open(path, encoding="utf-8") as f:
            _POOL_CACHE[key] = json.load(f)
    return _POOL_CACHE[key]


def search_overlap(question, k=2, *, 
                   exclude_id: str | None = None, 
                   path: Path | str = TASKS_PATH) -> list[dict]:
    pool = [t for t in load_pool(path) if t.get("gold_sql") and t["id"] != exclude_id]
    ranked = sorted(pool, key=lambda t: overlap_score(question, t["question"]), reverse=True)
    return ranked[:k]

def make_suffix(examples: list[dict]) -> str:
    if not examples:
        return ""
    blocks = "\n\n".join(f"Q: {t['question']}\nA: {t['gold_sql']}" for t in examples)
    return (f"\n ## Worked examples\n"
            f"Questions like yours, and the query that answers them. \n\n{blocks}\n")


def make_suffix_for(question: str, k: int = 2) -> str:
    return make_suffix(search_overlap(question, k=k))


# ---------------------------------------------------------------- retrievers
# runner calls all of these through one signature: (question, *, exclude_id) -> str
# Adding a retriever means adding a row to RETRIEVERS, not editing runner.

K_EXAMPLES = 2          # same count as the fixed few-shot arm, so "how many
                        # examples" does not become a confounding variable

# Placebo seed. Changing it means a different placebo set, and past results
# stop being comparable.
PLACEBO_SEED = "placebo-v1"


def suffix_overlap(question: str, *, exclude_id: str | None = None) -> str:
    """Word-overlap retrieval."""
    with span("retrieve", retriever="overlap") as sp:
        hits = search_overlap(question, k=K_EXAMPLES, exclude_id=exclude_id)
        sp.attrs["retrieved_ids"] = [h["id"] for h in hits]   # which examples, recorded
        return make_suffix(hits)


def suffix_vector(question: str, *, exclude_id: str | None = None) -> str:
    """Embedding retrieval -- the arm under test."""
    with span("retrieve", retriever="vector") as sp:
        hits = search_vector(question, k=K_EXAMPLES, exclude_id=exclude_id)
        sp.attrs["retrieved_ids"] = [h["id"] for h in hits]   # which examples, recorded
        return make_suffix(hits)


def suffix_random(question: str, *, exclude_id: str | None = None) -> str:
    """Placebo: k examples drawn at random.

    Identical to suffix_vector in every respect -- real questions, real gold
    SQL, same format, same count, self excluded. The only difference is
    relevance, which is what makes this arm worth running: earlier controls
    varied the *content* of the suffix, or the *difficulty* of the examples.
    This one isolates relevance alone.

    The seed is built from strings, never from hash(): hash(str) is randomised
    per process by PYTHONHASHSEED, so the experiment would not reproduce.
    Measured: random.Random("abc") agrees across processes, and
    random.Random(hash("abc")) does not.

    The question goes into the seed too. Without it every task draws the same
    examples, which is not "random relevance" but a second fixed few-shot set.
    """
    pool = [t for t in load_pool() if t.get("gold_sql") and t["id"] != exclude_id]
    rng = random.Random(f"{PLACEBO_SEED}::{question}")
    return make_suffix(rng.sample(pool, min(K_EXAMPLES, len(pool))))


def build_matrix(path: Path | str = TASKS_PATH) -> tuple[np.ndarray, list[dict]]:
    key = str(path)
    if key not in _MATRIX_CACHE:
        pool = [t for t in load_pool(path) if t.get("gold_sql")]
        M = np.array(embed_cached([t["question"] for t in pool]))
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        assert np.allclose(np.linalg.norm(M, axis=1), 1.0)
        _MATRIX_CACHE[key] = (M, pool)
    return _MATRIX_CACHE[key]


def search_vector(question: str, k: int = 2, *, exclude_id: str | None = None, path: Path | str = TASKS_PATH) -> list[dict]:
    M, pool = build_matrix(path)
    q = np.array(embed_cached([question])[0])
    q = q / np.linalg.norm(q)
    scores = M @ q
    if exclude_id is not None:
        for i, t in enumerate(pool):
            if t["id"] == exclude_id:
                scores[i] = -np.inf

    idx = np.argsort(-scores)[:k]
    return [pool[i] for i in idx]


def main() -> int:
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = [
            "What are our three best-selling products by units sold?",
            "How many users are in Chengdu?",
            "What is the average order value?",
        ]

    pool = load_pool()
    print(f"pool: {len(pool)} tasks\n")
    for q in questions:
        print(f"Q: {q}")
        for t in search_overlap(q, k=3):
            score = overlap_score(q, t["question"])
            print(f"   {score:.3f}.  {t['id']}  {t['question'][:64]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())