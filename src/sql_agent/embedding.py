import httpx
import json
import sys

from . import paths

CACHE_PATH = paths.CACHE_DIR / "embed_cache.json"
OLLAMA_URL = "http://127.0.0.1:11434/v1/embeddings"
DEFAULT_MODEL = "nomic-embed-text"


def _load_cache() -> dict[str, list[float]]:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def embed(texts: list[str], model: str = DEFAULT_MODEL) -> list[list[float]]:
    resp = httpx.post(
            OLLAMA_URL,
            json={"model": model, "input": texts},
            timeout=120.0,
        )
    resp.raise_for_status()
    data = resp.json()
    items = sorted(data["data"], key=lambda d: d["index"])
    return [item["embedding"] for item in items]


def embed_cached(texts: list[str], model: str = DEFAULT_MODEL) -> list[list[float]]:
    cache = _load_cache()
    misses = [t for t in dict.fromkeys(texts) if f"{model}::{t}" not in cache]
    if misses:
        for text, vec in zip(misses, embed(misses, model)):
            cache[f"{model}::{text}"] = vec
        _save_cache(cache)
    return [cache[f"{model}::{t}"] for t in texts]


def main() -> int:
    vecs = embed(["hello", "how many users do you have"])
    print("count", len(vecs), "| dim", len(vecs[0]),
          "| norm", sum(v * v for v in vecs[0]) ** 0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())