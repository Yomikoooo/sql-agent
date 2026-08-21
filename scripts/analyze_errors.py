from sql_agent.paths import ROOT  # noqa: F401
import json
from collections import Counter


def load_latest() -> list[dict]:
    path =  sorted((ROOT / "results").glob("*.jsonl"))[-1]
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines]



def count_errors(rows: list[dict]) -> Counter:
    counts = Counter()
    for row in rows:
        for msg in row["messages"]:
            if msg["role"] != "tool":
                continue
            text = msg["content"]
            if text.startswith(("SQL error:", "Error:")):
                counts[text[:80]] += 1

    return counts


if __name__ == "__main__":
    counts = count_errors(load_latest())
    for text, n in counts.most_common(10):
        print(f"{n:3d}  {text}")
