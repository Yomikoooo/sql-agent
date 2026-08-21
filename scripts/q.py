from sql_agent.paths import ROOT  # noqa: F401
import sqlite3
import sys
from pathlib import Path


DB = ROOT / "data" / "shop.db"


def run(sql: str) -> None:
    conn = sqlite3.connect(DB)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
    except sqlite3.Error as e:
        print(f"SQL error {e}")
        return
    finally:
        conn.close()

    if cols:
        print(" | ".join(cols))
        print("-" * 60)
    for r in rows[:30]:
        print(" | ".join(str(x) for x in r))
    print(f"\n({len(rows)} rows)")

if __name__ == "__main__":
    run(sys.argv[1])

    