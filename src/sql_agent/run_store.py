"""Persist the configuration of a run alongside its checkpoint.

A LangGraph checkpoint stores only `State` (messages, counters). The run's
`model` / `max_turns` / `enable_export` are closure variables of `build_graph`
and are *not* in it. Once the server process restarts, they are gone, and the
approval that arrives afterwards has nothing to resume with -- while "approve
and continue after a restart" is exactly what this milestone has to prove.

Resuming with different values does not raise: a run started with
`max_turns=4` will happily resume with `max_turns=8`, silently becoming a
different run. So this table is not only about being able to resume, it is
about resuming *the same* run.

Stored in the same sqlite file as the checkpoints because their lifetimes are
identical; splitting them eventually yields a checkpoint with no config, or
the reverse.
"""
import json
import sqlite3
from . import paths

STATE_DIR = paths.STATE_DIR
DB_PATH = STATE_DIR / "checkpoints.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    config     TEXT NOT NULL,      -- JSON: the build_graph arguments
    status     TEXT NOT NULL,      -- awaiting_approval | done | error
    pending    TEXT,               -- JSON: actions awaiting approval, else NULL
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def connect() -> sqlite3.Connection:
    """Open the state database.

    `check_same_thread=False` because FastAPI runs sync endpoints in a thread
    pool: starting a run and approving it are rarely the same thread, and
    sqlite refuses outright without it.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def save(conn: sqlite3.Connection, run_id: str, question: str,
         config: dict, status: str, pending: list | None) -> None:
    conn.execute(
        "INSERT INTO runs (run_id, question, config, status, pending) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, "
        "pending=excluded.pending",
        (run_id, question, json.dumps(config, ensure_ascii=False), status,
         json.dumps(pending, ensure_ascii=False) if pending else None),
    )
    conn.commit()


def load(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute(
        "SELECT run_id, question, config, status, pending, created_at "
        "FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return {"run_id": row[0], "question": row[1], "config": json.loads(row[2]),
            "status": row[3], "pending": json.loads(row[4]) if row[4] else None,
            "created_at": row[5]}


def list_pending(conn: sqlite3.Connection) -> list[dict]:
    """Runs still awaiting a human. This is how a restarted process finds out
    that something is waiting for approval."""
    rows = conn.execute(
        "SELECT run_id, question, pending, created_at FROM runs "
        "WHERE status = 'awaiting_approval' ORDER BY created_at").fetchall()
    return [{"run_id": r[0], "question": r[1],
             "pending": json.loads(r[2]) if r[2] else None, "created_at": r[3]}
            for r in rows]
