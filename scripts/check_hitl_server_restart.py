"""Evidence at the service level: start the server, trigger an approval, kill the
server, start a new one, approve, and let the run finish.

The criterion being met: trigger a run that needs approval, restart the server
process, then approve.  The run has to resume from where it paused and end up
where it would have without the restart.

This is the service layer -- a real uvicorn, real HTTP, a real kill and a fresh
process.  The logic of what pauses and what happens after a rejection lives in
tests/test_hitl.py, which runs in milliseconds; this script covers what those
tests cannot reach.

Why the service layer needs its own proof: persisted graph state shows the
conversation survived, but a run also needs its configuration (model,
max_turns, enable_export) on disk.  If the configuration only ever lived in
memory, no checkpoint is complete enough to reconstruct the run after a restart
-- see the notes at the top of `run_store.py`.

Run it with:
    python scripts/check_hitl_server_restart.py
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from sql_agent.paths import ROOT  # noqa: F401

PORT = 8791
BASE = f"http://127.0.0.1:{PORT}"
EXPORT = ROOT / "exports" / "server_restart.csv"

# A fake model: ask for the export (the high-impact action) on the first turn,
# answer afterwards.  It is injected into the child process, because the server
# runs under uvicorn and cannot be monkeypatched from out here.
PATCH = r'''
import json
from sql_agent import agent, agent_graph

def fake(messages, tools, *, model=None, temperature=0.0, **kw):
    if not any(m.get("role") == "tool" for m in messages):
        return {"choices":[{"message":{"role":"assistant","content":"",
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"export_csv",
              "arguments": json.dumps({"query":"SELECT COUNT(*) AS n FROM users",
                                       "filename":"server_restart.csv"})}}]},
            "finish_reason":"tool_calls"}],"usage":{}}
    return {"choices":[{"message":{"role":"assistant","content":"exported"},
                        "finish_reason":"stop"}],"usage":{}}

agent.call_with_tools = fake
agent_graph.call_with_tools = fake
'''


def wait_port(proc: subprocess.Popen, timeout: float = 25.0) -> None:
    """Wait for the port, while also watching whether the process died.

    Waiting on the port alone means a server that fails to start is waited out
    to the full timeout, with the actual error sitting unread in stderr.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"the server exited before it came up (code={proc.returncode}):\n"
                             f"{proc.stderr.read() if proc.stderr else ''}")
        with socket.socket() as s:
            s.settimeout(0.4)
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                return
        time.sleep(0.3)
    raise SystemExit("the server never listened on the port within the timeout")


def start_server(tag: str) -> subprocess.Popen:
    # The child imports the installed package, so nothing has to be added to
    # sys.path here -- that indirection went away with the packaging move.
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"exec({PATCH!r});"
         "from sql_agent import server; import uvicorn;"
         f"uvicorn.run(server.app, host='127.0.0.1', port={PORT}, log_level='warning')"],
        cwd=str(ROOT), env={**os.environ}, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    wait_port(proc)
    print(f"  server {tag} is up, pid={proc.pid}")
    return proc


def stop(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGKILL)      # KILL, not TERM: a real crash gets no chance to clean up
    proc.wait(timeout=10)


def main() -> int:
    EXPORT.unlink(missing_ok=True)
    ok = True

    print("1. start the server and send a request that needs approval")
    s1 = start_server("A")
    try:
        r = httpx.post(f"{BASE}/runs", json={"question": "export the user count"},
                       timeout=60).json()
        run_id = r["run_id"]
        print(f"  -> status={r['status']}  run_id={run_id}")
        if r["status"] != "awaiting_approval":
            print("FAIL: it should have paused for approval"); ok = False
        pid1 = s1.pid
    finally:
        print("2. SIGKILL the server, so it gets no chance to clean up")
        stop(s1)

    print(f"  the file must not exist yet: {not EXPORT.exists()}")
    if EXPORT.exists():
        print("FAIL: the file was written before anyone approved"); ok = False

    print("3. start a brand new server process")
    s2 = start_server("B")
    try:
        pid2 = s2.pid
        print(f"  the two servers are different processes: {pid1 != pid2}  ({pid1} -> {pid2})")

        left = httpx.get(f"{BASE}/runs/pending", timeout=30).json()["pending"]
        print(f"4. the new process sees {len(left)} pending run(s)"
              f" (the run_id matches: {any(x['run_id'] == run_id for x in left)})")
        if not any(x["run_id"] == run_id for x in left):
            print("FAIL: the new process cannot see the run awaiting approval"); ok = False

        d = httpx.post(f"{BASE}/runs/{run_id}/approve", timeout=60).json()
        print(f"5. approve inside the new process -> status={d['status']} answer={d.get('answer')!r}")
        if d["status"] != "done" or d.get("stop_reason") != "done":
            print("FAIL: approving should have resumed the run to completion"); ok = False
    finally:
        stop(s2)

    print(f"6. the file is now written: {EXPORT.exists()}")
    if not EXPORT.exists():
        print("FAIL: the file should exist once the action was approved"); ok = False

    print()
    print("OK: killing the server and starting another keeps the pending run, "
          "and approving resumes it from where it paused" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
