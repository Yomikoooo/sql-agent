"""Where this project's data lives.

Every module used to locate `data/`, `tasks/`, `results/`, `state/` and
`exports/` with `Path(__file__).parent`, which ties the data to wherever the
module file happens to sit.  That works only while the modules sit in the
project root, and it fails silently the moment they move: the directories are
created with `mkdir(exist_ok=True)`, so a relocated module writes an empty
`tasks/` next to itself, a reader keeps reading the old one, and nothing raises.
Moving files this way has already broken relative paths here once.

So the project root is resolved once, here, and everything else asks for it.

The order, and why:

1. `SQLAGENT_HOME`, when set.  Explicit beats inferred, and it is what lets a
   container mount the data somewhere else, or a second copy of the data be
   evaluated without editing any file.  The precedent in this project is
   `SQLAGENT_PROMPT_SUFFIX`: an environment variable plus a subprocess lets
   variants run in isolation, with no file edits and no race.
2. Upward from this file, looking for a marker.  This identifies *this*
   installation and keeps working from any working directory.  It finds nothing
   once the package is installed into site-packages, which is correct -- the
   code is then no longer next to the data.
3. Upward from the current working directory, for that installed case.
4. Otherwise raise.  It must never quietly fall back to the working directory,
   because a wrong guess here creates empty directories in the wrong place and
   reports nothing.

The marker is `pyproject.toml`, which sits at the checkout root and nowhere
else -- notably not inside `src/sql_agent/`, so the upward search cannot stop
short.
"""
import os
from pathlib import Path

MARKERS = ("pyproject.toml",)


def _search_upward(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if any((candidate / m).exists() for m in MARKERS):
            return candidate
    return None


def find_root() -> Path:
    env = os.environ.get("SQLAGENT_HOME")
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"SQLAGENT_HOME points at {root}, which is not a directory")
        return root

    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        found = _search_upward(start)
        if found is not None:
            return found

    raise RuntimeError(
        "cannot locate the project root: none of "
        f"{MARKERS} was found above {Path(__file__).resolve().parent} or "
        f"above {Path.cwd().resolve()}. Set SQLAGENT_HOME to the checkout."
    )


ROOT = find_root()

DATA_DIR = ROOT / "data"
TASKS_DIR = ROOT / "tasks"

# `results/` holds the recorded evaluation runs, and it is deliberately not
# under `artifacts/`.  Every number quoted about this project traces back to a
# summary in here, so it is the one output directory that must survive; putting
# it beside the disposable ones invites a single `rm -rf` or one line of
# .gitignore to take the evidence with them.
RESULTS_DIR = ROOT / "results"

# Everything below is regenerable and none of it is committed.
ARTIFACTS_DIR = ROOT / "artifacts"
STATE_DIR = ARTIFACTS_DIR / "state"
EXPORT_DIR = ARTIFACTS_DIR / "exports"
LOGS_DIR = ARTIFACTS_DIR / "logs"
CACHE_DIR = ARTIFACTS_DIR / "cache"

DB_PATH = DATA_DIR / "shop.db"
