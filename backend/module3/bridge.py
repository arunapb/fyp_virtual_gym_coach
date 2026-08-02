"""
backend/module3/bridge.py — importing Module 3's service package into the
shared server process, lazily and without disturbing Modules 1, 2 or 4.

Two problems to solve.

1. Module 3's code uses ABSOLUTE top-level imports (`from services import
   exercise_service`, `from models.schemas import ...`), which only resolve
   with `module3/backend/` on `sys.path`. That directory is therefore appended
   here — appended, never inserted at the front. Module 2's directory sits at
   `sys.path[0]` (see backend/main.py) and must keep winning: both directories
   contain an `app.py`, and putting Module 3's ahead of it would change which
   `app` module any future `import app` resolves to. Appending can only ever
   ADD names (`services`, `models`) that nothing else on the path provides;
   it can never shadow a name that already resolves.

   `models` deserves a note: Module 1 and Module 2 both have a `models/`
   directory too, but they hold `.pt`/`.task` weights, not Python — they are
   namespace-package portions with no `__init__.py`, so Module 3's regular
   `models` package wins regardless of path order. Verified that nothing in
   module1/ or module2/ imports `models`, `services` or `app` as a module.

2. Module 3's heavy services load their models at IMPORT time, not on first
   call: `retrieval_service` reads the FAISS index, unpickles the recipe
   corpus, builds a BM25 index and embeds every unique ingredient with a
   SentenceTransformer; `feedback_service` pulls in torch/transformers/spaCy.
   Importing those at server boot would add tens of seconds and hundreds of MB
   before the Module 1 + Module 2 workout page could be reached, and would
   take the whole server down if the machine were offline. So every service is
   imported on first use instead, under a lock so concurrent first requests
   import exactly once. This mirrors `backend/module4/model.ensure_loaded()`.

   `exercise_service`, `profile_service`, `goal_service` and `history_service`
   are pure stdlib + JSON — importing them is free. That is what lets the
   end-of-workout exercise log (backend/module3/exercise_log.py) write its
   entry without ever dragging FAISS or torch into the analysis thread.
"""

import importlib
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODULE3_DIR = REPO_ROOT / "module3"
MODULE3_BACKEND_DIR = MODULE3_DIR / "backend"

# Services that load ML models / indexes at import time. Only used to report
# what is warm on the health endpoint — `load()` treats every service alike.
HEAVY_SERVICES = ("retrieval_service", "feedback_service", "recommender_service")

_lock = threading.RLock()
_modules: dict = {}
_path_added = False


def ensure_path() -> None:
    """Put `module3/backend/` on `sys.path` (append-only, idempotent)."""
    global _path_added
    if _path_added:
        return
    with _lock:
        if _path_added:
            return
        if not MODULE3_BACKEND_DIR.is_dir():
            raise RuntimeError(
                f"Module 3 not found at {MODULE3_BACKEND_DIR}. The shared "
                f"server imports its services from there rather than copying "
                f"them — see backend/module3/__init__.py.")
        entry = str(MODULE3_BACKEND_DIR)
        if entry not in sys.path:
            sys.path.append(entry)
        _path_added = True


def load(name: str):
    """
    Import (once) and return one of Module 3's `services.*` modules.

    Blocking, and for the heavy services genuinely slow the first time — call
    it from a worker thread, never on the event loop. router.py routes every
    request through `run_in_threadpool` for exactly this reason.
    """
    module = _modules.get(name)
    if module is not None:
        return module
    with _lock:
        module = _modules.get(name)
        if module is None:
            ensure_path()
            module = importlib.import_module(f"services.{name}")
            _modules[name] = module
        return module


def is_loaded(name: str) -> bool:
    """
    Is this service warm?

    Answered from `sys.modules`, not from `_modules`. Module 3's services
    import each other — `recommender_service` does `from . import
    retrieval_service` — so loading one warms two or three, and only the one
    that was asked for ever reaches `_modules`. Checking the local cache would
    report `retrieval_service` cold while its FAISS index sat fully loaded in
    memory.
    """
    return f"services.{name}" in sys.modules


# ── Named accessors ─────────────────────────────────────────────────────────
# Thin, so callers read as `bridge.profile().get_profile()` rather than
# repeating stringly-typed module names.

def exercise():      return load("exercise_service")     # noqa: E704 — table
def profile():       return load("profile_service")      # noqa: E704
def goal():          return load("goal_service")         # noqa: E704
def history():       return load("history_service")      # noqa: E704
def feedback():      return load("feedback_service")     # noqa: E704 (heavy)
def recommender():   return load("recommender_service")  # noqa: E704 (heavy)
