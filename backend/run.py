"""
backend/run.py — the launcher. Run the app with `python backend/run.py`, not
`uvicorn backend.main:app` from the CLI.

Why this file needs to exist and be this minimal
--------------------------------------------------
On Windows, `ProcessPoolExecutor` uses the `spawn` start method: each worker
process is a fresh interpreter that reconstructs just enough of the parent's
state to run, which includes re-executing the parent's ENTRY SCRIPT's
top-level code (under `__name__ == "__mp_main__"`, not `"__main__"`) — this is
exactly why the standard library tells you to guard multiprocessing entry
points with `if __name__ == "__main__":`.

If the entry point were `backend/main.py` itself (e.g. running it via
`uvicorn backend.main:app` or `python backend/main.py`), that file's
module-level code — `sys.path.insert(0, MODULE2_DIR)` followed by
`from config import ...` / `from src... import ...` — would re-run in every
spawned worker BEFORE that worker's own `module1_worker.init_worker()`
initializer gets a chance to claim `src`/`config` for Module 1. Whichever
claims `sys.modules['src']` first wins for that interpreter's whole lifetime,
so the worker would end up running Module 2's `src.session` instead of
Module 1's, and immediately fail (as verified while building this).

This file avoids that by keeping `backend.main` (and therefore Module 2's
sys.path setup) behind the `if __name__ == "__main__":` guard, imported
lazily by `uvicorn.run("backend.main:app", ...)` only inside that guard — so
the module-level re-execution in a spawned child never reaches it.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Repo root on sys.path, so `backend.main` is importable regardless of
    # whether this was launched as `python backend/run.py` (sys.path[0] would
    # otherwise be backend/ itself) or `python -m backend.run` (already correct).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000)
