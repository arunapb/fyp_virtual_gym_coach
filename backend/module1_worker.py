"""
backend/module1_worker.py — Module 1's pipeline, run inside its own worker process.

Module 1's `src`/`config` and Module 2's `src`/`config` are unrelated packages
that happen to share the same import names (`src.session`, `src.features`,
bare `import config`, ...). They cannot both be imported in one interpreter, so
Module 1 never runs in the main server process — only in here, a separate
process spawned by a `ProcessPoolExecutor`, which imports Module 1's `src`/
`config` and nothing of Module 2's.

`init_worker` is the pool's `initializer`: it runs once when the worker process
starts and stores one `GymCoachSession` in a module-level global, so per-frame
calls reuse the same loaded models and session state instead of re-loading the
two `.pt` files on every frame. One pool is created per job (see
backend/main.py), so a fresh worker process — and therefore a fresh session —
is guaranteed for every uploaded video.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE1_DIR = REPO_ROOT / "module1"
MODULE2_DIR = REPO_ROOT / "module2"

_session = None


def _same_path(a: str, b: str) -> bool:
    """
    Compare two sys.path entries as PATHS, not as strings.

    An exact string comparison is not enough: the same directory reaches
    sys.path in several equivalent spellings ('f:\\x' vs 'F:\\x' — Windows
    drive-letter case, forward vs back slashes, trailing separator, relative
    vs absolute). A missed match here fails silently and catastrophically —
    Module 2's directory survives on the path, `import src` resolves to its
    regular package instead of Module 1's namespace package, and the worker
    runs the wrong module entirely.
    """
    try:
        return (os.path.normcase(os.path.realpath(a))
                == os.path.normcase(os.path.realpath(b)))
    except (OSError, ValueError):
        return False


def init_worker() -> None:
    """
    Pool initializer: runs once in the freshly spawned worker process.

    Module 1's `src/` has no `__init__.py` (an implicit PEP 420 namespace
    package), while Module 2's `src/` does (a regular package). Per Python's
    import resolution, a REGULAR package found ANYWHERE on `sys.path` wins
    over a namespace-package portion found earlier — inserting Module 1's
    directory at sys.path[0] is not enough on its own if Module 2's directory
    (inherited from the parent process via spawn) is still present further
    down the path; `import src` would still resolve to Module 2's package.
    So Module 2's directory is dropped from `sys.path` outright, not just
    outranked, before anything here imports `src` or `config`.
    """
    global _session
    sys.path[:] = [p for p in sys.path if not _same_path(p, str(MODULE2_DIR))]
    sys.path.insert(0, str(MODULE1_DIR))

    # Evict any ALREADY-IMPORTED `src`/`config` before importing Module 1's.
    # Fixing sys.path only governs future lookups — an import that already
    # happened is cached in sys.modules and would be returned as-is. This
    # occurs whenever the process that spawned this worker had Module 2's
    # packages imported at the time of the spawn (on Windows, `spawn`
    # re-executes the parent's entry module in the child; see backend/run.py
    # for why the app's entry point is guarded against exactly this).
    for name in [m for m in sys.modules
                 if m == "src" or m.startswith("src.")
                 or m == "config" or m.startswith("config.")]:
        del sys.modules[name]

    from src.session import GymCoachSession  # Module 1's src — only ever imported here

    # Prove we got Module 1's package, not Module 2's same-named one. Without
    # this, a path-isolation failure surfaces later as a baffling AttributeError
    # or, worse, as plausible-looking output from the wrong pipeline.
    import src
    if getattr(src, "__file__", None):
        src_dir = os.path.dirname(src.__file__)    # regular package (has __init__)
    else:
        src_dir = next(iter(src.__path__), "")      # namespace package (Module 1's)
    if not _same_path(src_dir, str(MODULE1_DIR / "src")):
        raise ImportError(
            f"Module 1 worker resolved 'src' to {src_dir!r}, expected "
            f"{str(MODULE1_DIR / 'src')!r}. Module 2's package shadowed it — "
            f"sys.path isolation failed.")

    _session = GymCoachSession()


_EMPTY_RESULT = {"state": "null", "exercise": None, "label": None,
                 "confidence": 0.0, "state_probs": None, "early_guess": None}


def process_frame(jpeg_bytes: bytes) -> dict:
    """
    Run one JPEG-encoded frame through Module 1's unmodified pipeline.

    Uses `GymCoachSession.process_frame_bytes`, which already exists in Module 1
    exactly for this purpose (its own docstring: "regardless of whether they
    came from a browser webcam or a server-side video file"). Returns a small
    JSON-safe dict — the wall-clock-derived fields (`elapsed_active_sec`,
    `frame_index`) are dropped since frames here aren't paced to real time.

    `early_guess` — Module 1's confirmed `exercise` only appears once its own
    `ExercisePredictor` has locked in (15-90 consecutive confident frames,
    module1/src/exercise_rules.py). Reaching that lock inherently requires
    Module 1 to have already observed real exercise motion (its masking rules
    need actual leg movement / hip drop / wrist height to distinguish
    exercises), so by the time it locks, the user is already mid-rep — too
    late for Module 2's calibration, which needs a STILL starting pose.
    `ExercisePredictor` tracks its own best not-yet-locked guess on
    `_proposed_exercise` (set once single-frame confidence >= 50%, well before
    the multi-frame lock) — reading it here (read-only; nothing here writes to
    Module 1's state) lets Module 2 start calibrating on that earlier guess
    instead. If the guess later changes, Module 2's own signal-change handling
    (unmodified) already tears down and recalibrates for the corrected
    exercise, so an occasional wrong early guess costs a wasted calibration
    attempt, not a wrong session.
    """
    global _session
    if _session is None:
        init_worker()

    result = _session.process_frame_bytes(jpeg_bytes, include_preview=False)
    if result is None:
        return dict(_EMPTY_RESULT)

    early_guess = None
    if result["state"] == "active" and result["exercise"] is None:
        early_guess = getattr(_session.exercise_predictor, "_proposed_exercise", None)

    return {
        "state": result["state"],
        "exercise": result["exercise"],
        "label": result["label"],
        "confidence": result["confidence"],
        "state_probs": result["state_probs"],
        "early_guess": early_guess,
    }
