"""
backend/signal_mapping.py — Module 1 vocabulary -> Module 2 vocabulary.

Pure function, no imports from either module, so it can be called from either
process without pulling in an import collision.
"""

# Module 1's TARGET_EXERCISES (module1/config.py) -> Module 2's EXERCISE_REGISTRY
# keys (module2/src/session.py). "deadlift" has no Module 2 implementation; it is
# passed through unmapped on purpose — SessionController.set_signal() already
# no-ops gracefully on a known-but-unregistered label (logs, stays idle).
_EXERCISE_TO_MODULE2 = {
    "squat": "Squat",
    "bicep_curl": "BicepCurl",
    "shoulder_press": "ShoulderPress",
    "deadlift": "Deadlift",
}

SIGNAL_NULL = "Null"
SIGNAL_REST = "Rest"


def map_signal(state: str, exercise: str | None, early_guess: str | None = None) -> str:
    """
    Module 1's (state, exercise, early_guess) -> the single label string
    Module 2 consumes.

    Prefers the confirmed `exercise` once Module 1 has locked it in. Before
    that, while `state == "active"`, falls back to `early_guess` — Module 1's
    own not-yet-locked best guess — rather than Rest, so Module 2 starts
    calibrating closer to the actual onset of motion instead of waiting for
    Module 1's full multi-frame confirmation (by which point the user is
    already mid-rep; see module1_worker.py's process_frame docstring). If the
    guess is wrong, Module 2 (unmodified) tears down and recalibrates the
    moment the signal changes, so this only risks a wasted attempt, not a
    silently wrong session.
    """
    if state == "null":
        return SIGNAL_NULL
    if state == "active":
        if exercise in _EXERCISE_TO_MODULE2:
            return _EXERCISE_TO_MODULE2[exercise]
        if early_guess in _EXERCISE_TO_MODULE2:
            return _EXERCISE_TO_MODULE2[early_guess]
    return SIGNAL_REST
