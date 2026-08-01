"""
exercises — pluggable per-exercise analysis modules.

Each concrete exercise (SquatExercise, future BicepCurlExercise, ...) implements
the Exercise interface in base.py.  Adding a new exercise means adding one new
class here and registering it in session.EXERCISE_REGISTRY — no edits to the
generic measurement / drawing / audio / rep-counter machinery.
"""

from src.exercises.base import (
    Exercise, Calibration, RepState, DisplaySpec, InfoCell, AudioCueMapping,
)
from src.exercises.squat import SquatExercise

__all__ = [
    "Exercise", "Calibration", "RepState", "DisplaySpec", "InfoCell",
    "AudioCueMapping", "SquatExercise",
]
