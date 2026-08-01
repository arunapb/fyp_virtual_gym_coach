"""
signal_source.py — Module 1 signal abstraction.

Module 1 (exercise-state classifier) is not integrated yet.  It will emit one
label per frame: "Null" (background / no user), "Rest" (user present but idle),
or an exercise name ("Squat", "BicepCurl", ...).

`Module1SignalSource` is the interface the downstream system depends on.  Today
we use `KeyboardSignalSource` (dev stub) or `ConstantSignalSource` (scripted /
acceptance runs).  When the real Module 1 lands it implements the same interface
and nothing downstream changes.
"""

import threading
from abc import ABC, abstractmethod

# Canonical signal labels.
SIGNAL_NULL = "Null"
SIGNAL_REST = "Rest"


class Module1SignalSource(ABC):
    """Emits the current Module-1 signal label, polled once per frame."""

    @abstractmethod
    def current(self) -> str:
        """Return the current signal label for this frame."""

    def on_key(self, key: int) -> None:
        """Optional: react to a keypress (used by the keyboard dev stub)."""
        return None


class ConstantSignalSource(Module1SignalSource):
    """Always emits the same label.  Used by the acceptance harness (Squat)."""

    def __init__(self, label: str):
        self._label = label

    def current(self) -> str:
        return self._label


class LiveSignalSource(Module1SignalSource):
    """
    A label that a DIFFERENT thread sets while the frame loop reads it.

    This is the seam Module 1 plugs into.  The analysis loop polls `current()`
    once per frame, exactly as it polls the keyboard stub; the label is written
    from wherever the signal is produced — today the browser's signal control
    over the WebSocket, tomorrow Module 1's classifier — and neither side knows
    about the other.

    Thread safety is the whole point of the class.  `str` assignment happens to
    be atomic under CPython, but the lock also makes `last_change` consistent
    with the label it describes, and states the contract explicitly rather than
    resting on an implementation detail of one interpreter.
    """

    def __init__(self, label: str = SIGNAL_REST):
        self._lock = threading.Lock()
        self._label = label
        self._changes = 0

    def current(self) -> str:
        with self._lock:
            return self._label

    def set(self, label: str) -> bool:
        """Adopt a new label.  Returns True if it differed from the current one."""
        with self._lock:
            if label == self._label:
                return False
            self._label = label
            self._changes += 1
            return True

    @property
    def changes(self) -> int:
        with self._lock:
            return self._changes


class KeyboardSignalSource(Module1SignalSource):
    """
    Development stub: keyboard drives the signal until Module 1 is integrated.

        1 -> Null        2 -> Rest        3 -> Squat        4 -> BicepCurl

    Pressing a key for an exercise that is not implemented yet (BicepCurl) logs
    a message and leaves the current signal unchanged (handled in
    SessionController, which knows the registry).  Default at startup: Rest.
    """

    KEY_MAP = {
        ord("1"): "Null",
        ord("2"): "Rest",
        ord("3"): "Squat",
        ord("4"): "BicepCurl",
        ord("5"): "ShoulderPress",
    }

    def __init__(self, default: str = SIGNAL_REST):
        self._label = default

    def current(self) -> str:
        return self._label

    def on_key(self, key: int) -> None:
        label = self.KEY_MAP.get(key)
        if label is not None:
            self._label = label
