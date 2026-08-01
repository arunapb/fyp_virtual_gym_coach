"""
session.py — SessionController: routes the Module 1 signal to the right
exercise, owns the active-exercise lifecycle, and drives the per-frame pipeline.

Responsibilities
────────────────
  * Translate each frame's Module-1 signal (Null / Rest / <Exercise>) into an
    active exercise instance (or None for Null/Rest).
  * Manage transitions: instantiate on entry, tear everything down on exit,
    swap cleanly on exercise→exercise.
  * Own calibration buffering (only while an exercise is active and the user
    holds the calibration pose) and run the exact per-frame analysis order of
    the pre-refactor loop.
  * Expose state to drawing.py and feed the audio controller.

Adding a new exercise = add a class + one EXERCISE_REGISTRY entry.
"""

import re
from dataclasses import dataclass
from enum import Enum

from config import BASELINE_FRAMES, CALIB_MAX_FRAMES
from src.signal_source import SIGNAL_NULL, SIGNAL_REST, KeyboardSignalSource
from src.exercises.squat import SquatExercise
from src.exercises.bicep_curl import BicepCurlExercise
from src.exercises.shoulder_press import ShoulderPressExercise


# Maps a Module-1 exercise label to its Exercise class.
EXERCISE_REGISTRY = {
    "Squat": SquatExercise,
    "BicepCurl": BicepCurlExercise,
    "ShoulderPress": ShoulderPressExercise,
}


def _pretty_label(label: str) -> str:
    """'BicepCurl' -> 'Bicep Curl'.  Registry keys are CamelCase; prompts are not."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", label)


def rest_banner_text() -> str:
    """
    The Rest-state prompt, e.g. "Rest - press 3 for Squat, 4 for Bicep Curl".

    DERIVED, never hardcoded: the offered keys are the keyboard map INTERSECTED
    with EXERCISE_REGISTRY, so the prompt lists exactly those exercises that are
    both key-reachable and actually implemented.  Registering a new exercise
    therefore updates this prompt automatically instead of leaving it stale.

    ASCII only — this string is rendered with cv2.putText, and OpenCV's Hershey
    fonts have no glyphs beyond ASCII: a non-ASCII character (an em-dash here,
    historically) renders as "???".
    """
    hints = [f"{chr(key)} for {_pretty_label(label)}"
             for key, label in sorted(KeyboardSignalSource.KEY_MAP.items())
             if label in EXERCISE_REGISTRY]
    if not hints:
        return "Rest - no exercises available"
    return "Rest - press " + ", ".join(hints)


class SignalState(Enum):
    NULL = "NULL"        # background / no user — everything off
    REST = "REST"        # user present, idle — system-status audio only
    EXERCISE = "EXERCISE"  # an exercise is active — full pipeline


@dataclass
class FrameState:
    """Per-frame result handed to drawing + audio."""
    signal_state: SignalState
    active: bool                       # is an exercise active this frame
    banner: str | None = None          # shown for Null/Rest
    features: dict | None = None
    zones: dict | None = None
    rep_state: object | None = None    # RepState | None
    # (n, total, done, restarts) — restarts counts discarded calibration buffers
    calib_progress: tuple = (0, BASELINE_FRAMES, False, 0)


class SessionController:
    def __init__(self, audio_controller):
        self.audio = audio_controller
        self.active_exercise = None
        self.signal_state = SignalState.REST
        self._current_label = SIGNAL_REST

        # Calibration lifecycle (only meaningful while an exercise is active).
        self.calib_buffer = []
        self.calib_done = False
        self.calibration = None
        self.calib_restarts = 0        # buffers discarded as contaminated
        self._calib_frames_seen = 0    # pose-detected frames spent calibrating
        self._calib_best = None        # (rejected_count, Calibration) fewest rejections

        # Start audio in the Rest gate (system-status only, silent otherwise).
        self.audio.configure(signal_state=SignalState.REST.value)

    # ── Signal routing / transitions ─────────────────────────────────────────
    def set_signal(self, label: str) -> None:
        """Apply this frame's Module-1 signal, handling any transition."""
        if label == self._current_label and not self._is_unknown_exercise(label):
            return   # no change

        if label == SIGNAL_NULL:
            self._enter_idle(SignalState.NULL, label)
        elif label == SIGNAL_REST:
            self._enter_idle(SignalState.REST, label)
        elif label in EXERCISE_REGISTRY:
            self._enter_exercise(label)
        else:
            # Known signal vocabulary but no implementation yet (e.g. BicepCurl).
            print(f"[SESSION] {label} not implemented yet — staying in "
                  f"{self.signal_state.value}.")
            # Do NOT update _current_label, so it retries the message only on
            # a fresh keypress, and the current state is preserved.

    def _is_unknown_exercise(self, label: str) -> bool:
        return (label not in (SIGNAL_NULL, SIGNAL_REST)
                and label not in EXERCISE_REGISTRY)

    def _enter_idle(self, state: SignalState, label: str) -> None:
        self._teardown()
        self.signal_state = state
        self._current_label = label
        # Fresh audio slate; gate to NULL (silent) or REST (system-status only).
        self.audio.reset()
        self.audio.configure(signal_state=state.value)

    def _enter_exercise(self, label: str) -> None:
        self._teardown()
        self.active_exercise = EXERCISE_REGISTRY[label]()
        self.signal_state = SignalState.EXERCISE
        self._current_label = label
        # Fresh calibration.
        self.calib_buffer = []
        self.calib_done = False
        self.calibration = None
        self.calib_restarts = 0
        self._calib_frames_seen = 0
        self._calib_best = None
        # Fully fresh audio slate, configured with this exercise's cue mapping.
        self.audio.reset()
        self.audio.configure(cue_mapping=self.active_exercise.audio_cue_mapping,
                             signal_state=SignalState.EXERCISE.value)
        print(f"[SESSION] {label} active — {self.active_exercise.calibration_pose_description}")

    def _teardown(self) -> None:
        """Drop the active exercise and all its per-session state."""
        self.active_exercise = None
        self.calib_buffer = []
        self.calib_done = False
        self.calibration = None
        self.calib_restarts = 0
        self._calib_frames_seen = 0
        self._calib_best = None

    # ── Per-frame pipeline ────────────────────────────────────────────────────
    def process_frame(self, lm, pose_detected, w, h, frame_id) -> FrameState:
        """
        Run one frame.  When an exercise is active and a pose is present, this
        mirrors the pre-refactor order exactly:
          (buffer calib sample → maybe finalise calibration) → compute_features
          → classify_zones → (if calibrated) update_rep_counter.
        """
        if self.active_exercise is None:
            banner = ("No user detected" if self.signal_state == SignalState.NULL
                      else rest_banner_text())
            return FrameState(self.signal_state, active=False, banner=banner)

        ex = self.active_exercise
        features = zones = rep_state = None

        if pose_detected:
            if not self.calib_done:
                # Buffer the calibration-pose frame (no visibility gate, matching
                # the pre-refactor calibration block).
                self.calib_buffer.append((lm, w, h))
                self._calib_frames_seen += 1
                if len(self.calib_buffer) >= BASELINE_FRAMES:
                    self._finalise_or_restart_calibration(ex)

            calib_arg = self.calibration if self.calib_done else None
            features = ex.compute_features(lm, calib_arg, w, h)
            zones = ex.classify_zones(features)
            if self.calib_done:
                rep_state = ex.update_rep_counter(features, zones, frame_id, lm)

        progress = (len(self.calib_buffer), BASELINE_FRAMES, self.calib_done,
                    self.calib_restarts)
        return FrameState(self.signal_state, active=True, features=features,
                          zones=zones, rep_state=rep_state, calib_progress=progress)

    def _finalise_or_restart_calibration(self, ex) -> None:
        """
        Accept the full calibration buffer, or DISCARD it and resample.

        An exercise signals a contaminated buffer by returning a Calibration whose
        values carry `valid=False` (the bicep curl rejects any frame whose mean
        elbow angle falls outside its arms-down band).  Exercises with no guard —
        the squat — simply never set the key, so `.get("valid", True)` accepts on
        the first full buffer exactly as before this method existed.

        WHY DISCARD RATHER THAN FALL BACK.  The previous behaviour finalised
        unconditionally on the first full buffer, whatever its state.  Note that a
        rejected frame never pollutes the averages — BicepCurlExercise.calibrate
        skips it before accumulating — so the risk is NOT a contaminated mean.  It
        is SAMPLE SIZE: the baseline is silently computed from however few frames
        survive the guard.  The curl demo clip yields 18 of 30, which is adequate;
        a buffer yielding 3 of 30 would produce an equally "finished" calibration
        from three frames, with nothing but valid=False to distinguish them.  In
        the pathological case where EVERY frame is rejected, calibrate() falls back
        to the geometry of one arbitrary — and by definition contaminated — frame.
        Resampling instead seeks a buffer where all BASELINE_FRAMES frames pass, so
        the baseline rests on a full, uniformly clean sample.

        NEVER-CONVERGING GUARD, BEST-OF.  A user who never holds still would
        otherwise resample forever.  After CALIB_MAX_FRAMES pose-detected frames
        the search stops and the BEST buffer seen across every attempt — the one
        with the fewest rejected frames, and so the largest clean sample — is
        adopted, still flagged valid=False.  Falling back to whichever buffer
        happened to be current when the ceiling hit would decide on arrival order
        alone, letting a 16-rejection buffer beat a 5-rejection one for no reason.
        """
        candidate = ex.calibrate(self.calib_buffer)
        accepted = candidate.values.get("valid", True)

        if accepted:
            self.calibration = candidate
            self.calib_done = True
            if self.calib_restarts:
                print(f"[SESSION] calibration OK after {self.calib_restarts} "
                      f"restart(s), {self._calib_frames_seen} frames.")
            return

        # Keep the cleanest buffer seen so far.  BASELINE_FRAMES is the default
        # so a candidate that somehow omits the count never displaces a real one.
        rejected = candidate.values.get("rejected_frames", BASELINE_FRAMES)
        if self._calib_best is None or rejected < self._calib_best[0]:
            self._calib_best = (rejected, candidate)

        if self._calib_frames_seen >= CALIB_MAX_FRAMES:
            best_rejected, best_calib = self._calib_best
            attempts = self.calib_restarts + 1
            self.calibration = best_calib
            self.calib_done = True
            print(f"[SESSION] WARNING: no clean calibration window in "
                  f"{self._calib_frames_seen} frames over {attempts} attempt(s). "
                  f"Falling back to the BEST buffer seen "
                  f"({best_rejected} of {BASELINE_FRAMES} frames rejected, "
                  f"valid=False) so the session can continue — readings that "
                  f"depend on it (elbow drift, trunk deviation) may be offset.")
            return

        # Contaminated and still within budget: throw the buffer away and resample.
        self.calib_buffer = []
        self.calib_restarts += 1
        rejected = candidate.values.get("rejected_frames", "?")
        print(f"[SESSION] calibration restart {self.calib_restarts}: "
              f"{rejected} of {BASELINE_FRAMES} frames failed the pose guard "
              f"— hold the start position still.")

    # ── Audio ─────────────────────────────────────────────────────────────────
    def update_audio(self, pose_detected, lm, frame_state, now=None):
        """
        Feed the audio controller every frame.  Gating (which categories may
        fire) is governed by the signal_state the controller was configured with
        on the last transition; here we just supply the current frame's data.
        """
        if self.active_exercise is None:
            # Null/Rest: no exercise context.  The controller's gate decides
            # (Null -> silent, Rest -> system-status only).
            return self.audio.update(pose_detected=pose_detected, landmarks=lm,
                                     zones=None, rep_phase=None, rep_count=0,
                                     rep_history=[], now=now)

        ex = self.active_exercise
        rep_phase = ex.phase if self.calib_done else None
        return self.audio.update(pose_detected=pose_detected, landmarks=lm,
                                 zones=frame_state.zones, rep_phase=rep_phase,
                                 rep_count=ex.rep_count, rep_history=ex.rep_history,
                                 now=now)
