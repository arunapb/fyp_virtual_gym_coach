"""
backend/calibration_bank.py — user calibration as its own pipeline stage,
independent of both Module 1's classification and Module 2's session.

The workflow this implements
-----------------------------
    video starts
      -> collect the first BASELINE_FRAMES pose-detected frames
      -> compute the anthropometric baseline (ONCE, for every registered
         exercise, since each defines its own baseline maths)
      -> Module 1 classifies; when it names a valid exercise,
         Module 2 starts using the ALREADY-COMPUTED baseline

Why compute for every exercise rather than waiting to be told which one
-----------------------------------------------------------------------
Calibration is not one universal measurement: each `Exercise` defines its own
`calibrate()` (Squat locks trunk lean + hip height; Bicep Curl and Shoulder
Press additionally lock each arm's natural resting/racked elbow offset, and
reject a buffer that isn't in their required start pose). So "calibrate the
user first, decide the exercise later" is only possible by evaluating every
registered exercise's `calibrate()` against the same captured frames and
keeping all the results — then handing over whichever one Module 1 ends up
naming. Each is a pure function of the buffer (verified across all three
implementations), so running them all costs one pass over 30 buffered frames
and nothing else; no exercise instance created here is ever used for
analysis, only for its calibration maths.

Nothing in Module 2 is modified. `apply_to()` sets exactly the three
attributes `SessionController._finalise_or_restart_calibration()` sets on a
normal accept (`calib_buffer`, `calibration`, `calib_done`), with a value
produced by the exercise's own unmodified `calibrate()` and accepted only if
it passes that exercise's own `valid` check — so a rejected baseline falls
through to Module 2's standard collect-and-resample path untouched.
"""

from config import BASELINE_FRAMES
from src.session import EXERCISE_REGISTRY


class CalibrationBank:
    """One per analysis run. Fed every pose-detected frame; self-completing."""

    def __init__(self, baseline_frames: int = BASELINE_FRAMES):
        self.baseline_frames = baseline_frames
        self.buffer = []
        self.calibrations = {}      # exercise label -> Calibration
        self.done = False
        self.completed_frame = None

    @property
    def collected(self) -> int:
        return len(self.buffer)

    def observe(self, lm, w, h, frame_id) -> None:
        """Buffer one pose-detected frame; compute the baselines when full."""
        if self.done:
            return
        self.buffer.append((lm, w, h))
        if len(self.buffer) >= self.baseline_frames:
            self._compute(frame_id)

    def _compute(self, frame_id) -> None:
        for label, cls in EXERCISE_REGISTRY.items():
            try:
                self.calibrations[label] = cls().calibrate(self.buffer)
            except Exception as exc:
                # One exercise's calibration maths failing must not sink the
                # run: the others stay usable, and this one falls back to
                # Module 2's own in-session calibration if it is the one
                # Module 1 picks.
                print(f"[CALIB] {label} baseline failed: {exc}", flush=True)
        self.done = True
        self.completed_frame = frame_id
        accepted = [k for k, v in self.calibrations.items()
                    if v.values.get("valid", True)]
        print(f"[CALIB] baseline captured from the first {len(self.buffer)} "
              f"tracked frames; usable for: {', '.join(accepted) or 'none'}",
              flush=True)

    def is_valid_for(self, label: str) -> bool:
        calib = self.calibrations.get(label)
        return calib is not None and calib.values.get("valid", True)

    def apply_to(self, sessionc, label: str) -> bool:
        """
        Hand the pre-computed baseline to a just-activated exercise.

        Returns True if Module 2 is now calibrated and can analyse from this
        frame on; False if it must run its own calibration (no baseline for
        this exercise, or this exercise's own guard rejected it).
        """
        if not self.is_valid_for(label) or sessionc.calib_done:
            return False
        sessionc.calib_buffer = list(self.buffer)
        sessionc.calibration = self.calibrations[label]
        sessionc.calib_done = True
        return True

    def summary(self, fps, label=None) -> dict:
        """
        Calibration report in the same shape `module2/src/analysis.py`'s
        `_calibration_summary()` produces, so the result document and the UI
        tile read it without special-casing.
        """
        calib = self.calibrations.get(label) if label else None
        if calib is None and self.calibrations:
            calib = next(iter(self.calibrations.values()))
        values = calib.values if calib else {}
        return {
            "done": self.done,
            "frames_required": self.baseline_frames,
            "restarts": 0,
            "completed_frame": self.completed_frame,
            "completed_time": (round(self.completed_frame / fps, 2)
                               if self.completed_frame is not None and fps else None),
            "clean": bool(values.get("valid", True)),
            "rejected_frames": values.get("rejected_frames"),
            "values": {k: (round(v, 4) if isinstance(v, (int, float))
                           and not isinstance(v, bool) else v)
                       for k, v in values.items()},
        }
