"""
rep_counter.py — Squat rep counting and phase detection.

State machine driven by the smoothed average knee angle and its smoothed
first-derivative (velocity).

Phase cycle that constitutes one rep:
  STANDING → DESCENDING → BOTTOM → ASCENDING → STANDING

Rules
─────
• STANDING → DESCENDING requires the knee angle to drop below KNEE_DESCENDING
  AND a clearly negative velocity (angle actively decreasing), preventing
  standing-still jitter from starting phantom reps.
• A transition from DESCENDING → BOTTOM only fires when the knee angle has
  dropped below KNEE_BOTTOM, ensuring shallow bends are ignored.
• BOTTOM → ASCENDING requires the phase to be held for at least
  BOTTOM_MIN_DWELL_FRAMES frames (~400 ms at 30 FPS), so momentary dips do
  not count as completed reps.
• Every other phase requires REP_PHASE_MIN_FRAMES frames of dwell time before
  the next transition can fire, preventing jitter-driven false transitions.
• If the knee re-extends to KNEE_STANDING during DESCENDING without ever
  reaching KNEE_BOTTOM, the descent is judged a SHALLOW rep — counted, logged
  with rep_quality red and shallow=1, and armed as a synthetic red depth zone
  so the audio layer can cue it — provided it got at least as deep as
  KNEE_SHALLOW_ATTEMPT and lasted SHALLOW_MIN_DESCENT_SEC.  Shorter or
  shallower descents are still discarded as postural noise.
• All smoothing windows and dwells are derived from the SOURCE frame rate
  passed to __init__, not fixed frame counts (see config.py's TIMEBASE).
• Frames where MediaPipe visibility scores for hips or knees fall below
  LANDMARK_MIN_VISIBILITY are skipped entirely — no state update occurs.
• Per-rep statistics (min knee angle, worst zones, trunk deviation) are
  accumulated across DESCENDING + BOTTOM + ASCENDING, then packaged into a
  summary dict when the rep completes.
"""

from collections import deque
from enum import Enum

from config import (
    KNEE_STANDING,
    KNEE_DESCENDING,
    KNEE_BOTTOM,
    KNEE_SHALLOW_ATTEMPT,
    SHALLOW_MIN_DESCENT_SEC,
    SHALLOW_DEPTH_CUE_HOLD_SEC,
    REP_VELOCITY_DEAD_ZONE_DEG_PER_SEC,
    REP_ANGLE_SMOOTH_SEC,
    REP_VELOCITY_SMOOTH_SEC,
    REP_PHASE_MIN_SEC,
    BOTTOM_MIN_DWELL_SEC,
    LANDMARK_MIN_VISIBILITY,
    DEBUG_REP_COUNTER,
    FPS_REFERENCE, frames_at,
    IDX_HIP_L, IDX_HIP_R, IDX_KNEE_L, IDX_KNEE_R,
)
from src.zones import overall_zone, squat_depth_zones
from src.pose_utils import landmarks_reliable


class RepPhase(Enum):
    STANDING   = "STANDING"
    DESCENDING = "DESCENDING"
    BOTTOM     = "BOTTOM"
    ASCENDING  = "ASCENDING"


_ZONE_RANK = {"green": 0, "yellow": 1, "red": 2}
_RANK_ZONE = {v: k for k, v in _ZONE_RANK.items()}


def _worse(a, b):
    """Return whichever of two zone strings carries higher risk."""
    return _RANK_ZONE[max(_ZONE_RANK[a], _ZONE_RANK[b])]


class RepCounter:
    """
    Tracks squat phases and counts completed reps in real time.

    Usage
    ─────
    counter = RepCounter()
    # inside the main frame loop (call only after calibration is done):
    phase = counter.update(feats, zones, frame_id, landmarks=lm)

    Attributes exposed after each update()
    ───────────────────────────────────────
    phase            : RepPhase   — current movement phase
    rep_count        : int        — total completed reps
    last_rep_summary : dict|None  — stats for the most recently completed rep
    rep_history      : list[dict] — stats for every completed rep (oldest first)
    """

    def __init__(self, fps=FPS_REFERENCE):
        # Every window and dwell below is derived from the SOURCE frame rate, so
        # a 15 fps clip and a 30 fps clip smooth over the same real interval and
        # reach the same phase at the same moment of the movement.  See the
        # TIMEBASE section of config.py for what a fixed frame count did to the
        # BOTTOM phase on 15 fps footage.
        self.fps = fps if fps and fps > 0 else FPS_REFERENCE
        self._dead_zone   = REP_VELOCITY_DEAD_ZONE_DEG_PER_SEC / self.fps
        self._phase_min   = frames_at(REP_PHASE_MIN_SEC,          self.fps)
        self._bottom_min  = frames_at(BOTTOM_MIN_DWELL_SEC,       self.fps)
        self._shallow_min = frames_at(SHALLOW_MIN_DESCENT_SEC,    self.fps)
        self._shallow_hold_frames = frames_at(SHALLOW_DEPTH_CUE_HOLD_SEC, self.fps)

        self.phase     = RepPhase.STANDING
        self.rep_count = 0

        # Smoothing buffers
        self._angle_buf   = deque(maxlen=frames_at(REP_ANGLE_SMOOTH_SEC, self.fps))
        self._vel_buf     = deque(maxlen=frames_at(REP_VELOCITY_SMOOTH_SEC, self.fps))
        self._prev_smooth = None

        # Frames remaining on each synthetic red depth zone, armed at rep
        # completion and ticked down by the matching consume_*() call.  Two
        # counters rather than one because "go deeper" and "not so deep" are
        # opposite corrections needing different clips.
        self._shallow_cue_frames = 0
        self._excess_cue_frames  = 0

        # Frames elapsed since the most recent phase transition
        self._frames_in_phase = 0

        # Per-rep accumulators (reset at start of every new descent)
        self._rep_start     = 0
        self._min_angle     = 180.0
        self._max_trunk_dev = 0.0
        self._worst = {"valgus_l": "green", "valgus_r": "green",
                       "trunk":    "green", "depth":    "green"}

        # Running sums for mean computation  (n = frames accumulated)
        self._sum_knee      = 0.0
        self._sum_trunk_dev = 0.0
        self._sum_valgus_l  = 0.0
        self._sum_valgus_r  = 0.0
        self._sum_depth     = 0.0
        self._n_frames      = 0
        # Numeric peak valgus (most extreme signed offset seen in this rep)
        self._peak_valgus_l = 0.0   # max positive → worst left valgus
        self._peak_valgus_r = 0.0   # min negative → worst right valgus

        self.rep_history      = []
        self.last_rep_summary = None

        # Session-level visibility skip counters
        self._skipped_frames = 0
        self._total_frames   = 0

        # Scratch-pad for the current frame's values (used by debug logging)
        self._dbg_frame  = 0
        self._dbg_raw    = 0.0
        self._dbg_smooth = 0.0
        self._dbg_vel    = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, feats, zones, frame_id, landmarks=None):
        """
        Advance the state machine by one frame.

        Parameters
        ----------
        feats     : dict      — output of measurements.compute_squat_features()
        zones     : dict      — output of zones.compute_zones()
        frame_id  : int       — current frame index (used in per-rep records)
        landmarks : list|None — MediaPipe NormalizedLandmark list for this frame.
                                When provided, frames where any of the four key
                                landmarks (hips + knees) have visibility below
                                LANDMARK_MIN_VISIBILITY are skipped entirely.

        Returns
        -------
        RepPhase — the current phase after this update
        """
        self._total_frames += 1

        # Visibility gate — skip unreliable frames without advancing the state.
        if landmarks is not None and not landmarks_reliable(
            landmarks,
            [IDX_HIP_L, IDX_HIP_R, IDX_KNEE_L, IDX_KNEE_R],
            LANDMARK_MIN_VISIBILITY,
        ):
            self._skipped_frames += 1
            return self.phase

        raw    = (feats["knee_angle_l_deg"] + feats["knee_angle_r_deg"]) / 2.0
        smooth = self._smooth_angle(feats)
        vel    = self._velocity(smooth)
        self._frames_in_phase += 1

        # Store current-frame values so _enter() can include them in debug logs
        self._dbg_frame  = frame_id
        self._dbg_raw    = raw
        self._dbg_smooth = smooth
        self._dbg_vel    = vel

        if self.phase == RepPhase.STANDING:
            self._state_standing(smooth, vel, frame_id)

        elif self.phase == RepPhase.DESCENDING:
            self._accumulate(smooth, feats, zones)
            self._state_descending(smooth, vel, frame_id)

        elif self.phase == RepPhase.BOTTOM:
            self._accumulate(smooth, feats, zones)
            self._state_bottom(vel)

        elif self.phase == RepPhase.ASCENDING:
            self._accumulate(smooth, feats, zones)
            self._state_ascending(smooth, frame_id)

        return self.phase

    def print_session_summary(self):
        """Print a one-line visibility-skip summary for this recording session."""
        if self._total_frames > 0:
            pct = 100.0 * self._skipped_frames / self._total_frames
            print(
                f"[REP COUNTER] Skipped {self._skipped_frames} of "
                f"{self._total_frames} frames due to low visibility "
                f"({pct:.1f}%)"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # State handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _state_standing(self, angle, vel, frame_id):
        # Require both: angle below trigger AND actively descending.
        # Velocity must be negative (angle decreasing) with magnitude > dead zone
        # to prevent standing-still jitter from starting a phantom rep.
        if angle < KNEE_DESCENDING and vel < -self._dead_zone:
            self._enter(RepPhase.DESCENDING, "below_descending_threshold_and_negative_velocity")
            self._rep_start = frame_id
            self._reset_stats()

    def _state_descending(self, angle, vel, frame_id):
        # Rose back to standing without reaching valid depth.
        #
        # This branch used to discard the descent outright — no count, no
        # summary, no cue — which is what made a partial squat structurally
        # invisible rather than merely ungraded.  A descent that got far enough
        # to be a real attempt is now recorded as a SHALLOW rep instead, and
        # arms the synthetic red depth zone that lets the audio layer say
        # "go deeper".  Anything shorter or shallower than the guards is still
        # discarded: that is postural noise, not a squat.
        if angle >= KNEE_STANDING:
            genuine_attempt = (self._min_angle <= KNEE_SHALLOW_ATTEMPT
                               and self._n_frames >= self._shallow_min)
            if genuine_attempt:
                self.rep_count += 1
                # _finalise_rep arms the "go deeper" hold itself, off the same
                # depth band every other rep is graded by — a descent stopping
                # above KNEE_BOTTOM (130) necessarily sits above
                # DEPTH_KNEE_SHALLOW_RED (125), so there is one code path
                # deciding depth rather than two that could disagree.
                self._finalise_rep(frame_id, shallow=True)
                self._enter(RepPhase.STANDING, "shallow_rep_rose_before_bottom")
            else:
                self._enter(RepPhase.STANDING, "aborted_rep_rose_before_bottom")
            self._reset_stats()
            return

        # Use the running minimum (not the instantaneous angle) so a brief dip
        # below the threshold is captured even while the person is still moving.
        # Transition when: valid depth was reached at any prior frame in this
        # descent AND motion has now slowed or reversed.
        if (self._min_angle < KNEE_BOTTOM
                and vel >= -self._dead_zone
                and self._frames_in_phase >= self._phase_min):
            self._enter(RepPhase.BOTTOM, "reached_depth_and_velocity_slowed")

    def _state_bottom(self, vel):
        # Wait for sustained upward velocity AND the minimum real-time dwell
        # before leaving BOTTOM, so momentary dips don't count as completed reps.
        if (vel >= self._dead_zone
                and self._frames_in_phase >= self._bottom_min):
            self._enter(RepPhase.ASCENDING, "sustained_upward_velocity_after_dwell")

    def _state_ascending(self, angle, frame_id):
        if angle >= KNEE_STANDING:
            self.rep_count += 1
            self._finalise_rep(frame_id)
            self._enter(RepPhase.STANDING, "returned_to_standing_angle")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _smooth_angle(self, feats):
        avg = (feats["knee_angle_l_deg"] + feats["knee_angle_r_deg"]) / 2.0
        self._angle_buf.append(avg)
        return sum(self._angle_buf) / len(self._angle_buf)

    def _velocity(self, smooth):
        """Smoothed first derivative of knee angle (degrees / frame)."""
        if self._prev_smooth is not None:
            self._vel_buf.append(smooth - self._prev_smooth)
        self._prev_smooth = smooth
        return sum(self._vel_buf) / len(self._vel_buf) if self._vel_buf else 0.0

    def _enter(self, new_phase, reason=""):
        if DEBUG_REP_COUNTER:
            print(
                f"[REP STATE] Frame {self._dbg_frame:05d}: "
                f"{self.phase.value} -> {new_phase.value} "
                f"(raw={self._dbg_raw:.1f} smooth={self._dbg_smooth:.1f} "
                f"vel={self._dbg_vel:.1f}) "
                f"reason={reason}"
            )
        self.phase = new_phase
        self._frames_in_phase = 0

    def _accumulate(self, smooth, feats, zones):
        self._min_angle     = min(self._min_angle, smooth)
        trunk_dev           = abs(feats.get("trunk_lean_dev_deg", 0.0))
        self._max_trunk_dev = max(self._max_trunk_dev, trunk_dev)

        # DEPTH IS DELIBERATELY EXCLUDED from the per-frame _worst latch below.
        # "Did this rep reach the right depth" is a question about the bottom of
        # the movement — exactly like min_knee_angle, which _min_angle above
        # already tracks — so _finalise_rep answers it from that minimum
        # instead.  Latching the worst per-frame zone answers a different
        # question, and answers it wrongly on a fast descent: the depth zone's
        # BOTTOM gate opens as soon as the smoothed knee passes 110 deg, and for
        # a frame or two around that crossing the reading is still catching up.
        # Measured on job 75493c29e9d3 at frame 107 the zone read red while the
        # knee was already at 88.5 deg, two frames before the genuine bottom;
        # latching it marked both of that clip's full-depth squats as red reps.
        if zones:
            for key in ("valgus_l", "valgus_r", "trunk"):
                self._worst[key] = _worse(self._worst[key], zones[key])

        # Running sums for per-rep means
        vl = feats.get("norm_knee_offset_l", 0.0)
        vr = feats.get("norm_knee_offset_r", 0.0)
        self._sum_knee      += smooth
        self._sum_trunk_dev += trunk_dev
        self._sum_valgus_l  += vl
        self._sum_valgus_r  += vr
        self._sum_depth     += feats.get("norm_hip_depth", 0.0)
        self._peak_valgus_l  = max(self._peak_valgus_l, vl)
        self._peak_valgus_r  = min(self._peak_valgus_r, vr)
        self._n_frames      += 1

    def consume_shallow_cue(self) -> bool:
        """
        True while the synthetic red "too shallow" depth zone is still held,
        ticking the hold down by one frame.

        Exists because a depth verdict is EDGE-triggered — only knowable once
        the rep has ended — while the audio controller is LEVEL-triggered,
        counting consecutive red frames.  Holding the verdict for longer than
        CORRECTIVE_DWELL_SEC is what lets the two meet.  Same mechanism as the
        curl's CURL_ROM_CUE_HOLD_FRAMES.
        """
        if self._shallow_cue_frames <= 0:
            return False
        self._shallow_cue_frames -= 1
        return True

    def consume_excess_depth_cue(self) -> bool:
        """
        True while the synthetic red "too deep" zone is still held, ticking the
        hold down by one frame.  Counterpart to consume_shallow_cue(); see there.
        """
        if self._excess_cue_frames <= 0:
            return False
        self._excess_cue_frames -= 1
        return True

    def _finalise_rep(self, end_frame, shallow=False):
        # Depth verdict from the DEEPEST point of the rep — see _accumulate for
        # why this is re-derived here instead of latched frame by frame, and
        # config.py for why it is taken from the knee angle rather than the hip
        # depth ratio.  A rep whose descent was never accumulated (every frame
        # gated out by the visibility check) leaves _min_angle at its 180.0
        # sentinel, which would read as maximally shallow; that is missing data,
        # not a fault, so it is left green.
        depth_shallow, depth_excess = "green", "green"
        if self._n_frames:
            depth_shallow, depth_excess = squat_depth_zones(self._min_angle)
        self._worst["depth"] = overall_zone([depth_shallow, depth_excess])

        # Arm whichever correction this rep earned, as a synthetic zone held
        # long enough for the level-triggered audio controller to see it.  The
        # two are mutually exclusive by construction (an angle cannot be both
        # above and below the band), so they can never both speak.
        if depth_shallow == "red":
            self._shallow_cue_frames = self._shallow_hold_frames
        elif depth_excess == "red":
            self._excess_cue_frames = self._shallow_hold_frames

        quality = overall_zone(list(self._worst.values()))
        if shallow:
            # A rep that never reached depth is a failed rep whatever the other
            # channels say: its worst_depth is set red rather than left at
            # whatever the phase-gated depth zone happened to report, because
            # that zone is only scored at BOTTOM and a shallow rep never gets
            # there (see DEPTH_PHASE_BOTTOM_THRESHOLD's note on the circularity).
            self._worst["depth"] = "red"
            quality = "red"
        n       = max(self._n_frames, 1)
        summary = {
            "rep_num":          self.rep_count,
            "start_frame":      self._rep_start,
            "end_frame":        end_frame,
            "duration_frames":  end_frame - self._rep_start,
            # Knee angle
            "min_knee_angle":   round(self._min_angle, 1),
            "mean_knee_angle":  round(self._sum_knee / n, 1),
            # Trunk deviation
            "max_trunk_dev":    round(self._max_trunk_dev, 2),
            "mean_trunk_dev":   round(self._sum_trunk_dev / n, 2),
            # Valgus — worst-case numeric offsets + mean per side
            "peak_valgus_l":    round(self._peak_valgus_l, 3),
            "mean_valgus_l":    round(self._sum_valgus_l  / n, 3),
            "peak_valgus_r":    round(self._peak_valgus_r, 3),
            "mean_valgus_r":    round(self._sum_valgus_r  / n, 3),
            # Hip depth
            "mean_depth":       round(self._sum_depth / n, 3),
            # Worst zone per metric
            "worst_valgus_l":   self._worst["valgus_l"],
            "worst_valgus_r":   self._worst["valgus_r"],
            "worst_trunk":      self._worst["trunk"],
            "worst_depth":      self._worst["depth"],
            "shallow":          int(shallow),
            "rep_quality":      quality,
        }
        self.rep_history.append(summary)
        self.last_rep_summary = summary

    def _reset_stats(self):
        self._min_angle     = 180.0
        self._max_trunk_dev = 0.0
        self._worst         = {k: "green" for k in self._worst}
        self._sum_knee      = 0.0
        self._sum_trunk_dev = 0.0
        self._sum_valgus_l  = 0.0
        self._sum_valgus_r  = 0.0
        self._sum_depth     = 0.0
        self._peak_valgus_l = 0.0
        self._peak_valgus_r = 0.0
        self._n_frames      = 0
