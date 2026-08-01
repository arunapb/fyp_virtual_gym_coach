"""
exercises/bicep_curl.py — BicepCurlExercise (Level 1: live coaching only).

Second concrete Exercise, after SquatExercise.  Handles BOTH curl modes with a
single mechanism: each arm runs its OWN independent rep state machine, so an
alternating curl (the common way people curl dumbbells, and the only mode
present in this project's dataset) counts correctly, and a simultaneous curl is
simply the case where the two machines happen to advance in lockstep.  No mode
detection is needed or performed.

SCOPE NOTE — every threshold consumed here is a feel-tuned initial value from
the "BICEP CURL THRESHOLDS" block in config.py.  Unlike the squat thresholds
(percentiles of REHAB24-6 Ex6 correct-rep frames) NONE of them is derived from
annotated data, and no fault-detection evaluation has been run.  They must not
be reported as validated.  ELBOW_DRIFT_YELLOW/RED are additionally STALE: they
were tuned against the un-baselined drift metric and must be re-derived.

Design constraints honoured by this module:
  * measurements.py / zones.py / rep_counter.py are squat-specific and are NOT
    imported — the curl's geometry, zone classifiers and state machine live
    here.  Only the generic helpers in pose_utils.py are reused.
  * Nothing outside exercises/ and the curl block of config.py is modified; the
    class satisfies the existing Exercise interface, so session / drawing /
    audio consume it unchanged.
"""

from collections import deque

from config import (
    SMOOTH_N, OVERALL_LABEL,
    CURL_CSV_HEADER, CURL_PHASE_COLORS, POSE_CONNECTIONS,
    CURL_CSV, CURL_REP_SUMMARY_CSV, CURL_REP_SUMMARY_HEADER,
    CURL_LOWERING_STALL_FRAMES,
    LANDMARK_MIN_VISIBILITY,
    IDX_SHOULDER_L, IDX_SHOULDER_R, IDX_ELBOW_L, IDX_ELBOW_R,
    IDX_WRIST_L, IDX_WRIST_R, IDX_HIP_L, IDX_HIP_R,
    ELBOW_DRIFT_YELLOW, ELBOW_DRIFT_RED,
    BODY_SWING_YELLOW, BODY_SWING_RED,
    CURL_ARM_PEAK_DIFF_YELLOW, CURL_ARM_PEAK_DIFF_RED,
    CURL_ARM_ROM_DIFF_YELLOW, CURL_ARM_ROM_DIFF_RED,
    ROM_MIN_ACCEPTABLE,
    CURL_EXTENDED, CURL_CURLING, CURL_CONTRACTED, CURL_LOWERING,
    CURL_CONTRACTED_MIN_DWELL_FRAMES,
    CURL_CALIB_ELBOW_MIN, CURL_CALIB_ELBOW_MAX,
    CURL_ROM_CUE_HOLD_FRAMES,
    AUDIO_CURL_ELBOWS_PINNED, AUDIO_CURL_STOP_SWINGING,
    AUDIO_CURL_MATCH_ARMS, AUDIO_CURL_FULL_EXTENSION, AUDIO_CURL_HIGHER,
)
from src.pose_utils import (
    get_px, midpoint, dist2d, safe_div, angle_at_joint, trunk_lean_angle,
    landmarks_reliable,
)

from src.exercises.base import (
    Exercise, Calibration, RepState, DisplaySpec, InfoCell, AudioCueMapping,
)


# Aura body-region groupings.  Arms are the story for this exercise, so each
# arm chain is coloured by its own drift zone and the legs stay neutral.
_LEFT_ARM  = [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21)]
_RIGHT_ARM = [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22)]
_TORSO     = [(11, 12), (11, 23), (12, 24), (23, 24)]
_SPECIAL   = set(_LEFT_ARM) | set(_RIGHT_ARM) | set(_TORSO)
_OTHER     = [(a, b) for a, b in POSE_CONNECTIONS if (a, b) not in _SPECIAL]

_VIS_JOINTS = [IDX_SHOULDER_L, IDX_SHOULDER_R, IDX_ELBOW_L, IDX_ELBOW_R,
               IDX_WRIST_L, IDX_WRIST_R]

_VELOCITY_N = 5      # frames over which elbow-angle velocity is estimated
_VEL_DEAD_ZONE = 0.5  # deg/frame; below this magnitude motion counts as stopped

_ALL_PHASES = {"EXTENDED", "CURLING", "CONTRACTED", "LOWERING"}

# Rank used to pick the single REPRESENTATIVE phase reported through the
# Exercise.phase property.  The audio controller gates each cue on one phase
# string, so with two independent arms the more ACTIVE arm is reported: that
# keeps "only cue while the user is actually moving" working in both modes.
# The full per-arm phases are shown on screen and logged to CSV separately.
_PHASE_RANK = {"EXTENDED": 0, "LOWERING": 1, "CURLING": 2, "CONTRACTED": 3}

# Compact on-screen phase labels, so "L:CONTR  R:EXT" fits the info-bar column.
_PHASE_SHORT = {"EXTENDED": "EXT", "CURLING": "CURL",
                "CONTRACTED": "CONTR", "LOWERING": "LOWER"}


def _worse(a, b):
    order = {"green": 0, "yellow": 1, "red": 2}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def _overall(zone_list):
    if "red" in zone_list:
        return "red"
    if "yellow" in zone_list:
        return "yellow"
    return "green"


def _band(value, yellow, red):
    if value > red:
        return "red"
    if value > yellow:
        return "yellow"
    return "green"


def _oriented_drift(sh_l, sh_r, el_l, el_r, shoulder_width):
    """
    Raw outward-oriented elbow drift for both arms, normalised by shoulder width.

    ORIENTATION — the bare (elbow_x - shoulder_x) difference is MIRROR-SIGNED
    between the two arms: an elbow flaring away from the torso INCREASES x on one
    side and DECREASES it on the other.  Classifying that with a single
    positive-is-bad rule leaves one arm's channel structurally dead.  Each side is
    signed by which side of the shoulder midline its own shoulder sits on, so
    positive ALWAYS means "flared outward" for both arms.  Deriving the sign from
    the observed shoulders rather than hard-coding it also keeps the metric correct
    whether or not the frame was mirrored (live_runner flips webcam, not video).

    This is the RAW value.  The per-user resting offset is subtracted separately
    (see compute_features) — shared here so calibrate() and compute_features()
    cannot drift apart in how they orient the sign.
    """
    sh_mid_x = (sh_l[0] + sh_r[0]) / 2.0
    side_l = 1.0 if sh_l[0] >= sh_mid_x else -1.0
    side_r = 1.0 if sh_r[0] >= sh_mid_x else -1.0
    return (safe_div((el_l[0] - sh_l[0]) * side_l, shoulder_width),
            safe_div((el_r[0] - sh_r[0]) * side_r, shoulder_width))


class _ArmRepMachine:
    """
    One arm's independent EXTENDED -> CURLING -> CONTRACTED -> LOWERING cycle.

    Two of these run side by side, one per elbow, each with its own phase,
    frames-in-phase counter, velocity window, dwell tracking, abort path, ROM
    cue and rep tally.  Every guard that the previous single mean-driven machine
    applied is applied here per arm, unchanged.

    Independence is what makes alternating curls work: the left machine can be
    mid-CURLING while the right sits in EXTENDED, and each counts its own reps.
    A simultaneous curl needs no special handling — both machines simply advance
    together.
    """

    def __init__(self, arm):
        self.arm = arm                                   # "L" or "R"
        self._angle_key = f"elbow_angle_{arm.lower()}_deg"
        self._drift_key = f"elbow_drift_{arm.lower()}"
        self.phase = "EXTENDED"
        self.frames_in_phase = 0
        self.rep_count = 0
        self.rep_history = []
        self.last_rep_summary = None
        self._vel_buf = deque(maxlen=_VELOCITY_N)
        self._prev_angle = None
        self._rep_start_frame = 0
        self._reached_contracted = False
        self._rom_cue_frames = 0
        self._reset_rep_stats()

    # ── Per-rep accumulators ─────────────────────────────────────────────────
    def _reset_rep_stats(self):
        self._min = 180.0
        self._max = 0.0
        self._peak_drift = 0.0
        self._peak_trunk = 0.0
        # rep_quality is built from THIS arm's drift plus the shared body-swing
        # channel.  The asymmetry channel is deliberately excluded: it is now a
        # comparison BETWEEN the two arms' completed reps, so folding it in here
        # would judge this rep using a verdict derived from earlier reps.
        self._worst = {"drift": "green", "body_swing": "green"}

    def _accumulate(self, f, z):
        angle = f[self._angle_key]
        self._min = min(self._min, angle)
        self._max = max(self._max, angle)
        self._peak_drift = max(self._peak_drift, f[self._drift_key])
        self._peak_trunk = max(self._peak_trunk, f["trunk_deviation_deg"])
        if z:
            self._worst["drift"] = _worse(self._worst["drift"],
                                          z.get(self._drift_key, "green"))
            self._worst["body_swing"] = _worse(self._worst["body_swing"],
                                               z.get("body_swing", "green"))

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _velocity(self, angle):
        if self._prev_angle is not None:
            self._vel_buf.append(angle - self._prev_angle)
        self._prev_angle = angle
        return sum(self._vel_buf) / len(self._vel_buf) if self._vel_buf else 0.0

    def _enter(self, phase):
        self.phase = phase
        self.frames_in_phase = 0

    def is_stalled(self):
        """This arm has hovered in LOWERING without ever straightening."""
        return (self.phase == "LOWERING"
                and self.frames_in_phase >= CURL_LOWERING_STALL_FRAMES)

    def consume_rom_cue(self):
        """Tick down and report this arm's edge-triggered 'curl higher' hold."""
        if self._rom_cue_frames > 0:
            self._rom_cue_frames -= 1
            return True
        return False

    # ── State machine ────────────────────────────────────────────────────────
    def update(self, angle, features, zones, frame_id, global_rep_number):
        """
        Advance one frame.  `angle` is this arm's SMOOTHED elbow angle, which
        decreases as the weight rises.  Returns a completed-rep summary or None.
        """
        vel = self._velocity(angle)
        self.frames_in_phase += 1
        completed = None

        if self.phase != "EXTENDED":
            self._accumulate(features, zones)

        if self.phase == "EXTENDED":
            # Commit to a lift only on a genuine downward sweep, so an arm held
            # slightly bent at rest cannot start a phantom rep.
            if angle < CURL_CURLING and vel < -_VEL_DEAD_ZONE:
                self._rep_start_frame = frame_id
                self._reset_rep_stats()
                self._reached_contracted = False
                self._accumulate(features, zones)
                self._enter("CURLING")

        elif self.phase == "CURLING":
            if angle >= CURL_EXTENDED:
                self._enter("EXTENDED")            # aborted — never neared the top
            elif angle <= CURL_CONTRACTED:
                self._reached_contracted = True
                self._enter("CONTRACTED")

        elif self.phase == "CONTRACTED":
            # Dwell gate REJECTS bounce reps rather than merely delaying them.
            # The test is "did the arm LEAVE the contracted region before the
            # minimum hold was satisfied", i.e. angle back above CURL_LOWERING —
            # NOT "did it return all the way to CURL_EXTENDED".  The driving
            # angle is smoothed and lags the raw signal by about the smoothing
            # window, so a full return to 155 deg can essentially never register
            # inside 8 frames; keying the gate on that would pass every bounce.
            if angle > CURL_LOWERING and vel > _VEL_DEAD_ZONE:
                if self.frames_in_phase < CURL_CONTRACTED_MIN_DWELL_FRAMES:
                    self._reached_contracted = False
                self._enter("LOWERING")
            elif angle >= CURL_EXTENDED:
                # Snapped straight back past the lowering band in one window.
                self._reached_contracted = False
                self._enter("EXTENDED")

        elif self.phase == "LOWERING":
            if angle >= CURL_EXTENDED:
                if self._reached_contracted:
                    self.rep_count += 1
                    completed = self._finalise(frame_id, global_rep_number)
                self._enter("EXTENDED")

        return completed

    def _finalise(self, end_frame, global_rep_number):
        rep_range = self._max - self._min
        rom_incomplete = rep_range < ROM_MIN_ACCEPTABLE

        quality = _overall(list(self._worst.values()))
        if rom_incomplete:
            # A short rep degrades quality one step: a technically clean but
            # partial rep should not be reported as good form.
            quality = {"green": "yellow", "yellow": "red", "red": "red"}[quality]
            # Arm the "curl higher" cue for a TOP-end shortfall.
            # NOTE: with independent per-arm machines this branch is now close to
            # unreachable, because an arm must drive its own smoothed angle to
            # <= CURL_CONTRACTED to count a rep at all, which forces self._min
            # below that threshold.  The fault it used to describe — one arm not
            # curling as high as the other — is now caught properly by the
            # per-rep asymmetry channel instead, where that arm simply logs a
            # smaller peak flexion (or no rep at all).  Left in place because it
            # is still the correct verdict if ROM_MIN_ACCEPTABLE is re-derived
            # upward; see the report note accompanying this change.
            if self._min > CURL_CONTRACTED:
                self._rom_cue_frames = CURL_ROM_CUE_HOLD_FRAMES

        summary = {
            "rep_number":      global_rep_number,
            "arm":             self.arm,
            "arm_rep_number":  self.rep_count,
            "start_frame":     self._rep_start_frame,
            "end_frame":       end_frame,
            "duration_frames": end_frame - self._rep_start_frame,
            "min_elbow":       round(self._min, 1),
            "max_elbow":       round(self._max, 1),
            "rep_range":       round(rep_range, 1),
            "peak_drift":      round(self._peak_drift, 3),
            "peak_trunk_deviation": round(self._peak_trunk, 2),
            "rom_incomplete":  int(rom_incomplete),
            "rep_quality":     quality,
        }
        self.rep_history.append(summary)
        self.last_rep_summary = summary
        return summary


class BicepCurlExercise(Exercise):
    name = "BicepCurl"
    calibration_pose_description = (
        "Stand with arms straight down at your sides. Hold still."
    )

    def __init__(self):
        # Rolling smoothers, fresh per activation — same 5-frame pattern the
        # squat uses, so zone colours do not flicker on landmark jitter.
        # elbow_l / elbow_r each drive their own state machine.
        self._smoother = {k: deque(maxlen=SMOOTH_N) for k in (
            "drift_l", "drift_r", "trunk_dev", "elbow_l", "elbow_r")}

        self._left = _ArmRepMachine("L")
        self._right = _ArmRepMachine("R")

        self._rep_history = []          # both arms, in completion order
        self._last_rep_summary = None
        self._prev_rep_count = 0

        # Latest per-rep asymmetry comparison, for the info bar.
        self._arm_peak_diff = None
        self._arm_rom_diff = None

        # Visibility gate bookkeeping
        self._skipped_frames = 0
        self._total_frames = 0

    # ── Calibration ──────────────────────────────────────────────────────────
    def calibrate(self, landmarks_buffer):
        """
        Average the arms-down reference pose over the buffered frames.

        As well as the segment lengths and trunk lean, this locks
        baseline_elbow_drift_l/r — the subject's NATURAL resting elbow offset.
        Elbows do not hang directly beneath the shoulders, so the raw drift is
        already ~0.2 of shoulder width at rest for some subjects; without this
        baseline the drift channel reads red before any fault exists.  This is
        the same treatment baseline_trunk_lean already receives.

        Guard: if ANY buffered frame's mean elbow angle falls outside
        [CURL_CALIB_ELBOW_MIN, CURL_CALIB_ELBOW_MAX] the user was not holding
        arms straight down, so the calibration is rejected.  The SessionController
        owns the buffer and cannot be signalled to refill it, so a rejected
        calibration is reported via `valid=False` and the exercise falls back to
        the observed means — the on-screen info bar shows the rejection so the
        user can restart the session deliberately.

        DEFERRED: true reject-and-RESTART (discard the buffer and keep sampling)
        needs a sentinel return honoured by SessionController.process_frame,
        which finalises calibration unconditionally once the buffer fills; that
        is a session.py change and is out of scope here.
        """
        ua_l, ua_r, fa_l, fa_r, sw = [], [], [], [], []
        trunk, elb_l, elb_r, dr_l, dr_r = [], [], [], [], []
        rejected = 0

        def sample(lm, w, h):
            pts = {i: get_px(lm, i, w, h) for i in
                   (IDX_SHOULDER_L, IDX_SHOULDER_R, IDX_ELBOW_L, IDX_ELBOW_R,
                    IDX_WRIST_L, IDX_WRIST_R, IDX_HIP_L, IDX_HIP_R)}
            return (pts[IDX_SHOULDER_L], pts[IDX_SHOULDER_R],
                    pts[IDX_ELBOW_L], pts[IDX_ELBOW_R],
                    pts[IDX_WRIST_L], pts[IDX_WRIST_R],
                    pts[IDX_HIP_L], pts[IDX_HIP_R])

        def record(sh_l, sh_r, el_l, el_r, wr_l, wr_r, hi_l, hi_r):
            width = dist2d(sh_l, sh_r)
            d_l, d_r = _oriented_drift(sh_l, sh_r, el_l, el_r, width)
            ua_l.append(dist2d(sh_l, el_l))
            ua_r.append(dist2d(sh_r, el_r))
            fa_l.append(dist2d(el_l, wr_l))
            fa_r.append(dist2d(el_r, wr_r))
            sw.append(width)
            trunk.append(trunk_lean_angle(midpoint(sh_l, sh_r), midpoint(hi_l, hi_r)))
            elb_l.append(angle_at_joint(sh_l, el_l, wr_l))
            elb_r.append(angle_at_joint(sh_r, el_r, wr_r))
            dr_l.append(d_l)
            dr_r.append(d_r)

        for lm, w, h in landmarks_buffer:
            p = sample(lm, w, h)
            a_l = angle_at_joint(p[0], p[2], p[4])
            a_r = angle_at_joint(p[1], p[3], p[5])
            if not (CURL_CALIB_ELBOW_MIN <= (a_l + a_r) / 2.0 <= CURL_CALIB_ELBOW_MAX):
                rejected += 1
                continue
            record(*p)

        valid = len(ua_l) > 0 and rejected == 0

        if not ua_l:
            # Every frame failed the pose guard — fall back to the raw geometry
            # of the final frame so downstream maths still has finite denominators.
            lm, w, h = landmarks_buffer[-1]
            record(*sample(lm, w, h))

        mean = lambda xs: sum(xs) / len(xs)
        return Calibration({
            "upper_arm_length_l": mean(ua_l),
            "upper_arm_length_r": mean(ua_r),
            "forearm_length_l":   mean(fa_l),
            "forearm_length_r":   mean(fa_r),
            "shoulder_width":     mean(sw),
            "baseline_trunk_lean": mean(trunk),
            "baseline_elbow_angle_l": mean(elb_l),
            "baseline_elbow_angle_r": mean(elb_r),
            "baseline_elbow_drift_l": mean(dr_l),
            "baseline_elbow_drift_r": mean(dr_r),
            "valid":              valid,
            "rejected_frames":    rejected,
        })

    # ── Per-frame features ───────────────────────────────────────────────────
    def compute_features(self, lm, calibration, w, h) -> dict:
        sh_l = get_px(lm, IDX_SHOULDER_L, w, h)
        sh_r = get_px(lm, IDX_SHOULDER_R, w, h)
        el_l = get_px(lm, IDX_ELBOW_L, w, h)
        el_r = get_px(lm, IDX_ELBOW_R, w, h)
        wr_l = get_px(lm, IDX_WRIST_L, w, h)
        wr_r = get_px(lm, IDX_WRIST_R, w, h)
        hi_l = get_px(lm, IDX_HIP_L, w, h)
        hi_r = get_px(lm, IDX_HIP_R, w, h)

        shoulder_width = dist2d(sh_l, sh_r)
        elbow_l = angle_at_joint(sh_l, el_l, wr_l)
        elbow_r = angle_at_joint(sh_r, el_r, wr_r)

        # Elbow drift, normalised by SHOULDER width (not hip width as the squat
        # metrics use): the shoulder girdle is the body scale the arm hangs from.
        raw_l, raw_r = _oriented_drift(sh_l, sh_r, el_l, el_r, shoulder_width)

        # BASELINE SUBTRACTION — applied AFTER the outward-orientation signing,
        # so the baseline is itself expressed in the outward direction and the
        # sign convention survives.  The metric now measures travel from THIS
        # user's own resting elbow position rather than from the shoulder, which
        # is what makes 0.0 mean "no fault".  Before calibration completes the
        # baseline is the current frame, so drift reads 0 during the calibration
        # window — exactly how trunk_deviation behaves.
        if calibration:
            base_l = calibration.values["baseline_elbow_drift_l"]
            base_r = calibration.values["baseline_elbow_drift_r"]
        else:
            base_l, base_r = raw_l, raw_r

        lean = trunk_lean_angle(midpoint(sh_l, sh_r), midpoint(hi_l, hi_r))
        base_lean = calibration.values["baseline_trunk_lean"] if calibration else lean

        return {
            "shoulder_width_px": shoulder_width,
            "upper_arm_l_px":    dist2d(sh_l, el_l),
            "upper_arm_r_px":    dist2d(sh_r, el_r),
            "forearm_l_px":      dist2d(el_l, wr_l),
            "forearm_r_px":      dist2d(el_r, wr_r),
            "elbow_angle_l_deg": elbow_l,
            "elbow_angle_r_deg": elbow_r,
            "elbow_mean_deg":    (elbow_l + elbow_r) / 2.0,
            # Diagnostic only — no longer drives the asymmetry zone.
            "elbow_symmetry_diff_deg": abs(elbow_l - elbow_r),
            "elbow_drift_l":     raw_l - base_l,
            "elbow_drift_r":     raw_r - base_r,
            "baseline_elbow_drift_l": base_l,
            "baseline_elbow_drift_r": base_r,
            "trunk_lean_deg":    lean,
            "baseline_trunk_lean_deg": base_lean,
            "trunk_deviation_deg": abs(lean - base_lean),
        }

    # ── Zones ────────────────────────────────────────────────────────────────
    def classify_zones(self, features: dict) -> dict:
        s = self._smoother
        s["drift_l"].append(features["elbow_drift_l"])
        s["drift_r"].append(features["elbow_drift_r"])
        s["trunk_dev"].append(features["trunk_deviation_deg"])
        s["elbow_l"].append(features["elbow_angle_l_deg"])
        s["elbow_r"].append(features["elbow_angle_r_deg"])

        mean = lambda dq: sum(dq) / len(dq)

        zd_l = self._classify_drift(mean(s["drift_l"]))
        zd_r = self._classify_drift(mean(s["drift_r"]))
        zsw = _band(mean(s["trunk_dev"]), BODY_SWING_YELLOW, BODY_SWING_RED)
        zas = self._asymmetry_zone()

        # overall_zone deliberately covers ONLY the four live channels.
        # rom_incomplete is NOT included: it is not a per-frame condition at all
        # but a verdict about a whole repetition, only knowable once the rep has
        # finished.  Folding it into a per-frame overall zone would either colour
        # frames retroactively or hold a stale red long after the rep ended.  It
        # instead degrades that rep's rep_quality (green->yellow, yellow->red) in
        # _ArmRepMachine._finalise, which is where a per-rep judgement belongs.
        zo = _overall([zd_l, zd_r, zsw, zas])

        # Synthetic ROM zones — non-overall, each reaching the audio layer by a
        # different route because the two ROM faults fail differently.
        #
        # rom_height is EDGE-triggered: "you did not curl high enough" is only
        # knowable once a rep has completed, so _finalise arms it and it is held
        # red for CURL_ROM_CUE_HOLD_FRAMES (15) to outlast the controller's
        # CORRECTIVE_DWELL_FRAMES (12).
        #
        # rom_extension is LEVEL-triggered off a stall, because the user it
        # describes never completes a rep at all: someone who stops short of
        # straightening hovers in LOWERING indefinitely, so an edge-triggered cue
        # keyed on rep completion could never fire and the failure mode would be
        # silence.  The zone stays red for as long as the stall lasts, which
        # satisfies the controller's dwell naturally.
        #
        # Both are ORed across the two arms — either arm stalling, or either arm
        # finishing short, is a real fault worth cueing.  consume_rom_cue() is
        # called on BOTH arms unconditionally (not short-circuited) so each arm's
        # hold counter ticks down independently.
        rom_ext = "red" if (self._left.is_stalled() or self._right.is_stalled()) else "green"
        hold_l = self._left.consume_rom_cue()
        hold_r = self._right.consume_rom_cue()
        rom_hgt = "red" if (hold_l or hold_r) else "green"

        return {
            "elbow_drift_l": zd_l,
            "elbow_drift_r": zd_r,
            "body_swing":    zsw,
            "asymmetry":     zas,
            "overall":       zo,
            "rom_extension": rom_ext,
            "rom_height":    rom_hgt,
            "phase":         self.phase,
        }

    def _asymmetry_zone(self):
        """
        Per-rep left/right comparison — "one arm is doing less work".

        Compares each arm's MOST RECENT COMPLETED rep on peak flexion (how high
        it curled) and range of motion (how far it travelled).  This replaces the
        old instantaneous |elbow_l - elbow_r|, which was structurally useless for
        an alternating curl: there one arm is flexed precisely while the other is
        extended, so the difference is permanently huge BY DESIGN and the channel
        sat red on good and bad clips alike.

        Green until BOTH arms have completed a rep — there is nothing to compare
        before that, and a user part-way through their first left rep has not yet
        demonstrated any asymmetry.
        """
        left, right = self._left.last_rep_summary, self._right.last_rep_summary
        if left is None or right is None:
            self._arm_peak_diff = self._arm_rom_diff = None
            return "green"
        self._arm_peak_diff = abs(left["min_elbow"] - right["min_elbow"])
        self._arm_rom_diff = abs(left["rep_range"] - right["rep_range"])
        return _worse(
            _band(self._arm_peak_diff, CURL_ARM_PEAK_DIFF_YELLOW, CURL_ARM_PEAK_DIFF_RED),
            _band(self._arm_rom_diff, CURL_ARM_ROM_DIFF_YELLOW, CURL_ARM_ROM_DIFF_RED))

    @staticmethod
    def _classify_drift(value):
        """
        Drift is signed, outward-oriented and baseline-subtracted, so positive
        means the elbow has travelled outward FROM ITS OWN RESTING POSITION on
        either arm — the fault direction.  Negative means it is tucked further in
        than at rest, which is not an elbow-drift fault, so the same
        positive-only rule serves both arms.
        """
        if value > ELBOW_DRIFT_RED:
            return "red"
        if value > ELBOW_DRIFT_YELLOW:
            return "yellow"
        return "green"

    # ── Rep state machines ───────────────────────────────────────────────────
    def update_rep_counter(self, features, zones, frame_id, lm) -> RepState:
        """
        Advance BOTH per-arm machines by one frame.

        Same visibility-gate pattern as the squat counter: an unreliable frame is
        skipped entirely, advancing neither arm.
        """
        self._total_frames += 1
        if lm is not None and not landmarks_reliable(
                lm, _VIS_JOINTS, LANDMARK_MIN_VISIBILITY):
            self._skipped_frames += 1
            return self._state()

        s = self._smoother
        angle_l = sum(s["elbow_l"]) / len(s["elbow_l"])
        angle_r = sum(s["elbow_r"]) / len(s["elbow_r"])

        total = self.rep_count
        done_l = self._left.update(angle_l, features, zones, frame_id, total + 1)
        # If the left arm completed this frame it has already consumed the next
        # global rep number, so the right arm's would-be number shifts by one.
        done_r = self._right.update(angle_r, features, zones, frame_id,
                                    total + (2 if done_l else 1))
        for summary in (done_l, done_r):
            if summary is not None:
                self._rep_history.append(summary)
                self._last_rep_summary = summary

        return self._state()

    def _state(self):
        total = self.rep_count
        new = total > self._prev_rep_count
        self._prev_rep_count = total
        return RepState(phase=self.phase, rep_count=total,
                        last_rep_summary=self._last_rep_summary,
                        new_rep_completed=new)

    # ── Live accessors ───────────────────────────────────────────────────────
    @property
    def phase(self) -> str:
        """
        The REPRESENTATIVE phase: whichever arm is more active.

        The audio controller gates every cue on a single phase string, so with
        two independent arms one must be chosen.  Reporting the more active arm
        preserves the original intent — corrective cues fire while the user is
        moving and stay quiet while both arms hang at rest — in both curl modes.
        The unabridged per-arm phases are on screen and in the CSV.
        """
        return max((self._left.phase, self._right.phase),
                   key=lambda p: _PHASE_RANK.get(p, 0))

    @property
    def phase_pair(self) -> str:
        """Compact both-arm phase label for the info bar, e.g. 'L:CURL  R:EXT'."""
        return (f"L:{_PHASE_SHORT.get(self._left.phase, self._left.phase)}  "
                f"R:{_PHASE_SHORT.get(self._right.phase, self._right.phase)}")

    @property
    def rep_count(self) -> int:
        """TOTAL across both arms — what a person counts aloud when alternating."""
        return self._left.rep_count + self._right.rep_count

    @property
    def rep_history(self) -> list:
        return self._rep_history

    # ── Audio ────────────────────────────────────────────────────────────────
    @property
    def audio_cue_mapping(self) -> AudioCueMapping:
        """
        Priority high -> low: drift, swing, asymmetry, then the two ROM cues.

        The three form channels are gated OUT of EXTENDED, using the
        representative (most active) phase: while BOTH arms hang at rest there is
        no fault the user can act on mid-rep, but as soon as either arm is
        working the channel is live.

        The two ROM channels are gated to ALL phases.  With independent arms the
        representative phase describes whichever arm is busier, which is not
        necessarily the arm the ROM fault belongs to — a left-arm stall can
        coexist with the right arm mid-CURLING, and a phase-restricted gate would
        silently suppress it.  The zone itself already encodes the full
        condition (a stall is happening / a rep finished short), so the phase
        gate would add nothing but a way to lose the cue.

        Everything still runs through the existing dwell + cooldown machinery, so
        audio_feedback.py needs no change.
        """
        moving = {"CURLING", "CONTRACTED", "LOWERING"}
        return AudioCueMapping(channels=[
            {"name": "elbow_drift", "zone_keys": ["elbow_drift_l", "elbow_drift_r"],
             "phases": moving, "clips": [AUDIO_CURL_ELBOWS_PINNED]},
            {"name": "body_swing", "zone_keys": ["body_swing"],
             "phases": moving, "clips": [AUDIO_CURL_STOP_SWINGING]},
            {"name": "asymmetry", "zone_keys": ["asymmetry"],
             "phases": moving, "clips": [AUDIO_CURL_MATCH_ARMS]},
            {"name": "rom_extension", "zone_keys": ["rom_extension"],
             "phases": _ALL_PHASES, "clips": [AUDIO_CURL_FULL_EXTENSION]},
            {"name": "rom_height", "zone_keys": ["rom_height"],
             "phases": _ALL_PHASES, "clips": [AUDIO_CURL_HIGHER]},
        ])

    # ── CSV ──────────────────────────────────────────────────────────────────
    # All three overridden: the curl's rep-summary dict has entirely different
    # keys from the squat's, so inheriting the defaults would both overwrite
    # squat_features.csv and raise KeyError on the first completed rep.
    def csv_path(self) -> str:
        return CURL_CSV

    def rep_summary_path(self) -> str:
        return CURL_REP_SUMMARY_CSV

    def rep_summary_header(self) -> list:
        return CURL_REP_SUMMARY_HEADER

    def csv_header(self) -> list:
        return CURL_CSV_HEADER

    def get_csv_row(self, frame_id, features, zones, rep_state, calib_done) -> list:
        f = features
        live = calib_done and rep_state is not None
        return [
            frame_id,
            round(f["shoulder_width_px"], 2),
            round(f["upper_arm_l_px"], 2), round(f["upper_arm_r_px"], 2),
            round(f["forearm_l_px"], 2),   round(f["forearm_r_px"], 2),
            round(f["elbow_angle_l_deg"], 3), round(f["elbow_angle_r_deg"], 3),
            round(f["elbow_mean_deg"], 3),
            round(f["elbow_symmetry_diff_deg"], 3),
            round(f["elbow_drift_l"], 4), round(f["elbow_drift_r"], 4),
            round(f["baseline_elbow_drift_l"], 4),
            round(f["baseline_elbow_drift_r"], 4),
            round(f["trunk_lean_deg"], 3),
            round(f["baseline_trunk_lean_deg"], 3),
            round(f["trunk_deviation_deg"], 3),
            zones["elbow_drift_l"], zones["elbow_drift_r"],
            zones["body_swing"], zones["asymmetry"], zones["overall"],
            self.phase if live else "EXTENDED",
            self._left.phase if live else "EXTENDED",
            self._right.phase if live else "EXTENDED",
            self.rep_count if live else 0,
            self._left.rep_count if live else 0,
            self._right.rep_count if live else 0,
        ]

    # ── Display ──────────────────────────────────────────────────────────────
    def get_display_spec(self, features, zones, rep_state, calib_progress,
                         source_label, frame_id, pose_detected, lm) -> DisplaySpec:
        from src.zones import get_zone_color      # generic colour map, not squat logic

        spec = DisplaySpec()
        n, total, done, restarts = calib_progress

        col1 = [
            InfoCell(f"Source : {source_label}", (200, 200, 200)),
            InfoCell(f"Frame  : {frame_id:05d}", (200, 200, 200)),
            InfoCell("Pose   : DETECTED" if pose_detected else "Pose   : not detected",
                     (0, 255, 120) if pose_detected else (80, 80, 255)),
        ]

        if not (pose_detected and features):
            spec.columns = [col1, [InfoCell("Waiting for pose...", (100, 100, 150))], []]
            # ASCII only — rendered via cv2.putText (see squat.py for the same note).
            note = (f" - restarted x{restarts}, keep still" if restarts else "")
            spec.overall = ({"text": f"Stand still - calibrating  {n} / {total}{note}",
                             "color": (255, 200, 0)} if not done else None)
            return spec

        f = features
        asym_col = get_zone_color(zones["asymmetry"]) if zones else (0, 230, 255)
        # The arm-gap row reports the PER-REP asymmetry measure: the peak-flexion
        # gap and the range-of-motion gap between each arm's most recent completed
        # rep.  It is unavailable until BOTH arms have finished a rep, so the
        # placeholder says WHY rather than showing a bare "--".
        # Wording is kept under the narrowest info-bar column (w//3 = 213 px on a
        # 640-wide webcam frame at scale 0.50) so it cannot bleed into column 3.
        if self._arm_peak_diff is None:
            gap_txt = "Arm gap: 1 rep per arm"
        else:
            gap_txt = (f"Arm gap: peak{self._arm_peak_diff:.0f}"
                       f"  range{self._arm_rom_diff:.0f}")
        # Full phase words, one row per arm — "L:EXT  R:EXT" was unreadable.
        col2 = [
            InfoCell(f"Elbow  L:{f['elbow_angle_l_deg']:.0f}"
                     f"  R:{f['elbow_angle_r_deg']:.0f}", (0, 230, 255)),
            InfoCell(f"Phase L: {self._left.phase}",
                     CURL_PHASE_COLORS.get(self._left.phase, (200, 200, 200))),
            InfoCell(f"Phase R: {self._right.phase}",
                     CURL_PHASE_COLORS.get(self._right.phase, (200, 200, 200))),
            InfoCell(gap_txt, asym_col),
        ]

        last = rep_state.last_rep_summary if rep_state else None
        if not done:
            # A restart resets the counter; say so or it reads as a fault.
            label = f"retry {restarts}" if restarts else "arms down"
            tail = InfoCell(f"Calib: {n}/{total}  {label}", (255, 200, 0))
        else:
            tail = InfoCell(f"Reps: {self.rep_count}  "
                            f"(L {self._left.rep_count} / R {self._right.rep_count})",
                            (0, 255, 120))
        col3 = [
            InfoCell(f"Drift L: {f['elbow_drift_l']:+.2f}",
                     get_zone_color(zones["elbow_drift_l"]) if zones else (200, 200, 200)),
            InfoCell(f"Drift R: {f['elbow_drift_r']:+.2f}",
                     get_zone_color(zones["elbow_drift_r"]) if zones else (200, 200, 200)),
            InfoCell(f"Swing  : {f['trunk_deviation_deg']:4.1f} deg",
                     get_zone_color(zones["body_swing"]) if zones else (200, 200, 200)),
            tail,
        ]
        spec.columns = [col1, col2, col3]

        if zones:
            spec.overall = {"text": f"Overall  :  {OVERALL_LABEL[zones['overall']]}",
                            "color": get_zone_color(zones["overall"])}
            spec.aura_regions = [
                (_LEFT_ARM,  get_zone_color(zones["elbow_drift_l"])),
                (_RIGHT_ARM, get_zone_color(zones["elbow_drift_r"])),
                (_TORSO,     get_zone_color(zones["body_swing"])),
                (_OTHER,     (0, 255, 120)),          # legs neutral for this exercise
            ]
            spec.joint_colors = {
                IDX_ELBOW_L: get_zone_color(zones["elbow_drift_l"]),
                IDX_ELBOW_R: get_zone_color(zones["elbow_drift_r"]),
            }
            spec.torso_color = get_zone_color(zones["body_swing"])

        if done and rep_state is not None:
            last_text = (f"Last {last['arm']}: {self._q_label(last['rep_quality'])}  "
                         f"range: {last['rep_range']:.0f}deg") if last else None
            # The badge carries a SINGLE phase word, matching the squat badge's
            # structure ("Phase: BOTTOM").  Cramming both arms in made it the
            # widest line on the panel and unreadable at a glance; the per-arm
            # breakdown lives in the info bar, one full word per row.
            common = {
                "phase": self.phase,
                "phase_color": CURL_PHASE_COLORS.get(self.phase, (200, 200, 200)),
                "rep_count": rep_state.rep_count,
                "last_text": last_text,
                "last_color": get_zone_color(last["rep_quality"]) if last else None,
            }
            spec.rep_row, spec.overlay = dict(common), dict(common)
        return spec

    @staticmethod
    def _q_label(quality):
        return {"green": "GOOD", "yellow": "WARN", "red": "RISK"}.get(quality, quality)
