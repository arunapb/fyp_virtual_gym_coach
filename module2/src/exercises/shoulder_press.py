"""
exercises/shoulder_press.py — ShoulderPressExercise (Level 1: live coaching only).

Third concrete Exercise, after SquatExercise and BicepCurlExercise.  Structure
mirrors exercises/bicep_curl.py deliberately — same accessor pattern, same
smoother handling, same CSV accessor overrides — so the two upper-body
exercises read as a pair.

MOVEMENT.  Standing dumbbell OVERHEAD PRESS, established by inspecting the
dataset rather than assumed: the elbow sweeps ~50-60 deg at the rack to
157-175 deg at lockout, and the wrists finish ~0.76 torso-lengths ABOVE the
nose.  A lateral or front raise would hold the elbow near-straight throughout
and top out at shoulder height (wrist at or below the nose), so those are ruled
out.  Both arms move together on 10:1 to 14:1 of frames, so ONE mean-driven
state machine is used.  The bicep curl needed two independent per-arm machines
because that movement is alternating; this one does not.

NOTE ON THE ANGLE DIRECTION.  The driving angle INCREASES from the start pose
(racked ~55 deg -> locked ~165 deg).  That is the opposite of both the curl's
elbow angle and the squat's knee angle, so every velocity sign and every
threshold comparison in the state machine runs the other way round from those
two.  Read the machine with that in mind.

SCOPE NOTE — every threshold consumed here is a feel-tuned initial value from
the "SHOULDER PRESS THRESHOLDS" block in config.py.  NONE is dataset-derived,
and no fault-detection evaluation has been run.  Critically, none CAN be run
against the dataset's good/bad folder labels: across 5 good and 5 bad front
clips no metric separated them, lockout_p95 was higher on bad, and each bad
clip failed differently.  Validation for this exercise is synthetic-driver only.

Design constraints honoured by this module:
  * measurements.py / zones.py / rep_counter.py are squat-specific and are NOT
    imported — the press's geometry, zone classifiers and state machine live
    here.  Only the generic helpers in pose_utils.py are reused.
  * Nothing outside this file, the config block, the session registry and the
    keyboard map is modified; the class satisfies the existing Exercise
    interface, so session / drawing / audio consume it unchanged.
"""

from collections import deque

from config import (
    SMOOTH_N, OVERALL_LABEL,
    PRESS_CSV_HEADER, PRESS_PHASE_COLORS, POSE_CONNECTIONS,
    PRESS_CSV, PRESS_REP_SUMMARY_CSV, PRESS_REP_SUMMARY_HEADER,
    LANDMARK_MIN_VISIBILITY,
    IDX_SHOULDER_L, IDX_SHOULDER_R, IDX_ELBOW_L, IDX_ELBOW_R,
    IDX_WRIST_L, IDX_WRIST_R, IDX_HIP_L, IDX_HIP_R,
    PRESS_ELBOW_FLARE_YELLOW, PRESS_ELBOW_FLARE_RED,
    PRESS_BODY_SWING_YELLOW, PRESS_BODY_SWING_RED,
    PRESS_ASYMMETRY_YELLOW, PRESS_ASYMMETRY_RED,
    PRESS_LOCKOUT_MIN_ANGLE, PRESS_LOCKOUT_MIN_HEIGHT,
    PRESS_RACKED, PRESS_PRESSING, PRESS_LOCKED, PRESS_LOWERING,
    PRESS_LOCKED_MIN_DWELL_FRAMES, PRESS_LOWERING_STALL_FRAMES,
    PRESS_LOCKOUT_CUE_HOLD_FRAMES,
    PRESS_CALIB_ELBOW_MIN, PRESS_CALIB_ELBOW_MAX,
    AUDIO_PRESS_ELBOWS_IN, AUDIO_PRESS_DONT_ARCH, AUDIO_PRESS_MATCH_ARMS,
    AUDIO_PRESS_LOCK_OUT, AUDIO_PRESS_ALL_THE_WAY,
)
from src.pose_utils import (
    get_px, midpoint, dist2d, safe_div, angle_at_joint, trunk_lean_angle,
    landmarks_reliable,
)

from src.exercises.base import (
    Exercise, Calibration, RepState, DisplaySpec, InfoCell, AudioCueMapping,
)


# Aura body-region groupings.  Arms are the story for this exercise, so each
# arm chain is coloured by its own flare zone and the legs stay neutral.
_LEFT_ARM  = [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21)]
_RIGHT_ARM = [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22)]
_TORSO     = [(11, 12), (11, 23), (12, 24), (23, 24)]
_SPECIAL   = set(_LEFT_ARM) | set(_RIGHT_ARM) | set(_TORSO)
_OTHER     = [(a, b) for a, b in POSE_CONNECTIONS if (a, b) not in _SPECIAL]

# Visibility gate: shoulders, elbows and wrists ONLY — six landmarks.
# Deliberately NO hips and NO knees.  Knee visibility was measured at 0.10-0.16
# on three of four profiled clips because the framing crops the lower legs, so a
# knee-inclusive gate would reject nearly every frame of a perfectly usable
# front-view press.
_VIS_JOINTS = [IDX_SHOULDER_L, IDX_SHOULDER_R, IDX_ELBOW_L, IDX_ELBOW_R,
               IDX_WRIST_L, IDX_WRIST_R]

IDX_NOSE = 0

_VELOCITY_N = 5      # frames over which elbow-angle velocity is estimated
_VEL_DEAD_ZONE = 0.5  # deg/frame; below this magnitude motion counts as stopped

_ALL_PHASES = {"RACKED", "PRESSING", "LOCKED", "LOWERING"}


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


def _oriented_flare(sh_l, sh_r, el_l, el_r, shoulder_width):
    """
    Raw outward-oriented elbow flare for both arms, normalised by shoulder width.

    ORIENTATION — the bare (elbow_x - shoulder_x) difference is MIRROR-SIGNED
    between the two arms: an elbow flaring away from the body INCREASES x on one
    side and DECREASES it on the other.  Classifying that with a single
    positive-is-bad rule leaves one arm's channel structurally dead — the bicep
    curl shipped exactly that bug, and its right-arm drift channel read 100%
    green across every clip until it was fixed.  Each side is therefore signed by
    which side of the shoulder midline its own shoulder sits on, so positive
    ALWAYS means "flared outward" for both arms.  Deriving the sign from the
    observed shoulders rather than hard-coding it also keeps the metric correct
    whether or not the frame was mirrored (live_runner flips webcam, not video).

    This is the RAW value.  The per-user resting flare is subtracted separately
    (see compute_features) — shared here so calibrate() and compute_features()
    cannot drift apart in how they orient the sign.
    """
    sh_mid_x = (sh_l[0] + sh_r[0]) / 2.0
    side_l = 1.0 if sh_l[0] >= sh_mid_x else -1.0
    side_r = 1.0 if sh_r[0] >= sh_mid_x else -1.0
    return (safe_div((el_l[0] - sh_l[0]) * side_l, shoulder_width),
            safe_div((el_r[0] - sh_r[0]) * side_r, shoulder_width))


class ShoulderPressExercise(Exercise):
    name = "ShoulderPress"
    calibration_pose_description = (
        "Hold the weights at shoulder height. Hold still."
    )

    def __init__(self):
        # Rolling smoothers, fresh per activation — same 5-frame pattern the
        # squat and curl use, so zone colours do not flicker on landmark jitter.
        self._smoother = {k: deque(maxlen=SMOOTH_N) for k in (
            "flare_l", "flare_r", "trunk_dev", "asymmetry", "elbow_mean")}
        self._vel_buf = deque(maxlen=_VELOCITY_N)
        self._prev_elbow_mean = None

        # State machine (single, mean-driven — the press is simultaneous).
        self._phase = "RACKED"
        self._frames_in_phase = 0
        self._rep_count = 0
        self._rep_history = []
        self._last_rep_summary = None
        self._prev_rep_count = 0
        self._rep_start_frame = 0
        self._reached_locked = False

        # Per-rep accumulators
        self._reset_rep_stats()

        # Synthetic lockout zone, held red for a window after a rep that failed
        # to lock out, so the dwell-based audio controller can observe it.  The
        # other failure (never returning to the rack) is detected as a LOWERING
        # stall instead — see classify_zones / audio_cue_mapping.
        self._lockout_cue_frames = 0

        # Visibility gate bookkeeping
        self._skipped_frames = 0
        self._total_frames = 0

    # ── Calibration ──────────────────────────────────────────────────────────
    def calibrate(self, landmarks_buffer):
        """
        Average the racked reference pose over the buffered frames.

        As well as the segment lengths and trunk lean, this locks
        baseline_elbow_flare_l/r — the subject's NATURAL racked elbow position.
        This is not optional.  The bicep curl shipped the equivalent lateral
        metric un-baselined and it read AT its RED threshold with the subject
        standing still, because elbows do not sit directly beneath the shoulder
        joint; the same would happen here, worse, because a racked press starts
        with the elbows already out to the sides.  baseline_trunk_lean receives
        the identical treatment and always has.

        Guard: if ANY buffered frame's mean elbow angle falls outside
        [PRESS_CALIB_ELBOW_MIN, PRESS_CALIB_ELBOW_MAX] the lifter was not
        holding the racked start pose, so the calibration is rejected via
        `valid=False`.  SessionController then DISCARDS the buffer and resamples,
        keeping the cleanest buffer seen if it never converges — so unlike the
        curl at the time it was written, rejection here is genuinely recoverable.
        """
        ua_l, ua_r, fa_l, fa_r, sw, tl = [], [], [], [], [], []
        trunk, elb_l, elb_r, fl_l, fl_r, wy_l, wy_r = [], [], [], [], [], [], []
        rejected = 0

        def sample(lm, w, h):
            pts = {i: get_px(lm, i, w, h) for i in
                   (IDX_SHOULDER_L, IDX_SHOULDER_R, IDX_ELBOW_L, IDX_ELBOW_R,
                    IDX_WRIST_L, IDX_WRIST_R, IDX_HIP_L, IDX_HIP_R, IDX_NOSE)}
            return (pts[IDX_SHOULDER_L], pts[IDX_SHOULDER_R],
                    pts[IDX_ELBOW_L], pts[IDX_ELBOW_R],
                    pts[IDX_WRIST_L], pts[IDX_WRIST_R],
                    pts[IDX_HIP_L], pts[IDX_HIP_R], pts[IDX_NOSE])

        def record(sh_l, sh_r, el_l, el_r, wr_l, wr_r, hi_l, hi_r, nose):
            width = dist2d(sh_l, sh_r)
            sh_mid, hi_mid = midpoint(sh_l, sh_r), midpoint(hi_l, hi_r)
            torso = dist2d(sh_mid, hi_mid)
            f_l, f_r = _oriented_flare(sh_l, sh_r, el_l, el_r, width)
            ua_l.append(dist2d(sh_l, el_l))
            ua_r.append(dist2d(sh_r, el_r))
            fa_l.append(dist2d(el_l, wr_l))
            fa_r.append(dist2d(el_r, wr_r))
            sw.append(width)
            tl.append(torso)
            trunk.append(trunk_lean_angle(sh_mid, hi_mid))
            elb_l.append(angle_at_joint(sh_l, el_l, wr_l))
            elb_r.append(angle_at_joint(sh_r, el_r, wr_r))
            fl_l.append(f_l)
            fl_r.append(f_r)
            # Stored NORMALISED (torso-lengths above the nose), not as a raw
            # pixel y — a raw coordinate would not survive a change of camera
            # distance or frame size.
            wy_l.append(safe_div(nose[1] - wr_l[1], torso))
            wy_r.append(safe_div(nose[1] - wr_r[1], torso))

        for lm, w, h in landmarks_buffer:
            p = sample(lm, w, h)
            a_l = angle_at_joint(p[0], p[2], p[4])
            a_r = angle_at_joint(p[1], p[3], p[5])
            if not (PRESS_CALIB_ELBOW_MIN <= (a_l + a_r) / 2.0 <= PRESS_CALIB_ELBOW_MAX):
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
            "torso_length":       mean(tl),
            "baseline_trunk_lean": mean(trunk),
            "baseline_elbow_angle_l": mean(elb_l),
            "baseline_elbow_angle_r": mean(elb_r),
            "baseline_elbow_flare_l": mean(fl_l),
            "baseline_elbow_flare_r": mean(fl_r),
            "baseline_wrist_y_l": mean(wy_l),
            "baseline_wrist_y_r": mean(wy_r),
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
        nose = get_px(lm, IDX_NOSE, w, h)

        shoulder_width = dist2d(sh_l, sh_r)
        sh_mid, hi_mid = midpoint(sh_l, sh_r), midpoint(hi_l, hi_r)
        torso_length = dist2d(sh_mid, hi_mid)
        elbow_l = angle_at_joint(sh_l, el_l, wr_l)
        elbow_r = angle_at_joint(sh_r, el_r, wr_r)

        # Elbow flare, normalised by SHOULDER width — the shoulder girdle is the
        # body scale the arm hangs from, so the ratio survives changes of subject
        # size and camera distance.
        raw_l, raw_r = _oriented_flare(sh_l, sh_r, el_l, el_r, shoulder_width)

        # BASELINE SUBTRACTION — applied AFTER the outward-orientation signing,
        # so the baseline is itself expressed in the outward direction and the
        # sign convention survives.  The metric therefore measures travel from
        # THIS lifter's own racked elbow position, which is what makes 0.0 mean
        # "no fault".  Before calibration completes the baseline is the current
        # frame, so flare reads 0 during the calibration window — exactly how
        # trunk_deviation behaves.
        if calibration:
            base_l = calibration.values["baseline_elbow_flare_l"]
            base_r = calibration.values["baseline_elbow_flare_r"]
        else:
            base_l, base_r = raw_l, raw_r

        lean = trunk_lean_angle(sh_mid, hi_mid)
        base_lean = calibration.values["baseline_trunk_lean"] if calibration else lean

        return {
            "shoulder_width_px": shoulder_width,
            "torso_length_px":   torso_length,
            "upper_arm_l_px":    dist2d(sh_l, el_l),
            "upper_arm_r_px":    dist2d(sh_r, el_r),
            "forearm_l_px":      dist2d(el_l, wr_l),
            "forearm_r_px":      dist2d(el_r, wr_r),
            "elbow_angle_l_deg": elbow_l,
            "elbow_angle_r_deg": elbow_r,
            "elbow_mean_deg":    (elbow_l + elbow_r) / 2.0,
            "elbow_symmetry_diff_deg": abs(elbow_l - elbow_r),
            "elbow_flare_l":     raw_l - base_l,
            "elbow_flare_r":     raw_r - base_r,
            "baseline_elbow_flare_l": base_l,
            "baseline_elbow_flare_r": base_r,
            # Positive = wrist ABOVE the nose, in torso-lengths.  Used to verify
            # the press actually went overhead rather than forward.
            "wrist_above_nose_l": safe_div(nose[1] - wr_l[1], torso_length),
            "wrist_above_nose_r": safe_div(nose[1] - wr_r[1], torso_length),
            "trunk_lean_deg":    lean,
            "baseline_trunk_lean_deg": base_lean,
            "trunk_deviation_deg": abs(lean - base_lean),
        }

    # ── Zones ────────────────────────────────────────────────────────────────
    def classify_zones(self, features: dict) -> dict:
        s = self._smoother
        s["flare_l"].append(features["elbow_flare_l"])
        s["flare_r"].append(features["elbow_flare_r"])
        s["trunk_dev"].append(features["trunk_deviation_deg"])
        s["asymmetry"].append(features["elbow_symmetry_diff_deg"])
        s["elbow_mean"].append(features["elbow_mean_deg"])

        mean = lambda dq: sum(dq) / len(dq)

        zf_l = self._classify_flare(mean(s["flare_l"]))
        zf_r = self._classify_flare(mean(s["flare_r"]))
        zsw = _band(mean(s["trunk_dev"]),
                    PRESS_BODY_SWING_YELLOW, PRESS_BODY_SWING_RED)
        # A per-FRAME asymmetry comparison is valid here only because the press
        # is simultaneous; see the config note on why the curl could not use one.
        zas = _band(mean(s["asymmetry"]),
                    PRESS_ASYMMETRY_YELLOW, PRESS_ASYMMETRY_RED)

        # overall_zone deliberately covers ONLY the four live channels.
        # lockout_incomplete is NOT included, for exactly the reason
        # rom_incomplete is excluded from the curl's overall zone: it is not a
        # per-frame condition at all but a verdict about a whole repetition, only
        # knowable once the rep has finished.  Folding it in would either colour
        # frames retroactively or hold a stale red long after the rep ended.  It
        # instead degrades that rep's rep_quality (green->yellow, yellow->red) in
        # _finalise_rep, which is where a per-rep judgement belongs.
        zo = _overall([zf_l, zf_r, zsw, zas])

        # Synthetic lockout zones — non-overall, each reaching the audio layer by
        # a different route because the two failures fail differently.
        #
        # lockout_short is EDGE-triggered: "you did not lock out" is only
        # knowable once a rep has completed, so _finalise_rep arms it and it is
        # held red for PRESS_LOCKOUT_CUE_HOLD_FRAMES (15) to outlast the
        # controller's CORRECTIVE_DWELL_FRAMES (12).
        #
        # lockout_stall is LEVEL-triggered off a stall, because the lifter it
        # describes never completes a rep at all: someone who parks mid-descent
        # sits in LOWERING indefinitely, so an edge-triggered cue keyed on rep
        # completion could never fire and the failure mode would be silence.
        stall = "red" if (self._phase == "LOWERING"
                          and self._frames_in_phase >= PRESS_LOWERING_STALL_FRAMES) else "green"
        short = "green"
        if self._lockout_cue_frames > 0:
            self._lockout_cue_frames -= 1
            short = "red"

        return {
            "elbow_flare_l": zf_l,
            "elbow_flare_r": zf_r,
            "body_swing":    zsw,
            "asymmetry":     zas,
            "overall":       zo,
            "lockout_short": short,
            "lockout_stall": stall,
            "phase":         self._phase,
        }

    @staticmethod
    def _classify_flare(value):
        """
        Flare is signed, outward-oriented and baseline-subtracted, so positive
        means the elbow has travelled outward FROM ITS OWN RACKED POSITION on
        either arm — the fault direction.  Negative means it is tucked further in
        than at the rack, which is not a flare fault, so the same positive-only
        rule serves both arms.
        """
        if value > PRESS_ELBOW_FLARE_RED:
            return "red"
        if value > PRESS_ELBOW_FLARE_YELLOW:
            return "yellow"
        return "green"

    # ── Rep state machine ────────────────────────────────────────────────────
    def update_rep_counter(self, features, zones, frame_id, lm) -> RepState:
        """
        RACKED -> PRESSING -> LOCKED -> LOWERING -> RACKED (rep completes).

        Driven by the smoothed MEAN elbow angle, which INCREASES as the weight
        rises — the reverse of the curl.  Same visibility-gate pattern as the
        other two exercises: an unreliable frame is skipped entirely without
        advancing any state.
        """
        self._total_frames += 1
        if lm is not None and not landmarks_reliable(
                lm, _VIS_JOINTS, LANDMARK_MIN_VISIBILITY):
            self._skipped_frames += 1
            return self._state()

        angle = sum(self._smoother["elbow_mean"]) / len(self._smoother["elbow_mean"])
        vel = self._velocity(angle)
        self._frames_in_phase += 1

        if self._phase != "RACKED":
            self._accumulate(features, zones)

        if self._phase == "RACKED":
            # Commit to a press only on a genuine upward sweep, so a lifter
            # fidgeting at the rack cannot start a phantom rep.
            if angle > PRESS_PRESSING and vel > _VEL_DEAD_ZONE:
                self._rep_start_frame = frame_id
                self._reset_rep_stats()
                self._reached_locked = False
                self._accumulate(features, zones)
                self._enter("PRESSING")

        elif self._phase == "PRESSING":
            if angle <= PRESS_RACKED:
                self._enter("RACKED")        # aborted — came back down unlocked
            elif angle >= PRESS_LOCKED:
                self._reached_locked = True
                self._enter("LOCKED")

        elif self._phase == "LOCKED":
            # Dwell gate REJECTS bounced reps rather than merely delaying them.
            # The test is "did the weight LEAVE the locked region before the
            # minimum hold was satisfied", i.e. angle back below PRESS_LOWERING —
            # NOT "did it return all the way to PRESS_RACKED".  That second form
            # was the curl's original bug: the driving angle is smoothed and lags
            # the raw signal by about the smoothing window, so a full return to
            # the start pose can essentially never register inside the dwell, and
            # keying the gate on it let every bounce through.
            if angle < PRESS_LOWERING and vel < -_VEL_DEAD_ZONE:
                if self._frames_in_phase < PRESS_LOCKED_MIN_DWELL_FRAMES:
                    self._reached_locked = False
                self._enter("LOWERING")
            elif angle <= PRESS_RACKED:
                # Dropped straight past the lowering band in one window.
                self._reached_locked = False
                self._enter("RACKED")

        elif self._phase == "LOWERING":
            if angle <= PRESS_RACKED:
                if self._reached_locked:
                    self._rep_count += 1
                    self._finalise_rep(frame_id)
                self._enter("RACKED")

        return self._state()

    def _velocity(self, angle):
        if self._prev_elbow_mean is not None:
            self._vel_buf.append(angle - self._prev_elbow_mean)
        self._prev_elbow_mean = angle
        return sum(self._vel_buf) / len(self._vel_buf) if self._vel_buf else 0.0

    def _enter(self, phase):
        self._phase = phase
        self._frames_in_phase = 0

    def _state(self):
        new = self._rep_count > self._prev_rep_count
        self._prev_rep_count = self._rep_count
        return RepState(phase=self._phase, rep_count=self._rep_count,
                        last_rep_summary=self._last_rep_summary,
                        new_rep_completed=new)

    # ── Per-rep statistics ───────────────────────────────────────────────────
    def _reset_rep_stats(self):
        self._min_l = self._min_r = 180.0
        self._max_l = self._max_r = 0.0
        self._peak_flare_l = self._peak_flare_r = 0.0
        self._peak_trunk = 0.0
        self._max_asym = 0.0
        self._peak_wrist_l = self._peak_wrist_r = -99.0
        self._worst = {"elbow_flare_l": "green", "elbow_flare_r": "green",
                       "body_swing": "green", "asymmetry": "green"}

    def _accumulate(self, f, z):
        self._min_l = min(self._min_l, f["elbow_angle_l_deg"])
        self._max_l = max(self._max_l, f["elbow_angle_l_deg"])
        self._min_r = min(self._min_r, f["elbow_angle_r_deg"])
        self._max_r = max(self._max_r, f["elbow_angle_r_deg"])
        self._peak_flare_l = max(self._peak_flare_l, f["elbow_flare_l"])
        self._peak_flare_r = max(self._peak_flare_r, f["elbow_flare_r"])
        self._peak_trunk = max(self._peak_trunk, f["trunk_deviation_deg"])
        self._max_asym = max(self._max_asym, f["elbow_symmetry_diff_deg"])
        self._peak_wrist_l = max(self._peak_wrist_l, f["wrist_above_nose_l"])
        self._peak_wrist_r = max(self._peak_wrist_r, f["wrist_above_nose_r"])
        if z:
            for k in self._worst:
                self._worst[k] = _worse(self._worst[k], z.get(k, "green"))

    def _finalise_rep(self, end_frame):
        # Lockout needs BOTH ends of the evidence: elbows extended AND wrists
        # actually overhead.  Angle alone can be satisfied by pressing the
        # weights forward rather than up; height alone can be satisfied with
        # bent arms.  The weaker arm and the lower hand decide, so a rep is only
        # credited when both sides genuinely finished.
        angle_ok = min(self._max_l, self._max_r) >= PRESS_LOCKOUT_MIN_ANGLE
        height_ok = (min(self._peak_wrist_l, self._peak_wrist_r)
                     >= PRESS_LOCKOUT_MIN_HEIGHT)
        lockout_achieved = angle_ok and height_ok

        quality = _overall(list(self._worst.values()))
        if not lockout_achieved:
            # A short rep degrades quality one step: a technically clean but
            # partial press should not be reported as good form.  Same
            # green->yellow->red degrade the curl applies for a short ROM.
            quality = {"green": "yellow", "yellow": "red", "red": "red"}[quality]
            self._lockout_cue_frames = PRESS_LOCKOUT_CUE_HOLD_FRAMES

        summary = {
            "rep_number":      self._rep_count,
            "start_frame":     self._rep_start_frame,
            "end_frame":       end_frame,
            "duration_frames": end_frame - self._rep_start_frame,
            "min_elbow_l":     round(self._min_l, 1),
            "max_elbow_l":     round(self._max_l, 1),
            "min_elbow_r":     round(self._min_r, 1),
            "max_elbow_r":     round(self._max_r, 1),
            "peak_flare_l":    round(self._peak_flare_l, 3),
            "peak_flare_r":    round(self._peak_flare_r, 3),
            "peak_trunk_deviation": round(self._peak_trunk, 2),
            "max_asymmetry":   round(self._max_asym, 2),
            "peak_wrist_above_nose_l": round(self._peak_wrist_l, 3),
            "peak_wrist_above_nose_r": round(self._peak_wrist_r, 3),
            "lockout_achieved": int(lockout_achieved),
            "rep_quality":     quality,
        }
        self._rep_history.append(summary)
        self._last_rep_summary = summary

    # ── Live accessors ───────────────────────────────────────────────────────
    @property
    def phase(self) -> str:
        return self._phase

    @property
    def rep_count(self) -> int:
        return self._rep_count

    @property
    def rep_history(self) -> list:
        return self._rep_history

    # ── Audio ────────────────────────────────────────────────────────────────
    @property
    def audio_cue_mapping(self) -> AudioCueMapping:
        """
        Priority high -> low: flare, swing, asymmetry, then the two lockout cues.

        The three form channels are gated OUT of RACKED: with the weights resting
        at the shoulders there is no fault the lifter can act on mid-rep.

        The two lockout channels are gated to ALL phases.  Their zones already
        encode the full condition — a stall is happening, or a rep finished short
        — so a phase gate would add nothing but a way to lose the cue if the
        lifter starts the next press before the dwell completes.  This matches
        where the curl's equivalent channels ended up.

        Everything runs through the existing dwell + cooldown machinery, so
        audio_feedback.py needs no change.
        """
        moving = {"PRESSING", "LOCKED", "LOWERING"}
        return AudioCueMapping(channels=[
            {"name": "elbow_flare", "zone_keys": ["elbow_flare_l", "elbow_flare_r"],
             "phases": moving, "clips": [AUDIO_PRESS_ELBOWS_IN]},
            {"name": "body_swing", "zone_keys": ["body_swing"],
             "phases": moving, "clips": [AUDIO_PRESS_DONT_ARCH]},
            {"name": "asymmetry", "zone_keys": ["asymmetry"],
             "phases": moving, "clips": [AUDIO_PRESS_MATCH_ARMS]},
            {"name": "lockout_short", "zone_keys": ["lockout_short"],
             "phases": _ALL_PHASES, "clips": [AUDIO_PRESS_LOCK_OUT]},
            {"name": "lockout_stall", "zone_keys": ["lockout_stall"],
             "phases": _ALL_PHASES, "clips": [AUDIO_PRESS_ALL_THE_WAY]},
        ])

    # ── CSV ──────────────────────────────────────────────────────────────────
    # All three overridden: this exercise's rep-summary dict has different keys
    # from both the squat's and the curl's, so inheriting the base defaults would
    # overwrite squat_features.csv and raise KeyError on the first completed rep.
    def csv_path(self) -> str:
        return PRESS_CSV

    def rep_summary_path(self) -> str:
        return PRESS_REP_SUMMARY_CSV

    def rep_summary_header(self) -> list:
        return PRESS_REP_SUMMARY_HEADER

    def csv_header(self) -> list:
        return PRESS_CSV_HEADER

    def get_csv_row(self, frame_id, features, zones, rep_state, calib_done) -> list:
        f = features
        live = calib_done and rep_state is not None
        return [
            frame_id,
            round(f["shoulder_width_px"], 2), round(f["torso_length_px"], 2),
            round(f["upper_arm_l_px"], 2), round(f["upper_arm_r_px"], 2),
            round(f["forearm_l_px"], 2),   round(f["forearm_r_px"], 2),
            round(f["elbow_angle_l_deg"], 3), round(f["elbow_angle_r_deg"], 3),
            round(f["elbow_mean_deg"], 3),
            round(f["elbow_symmetry_diff_deg"], 3),
            round(f["elbow_flare_l"], 4), round(f["elbow_flare_r"], 4),
            round(f["baseline_elbow_flare_l"], 4),
            round(f["baseline_elbow_flare_r"], 4),
            round(f["wrist_above_nose_l"], 4), round(f["wrist_above_nose_r"], 4),
            round(f["trunk_lean_deg"], 3),
            round(f["baseline_trunk_lean_deg"], 3),
            round(f["trunk_deviation_deg"], 3),
            zones["elbow_flare_l"], zones["elbow_flare_r"],
            zones["body_swing"], zones["asymmetry"], zones["overall"],
            self._phase if live else "RACKED",
            self._rep_count if live else 0,
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
            # ASCII only — rendered via cv2.putText (Hershey fonts have no glyphs
            # outside ASCII; an em-dash renders as "???").
            note = (f" - restarted x{restarts}, keep still" if restarts else "")
            spec.overall = ({"text": f"Hold at shoulders - calibrating  {n} / {total}{note}",
                             "color": (255, 200, 0)} if not done else None)
            return spec

        f = features
        # Wording kept under the narrowest info-bar column (w//3 = 213 px on a
        # 640-wide webcam frame at scale 0.50) so it cannot bleed into column 3.
        col2 = [
            InfoCell(f"Elbow  L:{f['elbow_angle_l_deg']:.0f}"
                     f"  R:{f['elbow_angle_r_deg']:.0f}", (0, 230, 255)),
            InfoCell(f"Sym gap: {f['elbow_symmetry_diff_deg']:.0f} deg",
                     get_zone_color(zones["asymmetry"]) if zones else (0, 230, 255)),
            InfoCell(f"Phase  : {self._phase}",
                     PRESS_PHASE_COLORS.get(self._phase, (200, 200, 200))),
        ]

        last = rep_state.last_rep_summary if rep_state else None
        if not done:
            label = f"retry {restarts}" if restarts else "at shoulders"
            tail = InfoCell(f"Calib: {n}/{total}  {label}", (255, 200, 0))
        elif last:
            ok = "LOCKED OUT" if last["lockout_achieved"] else "SHORT"
            tail = InfoCell(f"Reps: {self._rep_count}  last {ok}",
                            (0, 255, 120) if last["lockout_achieved"] else (80, 80, 255))
        else:
            tail = InfoCell(f"Reps: {self._rep_count}", (0, 255, 120))
        col3 = [
            InfoCell(f"Flare L: {f['elbow_flare_l']:+.2f}",
                     get_zone_color(zones["elbow_flare_l"]) if zones else (200, 200, 200)),
            InfoCell(f"Flare R: {f['elbow_flare_r']:+.2f}",
                     get_zone_color(zones["elbow_flare_r"]) if zones else (200, 200, 200)),
            InfoCell(f"Swing  : {f['trunk_deviation_deg']:4.1f} deg",
                     get_zone_color(zones["body_swing"]) if zones else (200, 200, 200)),
            tail,
        ]
        spec.columns = [col1, col2, col3]

        if zones:
            spec.overall = {"text": f"Overall  :  {OVERALL_LABEL[zones['overall']]}",
                            "color": get_zone_color(zones["overall"])}
            spec.aura_regions = [
                (_LEFT_ARM,  get_zone_color(zones["elbow_flare_l"])),
                (_RIGHT_ARM, get_zone_color(zones["elbow_flare_r"])),
                (_TORSO,     get_zone_color(zones["body_swing"])),
                (_OTHER,     (0, 255, 120)),          # legs neutral for this exercise
            ]
            spec.joint_colors = {
                IDX_ELBOW_L: get_zone_color(zones["elbow_flare_l"]),
                IDX_ELBOW_R: get_zone_color(zones["elbow_flare_r"]),
            }
            spec.torso_color = get_zone_color(zones["body_swing"])

        if done and rep_state is not None:
            # draw_overlay sizes the badge from the LONGEST line including this
            # one, so a verbose last_text no longer spills outside the panel.
            last_text = (f"Last: {self._q_label(last['rep_quality'])}  "
                         f"top: {max(last['max_elbow_l'], last['max_elbow_r']):.0f}deg"
                         ) if last else None
            common = {
                "phase": self._phase,
                "phase_color": PRESS_PHASE_COLORS.get(self._phase, (200, 200, 200)),
                "rep_count": rep_state.rep_count,
                "last_text": last_text,
                "last_color": get_zone_color(last["rep_quality"]) if last else None,
            }
            spec.rep_row, spec.overlay = dict(common), dict(common)
        return spec

    @staticmethod
    def _q_label(quality):
        return {"green": "GOOD", "yellow": "WARN", "red": "RISK"}.get(quality, quality)
