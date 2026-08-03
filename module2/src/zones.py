"""
zones.py — Form-risk zone classification and smoothing.

Each classify_* function maps a numeric metric to one of three zones:
  "green"  — acceptable technique
  "yellow" — technique warning
  "red"    — likely problematic technique

All thresholds are project-defined heuristics.
They are NOT medical or clinical injury thresholds.
Threshold values live in config.py so they can be tuned in one place.
"""

from config import (
    VALGUS_L_YELLOW, VALGUS_L_RED,
    VALGUS_R_YELLOW, VALGUS_R_RED,
    TRUNK_DEV_YELLOW, TRUNK_DEV_RED,
    DEPTH_YELLOW, DEPTH_RED,
    DEPTH_RATIO_YELLOW, DEPTH_RATIO_RED,
    DEPTH_KNEE_SHALLOW_YELLOW, DEPTH_KNEE_SHALLOW_RED,
    DEPTH_KNEE_DEEP_YELLOW, DEPTH_KNEE_DEEP_RED,
    DEPTH_PHASE_BOTTOM_THRESHOLD,
    VALGUS_3D_YELLOW, VALGUS_3D_RED,
    ASYMMETRY_YELLOW, ASYMMETRY_RED,
    SMOOTH_N,
)


# =============================================================================
# INDIVIDUAL CLASSIFIERS
# =============================================================================

def classify_valgus_left(value):
    """
    Classify left-leg valgus risk from the normalised knee offset.
    Positive value → left knee drifting right of the left ankle → valgus collapse.
    """
    if value > VALGUS_L_RED:
        return "red"
    if value > VALGUS_L_YELLOW:
        return "yellow"
    return "green"


def classify_valgus_right(value):
    """
    Classify right-leg valgus risk from the normalised knee offset.
    Negative value → right knee drifting left of the right ankle → valgus collapse.
    """
    if value < VALGUS_R_RED:
        return "red"
    if value < VALGUS_R_YELLOW:
        return "yellow"
    return "green"


def classify_trunk_deviation(value):
    """
    Classify trunk lean based on absolute deviation from the standing baseline.
    Both forward and lateral lean are captured by using abs(deviation).
    """
    abs_dev = abs(value)
    if abs_dev > TRUNK_DEV_RED:
        return "red"
    if abs_dev > TRUNK_DEV_YELLOW:
        return "yellow"
    return "green"


def classify_depth(value, baseline=None):
    """
    Classify squat depth.  Higher value = hips still elevated = shallower squat.
    This is a target-depth heuristic, NOT an injury threshold.
    Only meaningful at BOTTOM phase — call classify_depth_phase_aware instead.

    With a `baseline` (the user's own calibrated standing norm_hip_depth) the
    metric is the RATIO to that baseline — 1.0 standing, lower is deeper — and
    the DEPTH_RATIO_* thresholds apply.  Without one it falls back to the
    absolute DEPTH_* thresholds, which is what every caller predating the
    baseline did.

    The absolute path is retained for compatibility, not because it works: its
    thresholds come from a dataset whose subjects stand at ~1.92, and a subject
    standing at 1.70 cannot reach them at any depth.  See config.py.
    """
    if baseline:
        ratio = value / baseline
        if ratio > DEPTH_RATIO_RED:
            return "red"
        if ratio > DEPTH_RATIO_YELLOW:
            return "yellow"
        return "green"
    if value > DEPTH_RED:
        return "red"
    if value > DEPTH_YELLOW:
        return "yellow"
    return "green"


def squat_depth_zones(knee_deg):
    """
    Two-sided squat-depth verdict from a knee angle, as
    (too_shallow_zone, too_deep_zone).

    Both sides are reported separately rather than reduced to one zone because
    each drives its OWN audio cue: "go deeper" and "not so deep" are opposite
    corrections and a single channel can only speak one clip.  A given angle can
    only trip one side, so at most one of the two is ever non-green.

    Angle is the DEEPEST point of a rep for the per-rep verdict, or the smoothed
    current angle for the live BOTTOM-phase indicator.
    """
    shallow = ("red"    if knee_deg > DEPTH_KNEE_SHALLOW_RED else
               "yellow" if knee_deg > DEPTH_KNEE_SHALLOW_YELLOW else "green")
    deep    = ("red"    if knee_deg < DEPTH_KNEE_DEEP_RED else
               "yellow" if knee_deg < DEPTH_KNEE_DEEP_YELLOW else "green")
    return shallow, deep


def classify_depth_phase_aware(value, phase_label, baseline=None):
    """
    Phase-aware wrapper around classify_depth.
    Returns "green" for all phases except BOTTOM so that depth warnings don't
    fire while the subject is standing, descending, or ascending.

    NOTE this gate cannot catch a squat that stops well short of BOTTOM — the
    fault is what keeps it out of the phase that would grade it.  That case is
    handled by RepCounter's shallow-rep detection, which arms a synthetic red
    depth zone instead of relying on this classifier.
    """
    if phase_label == "BOTTOM":
        return classify_depth(value, baseline)
    return "green"


def classify_valgus_3d(value):
    """
    Classify 3-D knee valgus from knee_deviation_3d().

    Uses the absolute value so the same thresholds apply to both legs
    regardless of sign (left-leg valgus is positive, right-leg is negative).
    The direction is already captured by the per-side 2-D valgus zones;
    this zone adds the depth-corrected magnitude.
    """
    abs_val = abs(value)
    if abs_val > VALGUS_3D_RED:
        return "red"
    if abs_val > VALGUS_3D_YELLOW:
        return "yellow"
    return "green"


def classify_asymmetry(asymmetry_score):
    """
    Classify left-right movement asymmetry from the combined knee + hip
    symmetry difference score.

    asymmetry_score = max(|knee_sym_diff_deg|, |hip_sym_diff_deg|)

    GREEN  : score <  5°
    YELLOW : score 5–10°
    RED    : score > 10°
    """
    if asymmetry_score > ASYMMETRY_RED:
        return "red"
    if asymmetry_score > ASYMMETRY_YELLOW:
        return "yellow"
    return "green"


# =============================================================================
# SIMPLE PHASE DETECTION  (from knee angle signal only)
# =============================================================================

def _get_phase_label(knee_dq):
    """
    Derive a simple squat phase from the smoothed knee-angle deque.

    BOTTOM  : smoothed avg angle < DEPTH_PHASE_BOTTOM_THRESHOLD (110°)
    DESCENT : angle trending downward  (oldest > newest by > 1°)
    ASCENT  : angle trending upward    (newest > oldest by > 1°)
    STANDING: otherwise

    The deque spans SMOOTH_N frames so velocity is estimated over a short
    window rather than a single noisy frame-to-frame difference.
    """
    avg = sum(knee_dq) / len(knee_dq)
    if avg < DEPTH_PHASE_BOTTOM_THRESHOLD:
        return "BOTTOM"
    if len(knee_dq) >= 2:
        delta = knee_dq[-1] - knee_dq[0]   # positive = angle increasing = ascending
        if delta < -1.0:
            return "DESCENT"
        if delta > 1.0:
            return "ASCENT"
    return "STANDING"


# =============================================================================
# COLOR MAPPING
# =============================================================================

def get_zone_color(zone):
    """
    Map a zone string to a BGR colour tuple for OpenCV drawing.

      green  (0, 255, 120) — acceptable form
      yellow (0, 220, 255) — technique warning
      red    (80, 80, 255) — likely problematic technique
    """
    return {
        "green":  (0, 255, 120),
        "yellow": (0, 220, 255),
        "red":    (80, 80, 255),
    }.get(zone, (200, 200, 200))


# =============================================================================
# AGGREGATION
# =============================================================================

def overall_zone(zone_list):
    """
    Reduce a list of zone strings to one overall status.
      Any "red"    in the list → "red"
      Any "yellow" in the list → "yellow"
      All "green"              → "green"
    """
    if "red"    in zone_list: return "red"
    if "yellow" in zone_list: return "yellow"
    return "green"


# =============================================================================
# SMOOTHED ZONE COMPUTATION
# =============================================================================

def compute_zones(feats, smoother):
    """
    Push this frame's noisy metrics into rolling deque buffers, average them,
    then classify each zone.

    Smoothing over SMOOTH_N frames prevents single-frame pose jitter from
    flipping the colour indicator every few frames.

    Parameters
    ----------
    feats    : dict from measurements.compute_squat_features()
    smoother : dict of deque(maxlen=SMOOTH_N) buffers, keys:
                 "valgus_l", "valgus_r", "trunk_dev", "depth",
                 "valgus_3d_l", "valgus_3d_r",
                 "knee_angle",  "asymmetry"    ← new additions

    Returns a dict:
      valgus_l, valgus_r, trunk, overall          → zone strings
      depth        → too SHALLOW at BOTTOM (knee angle above the band)
      depth_excess → too DEEP at BOTTOM (knee angle below the band)
      valgus_3d_l, valgus_3d_r                   → 3-D valgus zone strings
      asymmetry_score                             → smoothed numeric score (float)
      asymmetry                                   → asymmetry zone string
      phase_label                                 → simple phase string
    """
    smoother["valgus_l"].append(feats["norm_knee_offset_l"])
    smoother["valgus_r"].append(feats["norm_knee_offset_r"])
    smoother["trunk_dev"].append(feats["trunk_lean_dev_deg"])
    smoother["depth"].append(feats["norm_hip_depth"])
    smoother["valgus_3d_l"].append(feats["knee_valgus_3d_l"])
    smoother["valgus_3d_r"].append(feats["knee_valgus_3d_r"])

    # Knee angle buffer for simple phase detection
    avg_knee = (feats["knee_angle_l_deg"] + feats["knee_angle_r_deg"]) / 2.0
    smoother["knee_angle"].append(avg_knee)

    # Asymmetry score buffer
    raw_asym = max(abs(feats["knee_sym_diff_deg"]), abs(feats["hip_sym_diff_deg"]))
    smoother["asymmetry"].append(raw_asym)

    def mean(dq):
        return sum(dq) / len(dq)

    zl   = classify_valgus_left(mean(smoother["valgus_l"]))
    zr   = classify_valgus_right(mean(smoother["valgus_r"]))
    zt   = classify_trunk_deviation(mean(smoother["trunk_dev"]))
    z3dl = classify_valgus_3d(mean(smoother["valgus_3d_l"]))
    z3dr = classify_valgus_3d(mean(smoother["valgus_3d_r"]))

    # Depth, from the smoothed knee angle, evaluated only at BOTTOM.
    #
    # The BOTTOM gate makes the LIVE indicator honest about what is knowable
    # frame by frame: mid-descent the knee necessarily passes through every
    # shallow angle on its way down, so grading depth then would read red on the
    # way to a perfectly good squat.  "Too deep", by contrast, IS knowable live —
    # if the knee is at 50 degrees right now, it is at 50 degrees right now.
    #
    # The consequence is that a rep which never reaches BOTTOM cannot be graded
    # here at all — see DEPTH_PHASE_BOTTOM_THRESHOLD's note on that circularity.
    # RepCounter grades every completed rep from its deepest point instead and
    # arms a synthetic zone, which is what actually catches a short squat.
    phase_label   = _get_phase_label(smoother["knee_angle"])
    smooth_knee   = mean(smoother["knee_angle"])
    if phase_label == "BOTTOM":
        zd, zd_excess = squat_depth_zones(smooth_knee)
    else:
        zd, zd_excess = "green", "green"

    # Asymmetry zone
    smooth_asym = mean(smoother["asymmetry"])
    za          = classify_asymmetry(smooth_asym)

    # Overall zone aggregates: 2D valgus L (zl), 2D valgus R (zr), trunk (zt),
    # and BOTH sides of the depth band — too shallow (zd) and too deep
    # (zd_excess).  Going well past parallel is a fault the overall status has
    # to reflect; leaving it out was what let an over-deep rep read GOOD.
    #
    # Three channels excluded from overall (still computed and CSV-logged):
    #   za (asymmetry): all three formula variants (max/knee-only/mean) yield
    #       p95 ~25 deg on 7,292 front-view correct-form frames (REHAB24-6).
    #       Noise originates in per-frame knee-angle estimates under the Lite model
    #       on a front-facing camera, not in the aggregation formula. Not reliably
    #       measurable from the deployment view. Excluded consistently with 3D valgus.
    #   z3dl, z3dr (3D valgus L/R): Lite model z-depth is too noisy -- correct-form
    #       reads ~1.0, making the metric non-discriminative. Recovery requires the
    #       Full/Heavy model. Logged for future model-upgrade analysis.
    zo = overall_zone([zl, zr, zt, zd, zd_excess])

    return {
        "valgus_l":        zl,
        "valgus_r":        zr,
        "trunk":           zt,
        "depth":           zd,
        "depth_excess":    zd_excess,
        "overall":         zo,
        "valgus_3d_l":     z3dl,
        "valgus_3d_r":     z3dr,
        "asymmetry_score": smooth_asym,
        "asymmetry":       za,
        "phase_label":     phase_label,
    }
