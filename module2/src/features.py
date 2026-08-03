"""
measurements.py — Body segment lengths and per-frame squat feature computation.

Two public functions:
  compute_body_measurements()  — pixel lengths of limb segments (shoulder width,
                                  hip width, femurs, tibias, torso)
  compute_squat_features()     — normalised squat metrics (stance width, knee
                                  offsets, hip depth, joint angles, trunk lean)
"""

from config import (
    IDX_SHOULDER_L, IDX_SHOULDER_R,
    IDX_HIP_L,      IDX_HIP_R,
    IDX_KNEE_L,     IDX_KNEE_R,
    IDX_ANKLE_L,    IDX_ANKLE_R,
)
import numpy as np
from src.pose_utils import (
    get_px, midpoint, dist2d, safe_div,
    angle_at_joint, trunk_lean_angle,
    get_3d, angle_at_joint_3d, knee_deviation_3d,
)


def compute_body_measurements(lm, w, h):
    """
    Estimate body segment lengths in pixels from one pose detection result.

    These values vary a few percent frame-to-frame due to model jitter.
    They are used as body-relative denominators when normalising squat metrics.

    Returns a dict:
      shoulder_width_px, hip_width_px, torso_len_px,
      femur_l_px, femur_r_px, tibia_l_px, tibia_r_px,
      avg_femur_px, avg_tibia_px
    """
    sh_l = get_px(lm, IDX_SHOULDER_L, w, h)
    sh_r = get_px(lm, IDX_SHOULDER_R, w, h)
    hi_l = get_px(lm, IDX_HIP_L,      w, h)
    hi_r = get_px(lm, IDX_HIP_R,      w, h)
    kn_l = get_px(lm, IDX_KNEE_L,     w, h)
    kn_r = get_px(lm, IDX_KNEE_R,     w, h)
    an_l = get_px(lm, IDX_ANKLE_L,    w, h)
    an_r = get_px(lm, IDX_ANKLE_R,    w, h)

    shoulder_width = dist2d(sh_l, sh_r)
    hip_width      = dist2d(hi_l, hi_r)

    # Torso proxy: shoulder midpoint → hip midpoint
    torso_len = dist2d(midpoint(sh_l, sh_r), midpoint(hi_l, hi_r))

    # Femur proxy: hip → knee (upper leg)
    femur_l = dist2d(hi_l, kn_l)
    femur_r = dist2d(hi_r, kn_r)

    # Tibia proxy: knee → ankle (lower leg)
    tibia_l = dist2d(kn_l, an_l)
    tibia_r = dist2d(kn_r, an_r)

    avg_femur = (femur_l + femur_r) / 2
    avg_tibia = (tibia_l + tibia_r) / 2

    return {
        "shoulder_width_px": shoulder_width,
        "hip_width_px":      hip_width,
        "torso_len_px":      torso_len,
        "femur_l_px":        femur_l,
        "femur_r_px":        femur_r,
        "tibia_l_px":        tibia_l,
        "tibia_r_px":        tibia_r,
        "avg_femur_px":      avg_femur,
        "avg_tibia_px":      avg_tibia,
    }


def compute_squat_features(lm, meas, calib, w, h):
    """
    Compute one frame's squat-specific metrics.

    Parameters
    ----------
    lm    : list of 33 MediaPipe NormalizedLandmark objects
    meas  : dict returned by compute_body_measurements()
    calib : calibration state dict (keys: 'trunk_lean', 'done')
    w, h  : frame pixel dimensions

    Returns a flat dict whose keys match SQUAT_CSV_HEADER (minus zone columns).
    """
    sh_l = get_px(lm, IDX_SHOULDER_L, w, h)
    sh_r = get_px(lm, IDX_SHOULDER_R, w, h)
    hi_l = get_px(lm, IDX_HIP_L,      w, h)
    hi_r = get_px(lm, IDX_HIP_R,      w, h)
    kn_l = get_px(lm, IDX_KNEE_L,     w, h)
    kn_r = get_px(lm, IDX_KNEE_R,     w, h)
    an_l = get_px(lm, IDX_ANKLE_L,    w, h)
    an_r = get_px(lm, IDX_ANKLE_R,    w, h)

    hip_width = meas["hip_width_px"]
    avg_femur = meas["avg_femur_px"]

    # ── Stance width ──────────────────────────────────────────────────────────
    # Distance between both ankles.
    # norm_stance_width = 1.0 → stance equals hip width (neutral stance)
    # > 1.0 = wider than hips  |  < 1.0 = narrower
    stance_width      = dist2d(an_l, an_r)
    norm_stance_width = safe_div(stance_width, hip_width)

    # ── Knee medial displacement (valgus / varus proxy) ───────────────────────
    # Each knee's horizontal pixel offset from its same-side ankle.
    # Left  leg offset > 0 → knee right of ankle → valgus (knee caving in)
    # Right leg offset < 0 → knee left  of ankle → valgus (knee caving in)
    # Dividing by hip_width gives a body-size-relative normalised ratio.
    knee_offset_l   = kn_l[0] - an_l[0]          # pixels, signed
    knee_offset_r   = kn_r[0] - an_r[0]          # pixels, signed
    norm_knee_off_l = safe_div(knee_offset_l, hip_width)
    norm_knee_off_r = safe_div(knee_offset_r, hip_width)

    # ── Hip depth ─────────────────────────────────────────────────────────────
    # Vertical distance from hip midpoint to ankle midpoint, divided by the
    # average femur length.  Image y increases downward, so ankle_y > hip_y when
    # standing normally.  As the person squats, hip_y increases and hip_height
    # shrinks — so HIGHER value = shallower, LOWER value = deeper.
    #
    # Measured on REHAB24-6 Ex6 front-view frames (clean re-run 2026-07-27),
    # median norm_hip_depth by rep phase:
    #   STANDING ≈ 1.92        DESCENT ≈ 1.91
    #   ASCENT   ≈ 1.92        BOTTOM  ≈ 1.72
    #
    # The scale therefore runs ≈2.0 standing down to ≈1.7 at the bottom of a
    # squat; it does NOT approach 1.0 or below in normal execution.  Depth zone
    # thresholds in config.py are set against this scale (DEPTH_YELLOW = 2.2836,
    # DEPTH_RED = 2.4794 — exceeding them means TOO SHALLOW at BOTTOM phase).
    hip_mid        = midpoint(hi_l, hi_r)
    an_mid         = midpoint(an_l, an_r)
    hip_height     = an_mid[1] - hip_mid[1]
    norm_hip_depth = safe_div(hip_height, avg_femur)

    # Depth as a fraction of THIS user's own standing depth: 1.0 while standing,
    # lower as they descend.  The absolute scale above varies with build and
    # camera distance far more than the depth thresholds allow for — see the
    # DEPTH_RATIO_* note in config.py.  Falls back to the current frame before
    # calibration completes, so the ratio reads 1.0 rather than dividing by None.
    base_hip_depth  = calib.get("hip_depth") or norm_hip_depth
    norm_depth_ratio = safe_div(norm_hip_depth, base_hip_depth)

    # ── Trunk lean ────────────────────────────────────────────────────────────
    shoulder_mid   = midpoint(sh_l, sh_r)
    lean           = trunk_lean_angle(shoulder_mid, hip_mid)
    base_lean      = calib.get("trunk_lean", lean)   # use current if not yet calibrated
    lean_deviation = lean - base_lean

    # ── Knee angles ───────────────────────────────────────────────────────────
    # Triangle: hip – knee – ankle.
    # 180° = fully extended leg.  ~90° = thighs parallel (good squat depth).
    knee_angle_l  = angle_at_joint(hi_l, kn_l, an_l)
    knee_angle_r  = angle_at_joint(hi_r, kn_r, an_r)
    knee_sym_diff = abs(knee_angle_l - knee_angle_r)

    # ── Hip angles ────────────────────────────────────────────────────────────
    # Triangle: shoulder – hip – knee.
    # 170–180° = standing upright.  Decreases as hip flexion increases during squat.
    hip_angle_l  = angle_at_joint(sh_l, hi_l, kn_l)
    hip_angle_r  = angle_at_joint(sh_r, hi_r, kn_r)
    hip_sym_diff = abs(hip_angle_l - hip_angle_r)

    # ── 3-D metrics (use MediaPipe z depth estimate) ──────────────────────────
    # All computed in normalised (x, y, z) space — no pixel conversion needed.
    # z is the neural-net depth estimate: smaller = closer to camera.

    hi_3d_l = get_3d(lm, IDX_HIP_L)
    hi_3d_r = get_3d(lm, IDX_HIP_R)
    kn_3d_l = get_3d(lm, IDX_KNEE_L)
    kn_3d_r = get_3d(lm, IDX_KNEE_R)
    an_3d_l = get_3d(lm, IDX_ANKLE_L)
    an_3d_r = get_3d(lm, IDX_ANKLE_R)

    # 3-D knee joint angle — uses full (x, y, z) triangle; more accurate than
    # the 2-D version when the person isn't exactly perpendicular to the camera.
    knee_angle_3d_l  = angle_at_joint_3d(hi_3d_l, kn_3d_l, an_3d_l)
    knee_angle_3d_r  = angle_at_joint_3d(hi_3d_r, kn_3d_r, an_3d_r)
    knee_3d_sym_diff = abs(knee_angle_3d_l - knee_angle_3d_r)

    # 3-D valgus — signed deviation of the knee from the hip-ankle axis in 3-D.
    # Positive left / negative right = valgus direction.
    # Uses the anatomically correct reference line (hip-to-ankle) rather than
    # just the ankle position, and incorporates z-depth to compensate for
    # off-axis camera angles.
    knee_valgus_3d_l = knee_deviation_3d(hi_3d_l, kn_3d_l, an_3d_l)
    knee_valgus_3d_r = knee_deviation_3d(hi_3d_r, kn_3d_r, an_3d_r)

    # Knee forward tracking — how far the knee is in front of the ankle in the
    # depth (z) direction, normalised by 3-D tibia length.
    # Positive = knee is closer to the camera than the ankle = knee tracks forward.
    # This captures "knee-over-toe" depth that 2-D analysis cannot see at all.
    tibia_3d_l    = float(np.linalg.norm(an_3d_l - kn_3d_l))
    tibia_3d_r    = float(np.linalg.norm(an_3d_r - kn_3d_r))
    knee_fwd_l = safe_div(float(an_3d_l[2] - kn_3d_l[2]), tibia_3d_l)
    knee_fwd_r = safe_div(float(an_3d_r[2] - kn_3d_r[2]), tibia_3d_r)

    return {
        "stance_width_px":         stance_width,
        "norm_stance_width":       norm_stance_width,
        "knee_offset_l_px":        knee_offset_l,
        "knee_offset_r_px":        knee_offset_r,
        "norm_knee_offset_l":      norm_knee_off_l,
        "norm_knee_offset_r":      norm_knee_off_r,
        "hip_height_px":           hip_height,
        "norm_hip_depth":          norm_hip_depth,
        "baseline_hip_depth":      base_hip_depth,
        "norm_depth_ratio":        norm_depth_ratio,
        "trunk_lean_deg":          lean,
        "baseline_trunk_lean_deg": base_lean,
        "trunk_lean_dev_deg":      lean_deviation,
        "knee_angle_l_deg":        knee_angle_l,
        "knee_angle_r_deg":        knee_angle_r,
        "knee_sym_diff_deg":       knee_sym_diff,
        "hip_angle_l_deg":         hip_angle_l,
        "hip_angle_r_deg":         hip_angle_r,
        "hip_sym_diff_deg":        hip_sym_diff,
        # 3-D metrics
        "knee_angle_3d_l_deg":     knee_angle_3d_l,
        "knee_angle_3d_r_deg":     knee_angle_3d_r,
        "knee_3d_sym_diff_deg":    knee_3d_sym_diff,
        "knee_valgus_3d_l":        knee_valgus_3d_l,
        "knee_valgus_3d_r":        knee_valgus_3d_r,
        "knee_fwd_l":              knee_fwd_l,
        "knee_fwd_r":              knee_fwd_r,
    }
