"""
pose_utils.py — Geometry and math helpers used across the project.

All functions here are pure maths with no side effects — they take numbers
in and return numbers out, making them easy to test in isolation.

2-D helpers  : get_px, midpoint, dist2d, safe_div, angle_at_joint, trunk_lean_angle
3-D helpers  : get_3d, angle_at_joint_3d, knee_deviation_3d
"""

import math
import cv2
import numpy as np


def get_px(lm, idx, w, h):
    """
    Convert a MediaPipe landmark's normalised (0–1) coordinates to pixel
    coordinates for a frame of size w × h.

    Parameters
    ----------
    lm  : list of 33 NormalizedLandmark objects (one pose detection result)
    idx : landmark index, 0–32
    w   : frame width  in pixels
    h   : frame height in pixels

    Returns (x_px, y_px) as integers.
    """
    return (int(lm[idx].x * w), int(lm[idx].y * h))


def midpoint(a, b):
    """Return the integer pixel midpoint between two (x, y) tuples."""
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def dist2d(a, b):
    """Euclidean distance between two 2-D pixel points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def safe_div(numerator, denominator, fallback=0.0):
    """
    Division that never raises ZeroDivisionError.
    Returns `fallback` whenever |denominator| < 1e-6.
    Used in every normalisation formula to guard against degenerate frames.
    """
    return numerator / denominator if abs(denominator) > 1e-6 else fallback


def angle_at_joint(a, b, c):
    """
    Interior angle (degrees) at vertex `b` in the triangle a – b – c.

    Formula:  cos θ = (b→a · b→c) / (|b→a| × |b→c|)

    Uses np.clip to keep the cosine in [-1, 1] before arccos, preventing NaN
    from floating-point rounding.  Returns 0.0 for degenerate cases (zero-
    length limb segment).

    Typical uses:
      Knee angle  → a=hip,      b=knee,  c=ankle   (≈180° standing, ≈90° parallel squat)
      Hip  angle  → a=shoulder, b=hip,   c=knee    (≈170° standing, decreases while squatting)
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def trunk_lean_angle(shoulder_mid, hip_mid):
    """
    Angle (degrees) of the torso vector (hip → shoulder) from the image vertical.

    Image y increases downward, so 'straight up' in pixel coords is (0, -1).
    Using atan2(Δx, −Δy) gives:
        0°       = perfectly upright
        positive = torso top tilting right  (or forward with a side-facing camera)
        negative = torso top tilting left

    Note — this is a 2-D projection:
      • Front-facing camera  → captures left / right lateral tilt
      • Side-facing camera   → captures forward / backward lean (best for squat analysis)
    The deviation from the standing baseline is what matters most for form feedback.
    """
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    return math.degrees(math.atan2(dx, -dy))


# =============================================================================
# 3-D HELPERS  (use MediaPipe's z depth estimate)
# =============================================================================
#
# MediaPipe z-coordinate properties
# ──────────────────────────────────
# • Origin  : midpoint of the two hips  (z ≈ 0 there)
# • Sign    : SMALLER value = CLOSER to the camera
# • Scale   : roughly the same as x (normalised, ~0–1 range)
# • Accuracy: estimated from 2-D images by a neural net; noisier than x/y,
#             especially with the Lite model.  Use with appropriate smoothing.
#
# These functions work in pure normalised (x, y, z) space — no pixel conversion.

def get_3d(lm, idx):
    """
    Return the normalised (x, y, z) position of landmark `idx` as a
    numpy float64 array.

    z is MediaPipe's depth estimate (smaller = closer to camera).
    """
    return np.array([lm[idx].x, lm[idx].y, lm[idx].z], dtype=float)


def angle_at_joint_3d(a, b, c):
    """
    Interior angle (degrees) at vertex `b` in the 3-D triangle a – b – c.

    Identical formula to the 2-D version but accepts numpy 3-element arrays,
    so it uses the full (x, y, z) position rather than just (x, y).
    This gives a more accurate joint angle when the person is not perfectly
    perpendicular to the camera.

    Returns 0.0 for degenerate (zero-length) segments.
    """
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def knee_deviation_3d(hip, knee, ankle):
    """
    Signed, length-normalised lateral deviation of the knee from the
    hip-to-ankle axis in 3-D space.

    Algorithm
    ─────────
    1. Form the hip → ankle reference axis (the straight-leg line).
    2. Project the hip → knee vector onto that axis to find the closest
       point on the axis to the knee.
    3. The perpendicular offset (knee − projected point) is the true 3-D
       displacement of the knee from the anatomical reference line.
    4. Normalise by hip-to-ankle distance (full leg length).
    5. Sign: positive x-component of the deviation → knee is to the right
       in the image frame → valgus direction for the left leg.

    Sign convention (matches the existing 2-D norm_knee_offset):
      Left  leg  : positive value = valgus (knee caving right in image)
      Right leg  : negative value = valgus (knee caving left  in image)

    Parameters — all numpy float64 arrays from get_3d()
    Returns     — float, signed normalised 3-D deviation
    """
    ha = ankle - hip      # hip → ankle vector (reference axis)
    hk = knee  - hip      # hip → knee  vector

    ha_sq = float(np.dot(ha, ha))
    if ha_sq < 1e-12:
        return 0.0

    # Point on the hip-ankle line closest to the knee
    t    = float(np.dot(hk, ha)) / ha_sq
    proj = hip + t * ha

    # 3-D perpendicular deviation of the knee from the reference line
    dev = knee - proj

    # Magnitude normalised by full leg length
    leg_len  = math.sqrt(ha_sq)
    norm_dev = float(np.linalg.norm(dev)) / leg_len

    # Sign from x-component: positive x → rightward in image → left-leg valgus
    sign = 1.0 if dev[0] >= 0.0 else -1.0
    return sign * norm_dev


def landmarks_reliable(landmarks, indices, min_visibility=0.6):
    """
    Returns True if every landmark in `indices` has visibility >= min_visibility.

    Parameters
    ----------
    landmarks      : list of MediaPipe NormalizedLandmark — one frame's pose result
    indices        : list of int — MediaPipe landmark indices to check
    min_visibility : float — minimum acceptable visibility score (0–1)

    Returns False if landmarks is None or empty, or if any checked landmark
    falls below the threshold.  Use this before computing joint angles to skip
    frames where occlusion or motion blur makes coordinates unreliable.
    """
    if not landmarks:
        return False
    for i in indices:
        if landmarks[i].visibility < min_visibility:
            return False
    return True
