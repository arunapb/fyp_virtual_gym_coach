"""
view_detection.py — camera-view detector (Stage 1: DETECTION ONLY).

Reports whether the subject is being filmed from the FRONT, at a DIAGONAL, or
in profile (SIDE), from landmark geometry alone.

WHY THIS EXISTS
───────────────
Every threshold in config.py was derived from FRONT-view frames.  Frontal-plane
metrics (knee valgus, left/right asymmetry, elbow drift) are not computable from
a profile view at all, and the landmark visibility gate rejects ~92-100% of
side-view frames because the far-side limb is self-occluded.  Before any channel
can be made view-dependent, view must be shown to be detectable reliably.  This
module produces that evidence.

STAGE 1 SCOPE — STRICT
──────────────────────
This is a PURE OBSERVER.  Nothing in session.py, any exercise, the visibility
gate or the audio layer imports or consumes it.  It changes no channel, no
threshold, no gate and no cue.  It exists to be measured.

GEOMETRIC BASIS OF THE SIGNALS
──────────────────────────────
As the camera rotates from frontal to profile, the body's left-right axis turns
away from the image plane, so every horizontal (left-to-right) measurement
foreshortens toward zero while vertical measurements are unaffected.  Each
signal below is a ratio built to exploit that, normalised so it is free of both
subject size and camera distance.

The classifier maps each signal onto a "profileness" in [0, 1] by linear
interpolation between a FRONT anchor and a SIDE anchor, then takes a weighted
mean.  Anchors and weights are named constants in config.py.
"""

import math
from collections import deque, Counter
from dataclasses import dataclass, field

from config import (
    VIEW_SMOOTH_FRAMES, VIEW_MIN_CONFIDENCE,
    VIEW_FRONT_MAX_SCORE, VIEW_SIDE_MIN_SCORE,
    VIEW_ANCHORS, VIEW_SIGNAL_WEIGHTS, VIEW_LOG_SIGNALS,
    IDX_SHOULDER_L, IDX_SHOULDER_R, IDX_ELBOW_L, IDX_ELBOW_R,
    IDX_WRIST_L, IDX_WRIST_R, IDX_HIP_L, IDX_HIP_R,
    IDX_KNEE_L, IDX_KNEE_R, IDX_ANKLE_L, IDX_ANKLE_R,
)

# Head landmarks used for the auxiliary orientation signals.
IDX_NOSE = 0
IDX_EAR_L, IDX_EAR_R = 7, 8

_LEFT_CHAIN = [IDX_SHOULDER_L, IDX_ELBOW_L, IDX_WRIST_L,
               IDX_HIP_L, IDX_KNEE_L, IDX_ANKLE_L]
_RIGHT_CHAIN = [IDX_SHOULDER_R, IDX_ELBOW_R, IDX_WRIST_R,
                IDX_HIP_R, IDX_KNEE_R, IDX_ANKLE_R]

_EPS = 1e-6

VIEW_FRONT = "front"
VIEW_DIAGONAL = "diagonal"
VIEW_SIDE = "side"
VIEW_UNKNOWN = "unknown"


@dataclass
class ViewResult:
    """One frame's view verdict, raw and smoothed."""
    raw_view: str                  # this frame alone
    raw_confidence: float
    view: str                      # smoothed over the rolling window
    confidence: float
    score: float                   # weighted profileness, 0 = front, 1 = side
    signals: dict = field(default_factory=dict)


def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def compute_view_signals(lm, w, h) -> dict:
    """
    Pure geometry — no thresholds, no state.  Returns the raw signal values.

    Kept deliberately separate from classification so that a validation harness
    can cache these once (the expensive MediaPipe pass) and then re-classify
    under different anchors without re-running pose estimation.

    Pixel coordinates are used wherever a horizontal and a vertical distance are
    mixed, so the frame's aspect ratio cannot distort the ratio.  Pure
    horizontal:horizontal ratios use normalised x directly, which is
    aspect-free by construction.
    """
    def px(i):
        return (lm[i].x * w, lm[i].y * h)

    def nx(i):
        return lm[i].x

    # ── (a) hip width / shoulder width ───────────────────────────────────────
    # Both hips project to nearly the same x in profile, so hip width collapses
    # hard; shoulder width collapses less because the shoulders sit further apart
    # along the same axis and the near shoulder stays visible.
    hip_w = abs(nx(IDX_HIP_L) - nx(IDX_HIP_R))
    sh_w = abs(nx(IDX_SHOULDER_L) - nx(IDX_SHOULDER_R))
    hip_shoulder_ratio = hip_w / (sh_w + _EPS)

    # ── (b) left/right visibility asymmetry ──────────────────────────────────
    # The far-side limb is occluded by the torso in profile, so MediaPipe's
    # confidence collapses on one side only.  Near-symmetric head-on.
    vis_l = _mean([lm[i].visibility for i in _LEFT_CHAIN])
    vis_r = _mean([lm[i].visibility for i in _RIGHT_CHAIN])
    vis_asym = abs(vis_l - vis_r)
    vis_asym_signed = vis_l - vis_r          # sign tells WHICH way the body faces

    # ── (c) shoulder width / torso length ────────────────────────────────────
    # Shoulder width foreshortens in profile; torso length (a vertical measure)
    # does not, so the ratio falls.
    sh_l, sh_r = px(IDX_SHOULDER_L), px(IDX_SHOULDER_R)
    hi_l, hi_r = px(IDX_HIP_L), px(IDX_HIP_R)
    sh_mid = ((sh_l[0] + sh_r[0]) / 2.0, (sh_l[1] + sh_r[1]) / 2.0)
    hi_mid = ((hi_l[0] + hi_r[0]) / 2.0, (hi_l[1] + hi_r[1]) / 2.0)
    sh_w_px = abs(sh_l[0] - sh_r[0])
    torso_len = ((sh_mid[0] - hi_mid[0]) ** 2 + (sh_mid[1] - hi_mid[1]) ** 2) ** 0.5
    shoulder_torso_ratio = sh_w_px / (torso_len + _EPS)

    # ── (d1) shoulder depth spread ───────────────────────────────────────────
    # MediaPipe's z estimate: in profile the two shoulders separate in DEPTH
    # while converging in x, so this ratio explodes.  Head-on it is near zero.
    z_spread = abs(lm[IDX_SHOULDER_L].z - lm[IDX_SHOULDER_R].z)
    shoulder_z_ratio = z_spread / (sh_w + _EPS)

    # ── (d2) stance width / shoulder width ───────────────────────────────────
    # The feet are planted apart along the same left-right axis, so stance
    # collapses in profile too — an independent confirmation of (a) that does not
    # depend on the hips being tracked well.
    ankle_w = abs(nx(IDX_ANKLE_L) - nx(IDX_ANKLE_R))
    ankle_shoulder_ratio = ankle_w / (sh_w + _EPS)

    # ── (d3) ear visibility asymmetry ────────────────────────────────────────
    # The head turns with the body; in profile one ear is fully occluded.
    ear_vis_asym = abs(lm[IDX_EAR_L].visibility - lm[IDX_EAR_R].visibility)

    # ── (d4) nose offset from the shoulder midline ───────────────────────────
    # Head-on the nose sits near the middle of the shoulder span; in profile it
    # projects out past the shoulders, so |offset| grows beyond ~0.5.
    nose_offset = abs(nx(IDX_NOSE) - (nx(IDX_SHOULDER_L) + nx(IDX_SHOULDER_R)) / 2.0)
    nose_shoulder_ratio = nose_offset / (sh_w + _EPS)

    return {
        "hip_shoulder_ratio":    hip_shoulder_ratio,
        "vis_asym":              vis_asym,
        "vis_asym_signed":       vis_asym_signed,
        "shoulder_torso_ratio":  shoulder_torso_ratio,
        "shoulder_z_ratio":      shoulder_z_ratio,
        "ankle_shoulder_ratio":  ankle_shoulder_ratio,
        "ear_vis_asym":          ear_vis_asym,
        "nose_shoulder_ratio":   nose_shoulder_ratio,
    }


def _profileness(name, value):
    """
    Map one raw signal onto [0, 1] profileness by interpolating between its
    FRONT anchor and its SIDE anchor.  Handles either polarity: some signals
    fall toward profile (shoulder_torso_ratio), others rise (vis_asym).

    Signals listed in VIEW_LOG_SIGNALS are interpolated in LOG space.  Those two
    span more than two orders of magnitude between frontal and profile
    (shoulder_z_ratio runs 0.16 -> 17), and on a linear scale that range is so
    dominated by the profile tail that a genuine diagonal lands within a few
    percent of the front anchor and is indistinguishable from it.  In log space
    the diagonal median falls near the middle of the range, which is what makes
    the three-way split possible at all.
    """
    front, side = VIEW_ANCHORS[name]
    if name in VIEW_LOG_SIGNALS:
        front = math.log(max(front, _EPS))
        side = math.log(max(side, _EPS))
        value = math.log(max(value, _EPS))
    if abs(side - front) < _EPS:
        return 0.0
    return _clamp01((value - front) / (side - front))


def classify_signals(signals: dict):
    """
    Combine the per-signal profileness scores into a label and a confidence.

    Returns (label, confidence, score).  `score` is the weighted profileness:
    0.0 = unambiguously frontal, 1.0 = unambiguously profile.

    Confidence is the distance from the score to the NEAREST band boundary,
    rescaled so that the middle of a band reads 1.0 and a value sitting exactly
    on a boundary reads 0.0.  A score parked between two bands is therefore
    reported as low-confidence rather than being silently forced into one — which
    is the behaviour that lets an ambiguous view surface as "unknown".
    """
    parts = {n: _profileness(n, signals[n]) for n in VIEW_SIGNAL_WEIGHTS}
    total_w = sum(VIEW_SIGNAL_WEIGHTS.values())
    score = sum(parts[n] * VIEW_SIGNAL_WEIGHTS[n] for n in parts) / (total_w + _EPS)

    lo, hi = VIEW_FRONT_MAX_SCORE, VIEW_SIDE_MIN_SCORE
    if score <= lo:
        label = VIEW_FRONT
        conf = _clamp01((lo - score) / (lo + _EPS))
    elif score >= hi:
        label = VIEW_SIDE
        conf = _clamp01((score - hi) / (1.0 - hi + _EPS))
    else:
        label = VIEW_DIAGONAL
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        conf = _clamp01(1.0 - abs(score - mid) / (half + _EPS))

    if conf < VIEW_MIN_CONFIDENCE:
        label = VIEW_UNKNOWN
    return label, conf, score


class ViewDetector:
    """
    Rolling-window camera-view detector.

    Call update(lm, w, h) once per frame.  Each call returns a ViewResult
    carrying BOTH the raw single-frame verdict and the smoothed one, so the
    validation harness can measure how much the smoothing is actually buying.

    Smoothing is a confidence-weighted majority vote over the last
    VIEW_SMOOTH_FRAMES frames.  Weighting by confidence rather than counting
    votes equally stops a run of barely-decided frames from outvoting a smaller
    number of clearly-decided ones.  "unknown" frames are retained in the window
    and vote for "unknown", so a genuinely ambiguous stretch stays ambiguous
    instead of being smoothed into a guess.
    """

    def __init__(self, window: int = VIEW_SMOOTH_FRAMES):
        self._window = deque(maxlen=window)
        self.last: ViewResult | None = None

    def reset(self):
        self._window.clear()
        self.last = None

    def update(self, lm, w, h) -> ViewResult:
        """Advance one frame from raw landmarks."""
        return self.update_from_signals(compute_view_signals(lm, w, h))

    def update_from_signals(self, signals: dict) -> ViewResult:
        """
        Advance one frame from PRE-COMPUTED signals.

        Exists so an offline validation harness can replay cached signals through
        the exact smoothing path used live, instead of reimplementing the rolling
        window — a duplicate implementation could drift from this one and would
        then be validating something the product does not do.
        """
        raw_label, raw_conf, score = classify_signals(signals)
        self._window.append((raw_label, raw_conf))

        weights = Counter()
        for label, conf in self._window:
            weights[label] += conf
        if weights:
            best = max(weights.items(), key=lambda kv: kv[1])
            total = sum(weights.values())
            smooth_label = best[0]
            smooth_conf = best[1] / (total + _EPS)
        else:                                             # pragma: no cover
            smooth_label, smooth_conf = VIEW_UNKNOWN, 0.0

        # A window dominated by low-confidence frames should not be promoted to a
        # confident label just because they agree with each other.
        mean_conf = _mean([c for _, c in self._window])
        if mean_conf < VIEW_MIN_CONFIDENCE:
            smooth_label = VIEW_UNKNOWN

        self.last = ViewResult(raw_view=raw_label, raw_confidence=raw_conf,
                               view=smooth_label, confidence=smooth_conf,
                               score=score, signals=signals)
        return self.last
