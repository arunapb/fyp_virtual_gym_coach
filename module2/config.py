"""
config.py — All constants, landmark indices, thresholds, and CSV schema.

Every other module imports from here, so changing a value in this file
automatically applies everywhere in the project.

Layout note.  This file sits at the repository ROOT, alongside the entry points
(`app.py`, `api.py`, `desktop_app.py`), while the pipeline itself lives in
`src/`.  That mirrors Module 1's repository so the two modules of the platform
read the same way.
"""

import os
from collections import deque

# =============================================================================
# TIMEBASE  (frame counts vs. wall-clock durations)
# =============================================================================
# Every dwell and smoothing window in this file is authored as a DURATION and
# converted to a frame count against the SOURCE's own frame rate by frames_at().
# They used to be written directly as frame counts tuned on 30 fps footage,
# which silently changed meaning on any other rate — and this project's own
# squat clips are 15 fps.
#
# Measured consequence on a 15 fps clip (job 75493c29e9d3, 2026-08-03):
# REP_ANGLE_SMOOTH_N of 7 spans 0.47 s there, longer than the 0.33 s descent it
# is meant to smooth.  The smoothed knee angle bottomed out at 103.4 deg where
# the raw signal reached 84.4 deg, and the BOTTOM phase label — which gates both
# the depth zone and the valgus cue — landed entirely on the ASCENT, on frames
# where the knee had already re-extended to 151-179 deg.  Both fault channels
# were therefore evaluated while the subject was standing back up.
#
# This file already carried that warning against the curl and press dwells; it
# had never been applied to the squat's own constants.

FPS_REFERENCE = 30.0    # rate every legacy frame-count constant was tuned at


def frames_at(seconds: float, fps: float | None, minimum: int = 1) -> int:
    """
    A duration in seconds as a frame count at `fps`, never below `minimum`.

    An fps of None or 0 (an unreadable source) falls back to FPS_REFERENCE
    rather than dividing by zero, which makes the constant exactly its
    historical value instead of raising in the middle of a run.
    """
    rate = fps if fps and fps > 0 else FPS_REFERENCE
    return max(minimum, int(round(seconds * rate)))

# =============================================================================
# DIRECTORIES
# =============================================================================
# Every path below is ABSOLUTE, derived from this file's own location.  Paths
# used to be relative to the working directory, which was fine while the only
# entry point was launched from the repository root, but a web server, a script
# under scripts/, and an IDE run configuration each start somewhere different.
# Anchoring to the repository root makes all of them resolve the same files.

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
ASSETS_DIR   = os.path.join(PROJECT_ROOT, "assets")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")     # pose model weights
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")    # generated CSVs / results
UPLOADS_DIR  = os.path.join(PROJECT_ROOT, "uploads")    # videos posted to the API
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")   # the web UI
DOCS_DIR     = os.path.join(PROJECT_ROOT, "docs")

for _d in (MODELS_DIR, OUTPUTS_DIR, UPLOADS_DIR):
    os.makedirs(_d, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================

MODEL_PATH    = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
MODEL_URL     = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
SAMPLE_VIDEO  = os.path.join(ASSETS_DIR, "subject_005_bicep_curl_good_front.mp4")
LANDMARKS_CSV = os.path.join(OUTPUTS_DIR, "pose_landmarks.csv")
SQUAT_CSV     = os.path.join(OUTPUTS_DIR, "squat_features.csv")
# ASCII ONLY, and no longer squat-specific: this string is both the OpenCV
# window title and a cv2 caption, and the Hershey fonts render any non-ASCII
# character as mojibake (the old em-dash showed up as 'a€"').
WINDOW_TITLE  = "MediaPipe Pose - Exercise Form Analysis - FYP Demo"

# =============================================================================
# DISPLAY
# =============================================================================

# Height in pixels of the info bar rendered BELOW the video frame.
# Increased from 150 → 180 to accommodate the rep/phase row.
BAR_H = 180

# =============================================================================
# CALIBRATION
# =============================================================================

# Number of pose-detected frames collected while the user stands still before
# the standing baseline (trunk lean, hip height) is locked in.
BASELINE_FRAMES = 30

# Ceiling on pose-detected frames spent trying to obtain a CLEAN calibration
# buffer before falling back to the best buffer seen (flagged valid=False).  An
# exercise whose calibrate() rejects a contaminated buffer causes
# SessionController to discard it and resample; without a ceiling a user who
# never holds still would never finish calibrating and the session would deadlock.
#
# 90 frames = three BASELINE_FRAMES buffers, i.e. two honest retries after the
# first attempt.  A live user prompted to hold still converges in roughly one
# buffer, so two retries is generous for the case this mechanism exists to serve.
# Beyond that the retries stop being informative: nine restarts is not a user
# correcting themselves, it is a source that contains no clean window at all, and
# no ceiling makes a window appear that is not there.  It was 300, which on
# pre-recorded clips spent ~65% of the footage calibrating before failing anyway.
CALIB_MAX_FRAMES = 90

# =============================================================================
# FORM-RISK WARNING THRESHOLDS
# =============================================================================
# These are PROJECT-DEFINED HEURISTIC thresholds for real-time squat-form
# feedback.  They are NOT medical or clinical injury thresholds.
# Adjust these values to suit your subject or camera angle.
#
# All thresholds below were updated on 2026-05-19 using data derived from
# REHAB24-6 Exercise 6 (squat), filter: exercise_id==6, mocap_erroneous==0,
# correctness==1.  Source: derive_thresholds.py / derived_thresholds.json.
# Yellow = p95 of correct-rep frames; Red = p99 of correct-rep frames.
# (For right-leg valgus: yellow = p5, red = p1 — low tail, negative direction.)
# n = 26,323 correct-rep frames across 9 subjects, 18 video views.
# Reference: Černek et al., REHAB24-6, SISAP 2024. DOI: 10.5281/zenodo.13305826

# ── Knee valgus proxy ────────────────────────────────────────────────────────
# Value = (knee_x − ankle_x) / hip_width   (normalised, signed)
# Convention: frames are processed MIRRORED (cv2.flip before MediaPipe) so
# that the subject's LEFT side is on the image LEFT (selfie view).
# Left leg : positive offset → knee right of ankle → valgus collapse
# Right leg: negative offset → knee left  of ankle → valgus collapse
# OLD (mixed-view, unmirrored): VALGUS_L_YELLOW = 1.0316, VALGUS_L_RED = 1.6990
# OLD (mixed-view, unmirrored): VALGUS_R_YELLOW = -6.6667, VALGUS_R_RED = -12.6649
# Derivation: p95/p99 (L) and p5/p1 (R) of correct-rep front-view frames
# after mirror-convention fix (n=7,292). Source: derive_thresholds.py 2026-05-22.
VALGUS_L_YELLOW =  0.4007
VALGUS_L_RED    =  0.4657
# Low-tail percentiles: bad values are MORE negative for right leg.
VALGUS_R_YELLOW = -0.2747
VALGUS_R_RED    = -0.3255

# ── Trunk lean deviation from standing baseline (degrees) ────────────────────
# OLD (mixed-view): TRUNK_DEV_YELLOW = 31.169, TRUNK_DEV_RED = 39.2431
# Derivation: p95 and p99 of |trunk_lean_dev_deg| on correct-rep front-view
# frames (n=7,292). Front camera measures lateral tilt. Source: 2026-05-22.
TRUNK_DEV_YELLOW = 5.448
TRUNK_DEV_RED    = 8.1405

# ── Normalised hip depth (hip_height / avg_femur) ────────────────────────────
# Value = (ankle_mid_y − hip_mid_y) / avg_femur_px.
# Scale: ~2.0 when standing, decreasing as the person squats down. Higher = shallower.
# Depth is only evaluated at BOTTOM phase (avg knee angle < DEPTH_PHASE_BOTTOM_THRESHOLD).
# OLD (mixed-view): DEPTH_YELLOW = 2.2320, DEPTH_RED = 2.4439
# Derivation: p95/p99 of norm_hip_depth on BOTTOM-phase correct-rep front-view
# frames (n=403). Source: derive_thresholds.py 2026-05-22.
DEPTH_YELLOW = 2.2938   # > this → yellow
DEPTH_RED    = 2.4373   # > this → red (barely squatting at BOTTOM phase)

# Avg knee angle (deg) below which the subject is considered at the BOTTOM of a squat.
# Used for phase-aware depth evaluation — depth is only classified in this phase.
#
# NOTE the circularity this creates, which is why the shallow-rep detector below
# does NOT rely on it: a rep can only be judged "too shallow" once it is already
# deep enough to be labelled BOTTOM.  A squat that stops at 140 deg — precisely
# the fault the depth channel exists to catch — never reaches this gate, so the
# depth zone stays green for it no matter how the thresholds are set.
DEPTH_PHASE_BOTTOM_THRESHOLD = 110.0

# ── Hip depth relative to the user's OWN standing baseline ───────────────────
# DEPTH_YELLOW / DEPTH_RED above are ABSOLUTE norm_hip_depth values derived from
# REHAB24-6, whose subjects stand at ~1.92.  That scale does not transfer
# between subjects or camera distances: on job 75493c29e9d3 the subject stands
# at 1.696 and never exceeded 1.879 at any point in the clip, so the absolute
# channel could not have fired at ANY squat depth — it was a dead channel, not a
# lenient one.
#
# Dividing by that user's own calibrated standing depth removes both
# dependencies: 1.0 is standing, lower is deeper.  When a calibration carries no
# "hip_depth" baseline the classifier falls back to the absolute thresholds, so
# nothing that predates this change breaks.
#
# SUPERSEDED on 2026-08-03, retained for reference and still honoured by
# classify_depth() so nothing that calls it with a baseline breaks.  The DEPTH
# ZONE and the per-rep depth verdict no longer read either of these.
#
# Why: norm_hip_depth divides by avg_femur measured in IMAGE PIXELS, and a femur
# foreshortens as the thigh angles toward the camera — so the denominator moves
# with the very pose the numerator is measuring.  The result is not monotonic in
# squat depth.  Measured on job 75493c29e9d3, mean ratio by knee angle:
#
#     knee 120 deg -> 0.997     knee 140 deg -> 0.954     knee 160 deg -> 0.998
#     knee  90 deg -> 0.879     knee  80 deg -> 0.793     knee  70 deg -> 0.793
#
# A metric that reads the same at 120 and 160 degrees cannot grade depth at all,
# in either direction, at any threshold.  norm_depth_ratio is still computed and
# logged to CSV as a diagnostic; it just no longer drives a decision.
DEPTH_RATIO_YELLOW = 0.86   # SUPERSEDED - no longer read by the depth zone
DEPTH_RATIO_RED    = 0.91   # SUPERSEDED - no longer read by the depth zone

# ── Squat depth band, on KNEE ANGLE at the deepest point of the rep ──────────
# Knee angle is a pure three-point angle: scale-free, unaffected by camera
# distance, and monotonic through the descent where the hip-depth ratio above is
# not (179 deg standing -> 84 deg at the bottom on the clip cited above).  It is
# also the measure the coaching convention is stated in — ~90 deg is thighs
# parallel to the floor.
#
# The band is TWO-SIDED.  Depth used to be graded only from above ("too
# shallow"), which left going too deep — where the lumbar spine starts to flex
# under load — silently green no matter how far past parallel the lifter went.
#
# FEEL-TUNED against the coaching convention (90 deg = parallel), NOT derived
# from REHAB24-6 or any other annotated set.  They are defensible starting
# points, not measured ones, and must be re-derived before any reported result
# rests on them.
# TIERS ARE NOT COSMETIC — only RED is spoken.  The audio controller counts
# consecutive RED frames (see FeedbackController._update_corrective_dwell), so a
# yellow band shows on the overlay, the zone strip and the rep's quality but
# stays silent, which is what keeps the coaching sparse rather than chatty.
# Setting the RED edge is therefore setting "how bad before it says something".
DEPTH_KNEE_SHALLOW_YELLOW = 100.0   # min knee angle above this → short of depth
DEPTH_KNEE_SHALLOW_RED    = 115.0   # a half squat — spoken as "go deeper"
DEPTH_KNEE_DEEP_YELLOW    =  70.0   # below this → well past parallel
DEPTH_KNEE_DEEP_RED       =  55.0   # deep enough that lumbar flexion is likely

# COVERAGE GAP, stated so it is not mistaken for a bug: a descent that stops
# between KNEE_SHALLOW_ATTEMPT (145) and KNEE_DESCENDING (150) is neither a
# counted rep nor a shallow rep, and passes in silence.  That window is the
# price of rejecting postural noise — on job 75493c29e9d3 two dips at 147.1 and
# 147.3 deg, lasting 3 and 2 frames, are the subject shifting weight, not
# squatting.  Lower KNEE_SHALLOW_ATTEMPT to close the gap and those start being
# reported as failed reps.

# ── Left-right asymmetry (max of knee and hip symmetry differences) ──────────
# asymmetry_score = max(|knee_sym_diff_deg|, |hip_sym_diff_deg|)
# OLD (feel-tuned): ASYMMETRY_YELLOW =  5.0
# OLD (feel-tuned): ASYMMETRY_RED    = 10.0
# Derivation: p95 and p99 of asymmetry_score on correct-rep frames (n=26,323).
ASYMMETRY_YELLOW = 48.7847   # score > this → yellow
ASYMMETRY_RED    = 65.8317   # score > this → red

# ── 3-D knee valgus (knee_deviation_3d) ──────────────────────────────────────
# Value = signed deviation of the knee from the hip-ankle axis in 3-D space,
# normalised by hip-to-ankle leg length.  Uses MediaPipe's estimated z-depth.
# Absolute value is used so the same thresholds apply to both legs.
# OLD (feel-tuned): VALGUS_3D_YELLOW = 0.07
# OLD (feel-tuned): VALGUS_3D_RED    = 0.13
# Derivation: p95 and p99 of |knee_valgus_3d_l| on correct-rep frames (n=26,323).
# Left-leg values used (more conservative); right-leg p95=1.1757, p99=1.7580.
VALGUS_3D_YELLOW = 1.0198   # |dev| > this → yellow
VALGUS_3D_RED    = 1.4396   # |dev| > this → red

# =============================================================================
# SMOOTHING
# =============================================================================

# Time over which noisy metrics are averaged before zone classification.
# Prevents single-frame pose jitter from flipping zone colours.  Authored as a
# duration for the reason given under TIMEBASE: at 15 fps the historical 5-frame
# window spans 0.33 s, and on job 75493c29e9d3 that was enough to average the
# trunk-deviation peak of 5.99 deg back under TRUNK_DEV_YELLOW (5.448), leaving
# the channel green on a frame that had genuinely crossed the threshold.
ZONE_SMOOTH_SEC = 0.167

# Historical 30-fps frame count (see the note on REP_ANGLE_SMOOTH_N).
SMOOTH_N = frames_at(ZONE_SMOOTH_SEC, FPS_REFERENCE)

# Human-readable label for each overall zone shown in the info bar
OVERALL_LABEL = {"green": "GOOD", "yellow": "WARNING", "red": "HIGH RISK"}

# =============================================================================
# REP COUNTER & PHASE DETECTION
# =============================================================================
# Thresholds for the state-machine in rep_counter.py.
# All values are in degrees (average of left + right knee angle).

# Knee angle (degrees) above which we consider the user fully standing.
# Must be meaningfully above KNEE_DESCENDING to create hysteresis and
# prevent landmark-jitter from flipping the state machine.
KNEE_STANDING = 165.0

# Dropping below this triggers the DESCENDING phase.
# The 15° gap between KNEE_STANDING and KNEE_DESCENDING is intentional:
# it prevents noisy landmarks while standing still from bouncing the state.
KNEE_DESCENDING = 150.0

# Smoothed knee angle (degrees) that must be reached during DESCENDING for
# the rep to count as a valid squat depth.
# Changed 120° → 130° on 2026-05-19: with 120° the BOTTOM phase rarely triggered
# in REHAB24-6 (only 1,789 of 26,323 rep frames classified BOTTOM), causing the
# Hip Depth zone to evaluate on too few frames and recall to drop to 0.18.
# 130° is still a genuine squat; the 7-frame smoother means raw knee angle at
# the bottom is typically 120–125°, so this threshold is reached in real reps.
# Re-run rehab24_runner.py → derive_thresholds.py → evaluate_thresholds.py
# after this change to see the updated Hip Depth evaluation numbers.
KNEE_BOTTOM = 130.0

# Angular speed (degrees per SECOND, signed) below which motion counts as
# "stopped".  Used to detect the inflection point between descent and ascent.
#
# Was 0.5 degrees per FRAME — which is 15 deg/s at 30 fps but only 7.5 deg/s at
# 15 fps, making the turnaround detector twice as twitchy on this project's own
# clips.  Per second it means the same thing at every rate.
REP_VELOCITY_DEAD_ZONE_DEG_PER_SEC = 15.0

# Smoothing window for the knee-angle signal.  0.233 s is the historical 7
# frames at 30 fps, unchanged for 30 fps sources.
REP_ANGLE_SMOOTH_SEC = 0.233

# Smoothing window for the velocity signal (historical 5 frames at 30 fps).
REP_VELOCITY_SMOOTH_SEC = 0.167

# Minimum time a phase must be held before a transition can fire.
# Guards against jitter-driven rapid phase flipping.  (3 frames at 30 fps.)
REP_PHASE_MIN_SEC = 0.10

# Minimum time the BOTTOM phase must be held before transitioning to ASCENDING —
# the minimum realistic dwell at the bottom of a real squat.  Prevents momentary
# dips counting as reps.  (12 frames at 30 fps.)
BOTTOM_MIN_DWELL_SEC = 0.40

# Historical 30-fps frame counts, retained because they are the documented
# reference values and are imported by scripts outside src/.  The live pipeline
# derives its own counts from the SOURCE fps via frames_at(); these are what
# that derivation returns for a 30 fps source.
REP_VELOCITY_DEAD_ZONE  = REP_VELOCITY_DEAD_ZONE_DEG_PER_SEC / FPS_REFERENCE
REP_ANGLE_SMOOTH_N      = frames_at(REP_ANGLE_SMOOTH_SEC,    FPS_REFERENCE)
REP_VELOCITY_SMOOTH_N   = frames_at(REP_VELOCITY_SMOOTH_SEC, FPS_REFERENCE)
REP_PHASE_MIN_FRAMES    = frames_at(REP_PHASE_MIN_SEC,       FPS_REFERENCE)
BOTTOM_MIN_DWELL_FRAMES = frames_at(BOTTOM_MIN_DWELL_SEC,    FPS_REFERENCE)

# ── Shallow ("partial") reps ─────────────────────────────────────────────────
# A descent that never reached KNEE_BOTTOM used to be DISCARDED outright by
# _state_descending's abort branch: no count, no summary, no cue, no trace in
# any artefact.  That made the most common squat fault structurally invisible —
# a partial rep was not a bad rep, it ceased to exist — and because the depth
# zone is only scored during BOTTOM phase, which a discarded descent never
# enters, nothing downstream could catch it either.
#
# Measured on job 75493c29e9d3: four of six squat attempts were discarded this
# way (min knee 138-147 deg, hips only 6-11% below standing) and the session
# reported "2 reps, both green".
#
# A descent now counts as a SHALLOW REP — logged, quality red, and cued — when
# it reaches at least KNEE_SHALLOW_ATTEMPT and is held for
# SHALLOW_MIN_DESCENT_SEC.  Both guards exist to separate a genuine partial
# squat from postural noise: on that same clip two further dips (147.1 and
# 147.3 deg, lasting 3 and 2 frames) are the subject shifting weight.
KNEE_SHALLOW_ATTEMPT    = 145.0
SHALLOW_MIN_DESCENT_SEC = 0.25

# How long the shallow verdict is held as a synthetic red depth zone so the
# dwell-based audio controller can observe it.  MUST exceed CORRECTIVE_DWELL_SEC
# or the "go deeper" cue can never fire — the same edge-triggered-verdict versus
# level-triggered-controller problem CURL_ROM_CUE_HOLD_FRAMES solves for the
# curl, applied to the squat for the first time here.
SHALLOW_DEPTH_CUE_HOLD_SEC = 0.60

# Minimum MediaPipe visibility score (0–1) for a landmark to be considered
# reliable.  Frames where any of the four key rep-counter landmarks (both hips
# and knees) fall below this threshold are skipped entirely by the state machine.
LANDMARK_MIN_VISIBILITY = 0.6

# Set to True to print a one-line log on every state-machine phase transition.
# Useful for diagnosing false reps; set to False for batch / production runs.
DEBUG_REP_COUNTER = False

# Per-rep summary CSV output path and header
REP_SUMMARY_CSV    = os.path.join(OUTPUTS_DIR, "rep_summary.csv")
REP_SUMMARY_HEADER = [
    "rep_num", "start_frame", "end_frame", "duration_frames",
    # Knee angle (deepest point + average during the rep)
    "min_knee_angle", "mean_knee_angle",
    # Trunk deviation (peak + average absolute deviation)
    "max_trunk_dev", "mean_trunk_dev",
    # Valgus — signed numeric offsets (peak = worst-case, mean = average)
    "peak_valgus_l", "mean_valgus_l",
    "peak_valgus_r", "mean_valgus_r",
    # Hip depth ratio average during the rep
    "mean_depth",
    # Worst zone reached for each metric during the rep
    "worst_valgus_l", "worst_valgus_r", "worst_trunk", "worst_depth",
    # 1 when the descent never reached KNEE_BOTTOM — a partial rep, which is
    # logged rather than discarded (see KNEE_SHALLOW_ATTEMPT).
    "shallow",
    "rep_quality",
]

# BGR colours used to draw the phase indicator on the video frame and info bar
PHASE_COLORS = {
    "STANDING":   (100, 210,  90),   # green
    "DESCENDING": (  0, 165, 255),   # orange
    "BOTTOM":     ( 80,  80, 255),   # red
    "ASCENDING":  (255, 200,  50),   # cyan-blue
}

# =============================================================================
# MEDIAPIPE LANDMARK INDICES
# =============================================================================
# Google's pose model assigns a fixed integer index to each of the 33 landmarks.
# These named constants make the code self-documenting.

IDX_SHOULDER_L, IDX_SHOULDER_R = 11, 12
IDX_ELBOW_L,    IDX_ELBOW_R    = 13, 14
# Wrists complete the shoulder-elbow-wrist chain needed for upper-limb joint
# angles (bicep curl, shoulder press).  Declared here for completeness; no
# current code path uses them, since Squat is the only implemented exercise.
IDX_WRIST_L,    IDX_WRIST_R    = 15, 16
IDX_HIP_L,      IDX_HIP_R      = 23, 24
IDX_KNEE_L,     IDX_KNEE_R     = 25, 26
IDX_ANKLE_L,    IDX_ANKLE_R    = 27, 28

# =============================================================================
# SKELETON CONNECTIONS
# =============================================================================
# Each tuple (a, b) draws a line between landmark a and landmark b.

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),    # face
    (9,10),                                               # mouth
    (11,12),                                              # shoulders
    (11,13),(13,15),(15,17),(17,19),(15,19),(15,21),     # left arm + hand
    (12,14),(14,16),(16,18),(18,20),(16,20),(16,22),     # right arm + hand
    (11,23),(12,24),(23,24),                              # torso
    (23,25),(25,27),(27,29),(29,31),(27,31),             # left leg + foot
    (24,26),(26,28),(28,30),(30,32),(28,32),             # right leg + foot
]

# Key joints drawn with a larger highlighted dot + text label on the video
KEY_JOINTS = {
    "L.Shldr": IDX_SHOULDER_L, "R.Shldr": IDX_SHOULDER_R,
    "L.Elbow": IDX_ELBOW_L,    "R.Elbow": IDX_ELBOW_R,
    "L.Hip":   IDX_HIP_L,      "R.Hip":   IDX_HIP_R,
    "L.Knee":  IDX_KNEE_L,     "R.Knee":  IDX_KNEE_R,
    "L.Ankle": IDX_ANKLE_L,    "R.Ankle": IDX_ANKLE_R,
}

# =============================================================================
# CSV SCHEMA
# =============================================================================

SQUAT_CSV_HEADER = [
    "frame",
    # ── Body segment lengths (pixels) ────────────────────────────────────────
    "shoulder_width_px", "hip_width_px", "torso_len_px",
    "femur_l_px", "femur_r_px", "tibia_l_px", "tibia_r_px",
    "avg_femur_px", "avg_tibia_px",
    # ── Stance ───────────────────────────────────────────────────────────────
    "stance_width_px", "norm_stance_width",
    # ── Knee medial displacement (valgus proxy) ───────────────────────────────
    "knee_offset_l_px", "knee_offset_r_px",
    "norm_knee_offset_l", "norm_knee_offset_r",
    # ── Hip height & depth ───────────────────────────────────────────────────
    # norm_depth_ratio is norm_hip_depth over the user's own calibrated standing
    # depth (1.0 = standing, lower = deeper); the baseline is logged alongside
    # it so the division is auditable the way baseline_trunk_lean_deg is.
    "hip_height_px", "norm_hip_depth", "baseline_hip_depth", "norm_depth_ratio",
    # ── Trunk lean ───────────────────────────────────────────────────────────
    "trunk_lean_deg", "baseline_trunk_lean_deg", "trunk_lean_dev_deg",
    # ── Knee angles ──────────────────────────────────────────────────────────
    "knee_angle_l_deg", "knee_angle_r_deg", "knee_sym_diff_deg",
    # ── Hip angles ───────────────────────────────────────────────────────────
    "hip_angle_l_deg", "hip_angle_r_deg", "hip_sym_diff_deg",
    # ── Form-risk warning zones (project heuristics, not medical thresholds) ─
    # depth_zone is the TOO-SHALLOW side of the depth band; depth_excess_zone is
    # the TOO-DEEP side.  Both are logged so a run can be audited for either.
    "valgus_zone_l", "valgus_zone_r", "trunk_zone",
    "depth_zone", "depth_excess_zone", "overall_zone",
    # ── Rep tracking ─────────────────────────────────────────────────────────
    "rep_phase", "rep_count",
    # ── 3-D metrics (MediaPipe z-depth estimate) ──────────────────────────────
    "knee_angle_3d_l_deg", "knee_angle_3d_r_deg", "knee_3d_sym_diff_deg",
    "knee_valgus_3d_l", "knee_valgus_3d_r",
    "knee_fwd_l", "knee_fwd_r",
    "valgus_3d_zone_l", "valgus_3d_zone_r",
    # ── Asymmetry & simple phase label ────────────────────────────────────────
    "asymmetry_score", "asymmetry_zone", "phase_label",
]

# =============================================================================
# AUDIO FEEDBACK  (Module 2 — audio coaching layer; LIVE mode only)
# =============================================================================
# Pre-rendered short MP3 clips are played back (non-blocking) when a trigger
# fires.  There are THREE categories with a STRICT master priority so that two
# clips never overlap (highest → lowest):
#
#     SYSTEM STATUS  >  CORRECTIVE CUES  >  ENCOURAGEMENT
#
# The audio layer is a read-only CONSUMER of existing zone / rep / visibility
# state — it never modifies zone classification or the rep counter.
#
# Design intent: sparse and well-timed, NOT chatty.  Every category has a
# sustained-trigger requirement (the condition must hold before firing) and a
# cooldown (the same cue cannot repeat more often than every N seconds).
#
# UNIT CONVENTION (read before tuning):
#   • Cooldowns        → wall-clock SECONDS  (time.monotonic; frame-rate safe)
#   • Dwell windows    → FRAME COUNTS        (how long a pose condition holds)
#   • System hysteresis dwell → SECONDS (it measures real-time detection
#                                          reliability, not pose-stream frames)

# Master switch.  Audio runs ONLY inside the live test_mediapipe_pose.py loop;
# NEVER enable this in any batch / dataset script (rehab24_runner.py, etc.).
AUDIO_FEEDBACK_ENABLED = True

# Directory holding the 14 pre-rendered clips.
AUDIO_CLIPS_DIR = os.path.join(ASSETS_DIR, "audio")

# Global minimum quiet gap (seconds) between ANY two spoken cues, across all
# categories.  Enforces the "sparse, never stacked" principle: when several
# faults are sustained at once, the master priority picks one cue now and the
# next-priority cue waits at least this long instead of firing on the very next
# frame (which would cut the first clip off mid-word).  SYSTEM STATUS bypasses
# this gap so a safety cue ("I can't see you") can always preempt; its own
# 2.5 s bad-dwell already prevents it from machine-gunning.
AUDIO_MIN_CUE_GAP_SEC = 2.0

# ── Corrective cues ──────────────────────────────────────────────────────────
# A zone must read RED CONTINUOUSLY for this long before its cue fires.
# Sustained-trigger: a single red frame (pose jitter) is ignored.
#
# Authored as a duration for the reason given under TIMEBASE.  As a flat 12
# frames it demanded 0.4 s of unbroken red at 30 fps but 0.8 s at 15 fps — and
# on a 15 fps clip a single rep's DESCENDING+BOTTOM window is only ~30 frames,
# so the cue needed the fault to persist across most of the rep to be heard.
CORRECTIVE_DWELL_SEC = 0.40

# Historical 30-fps frame count (see the note on REP_ANGLE_SMOOTH_N).
CORRECTIVE_DWELL_FRAMES = frames_at(CORRECTIVE_DWELL_SEC, FPS_REFERENCE)
# The SAME corrective cue cannot repeat more often than this many SECONDS.
CORRECTIVE_COOLDOWN_SEC = 4.0
# Order used when several zones are red on the same frame — only the single
# highest-priority cue plays.  valgus first: it is the validated / most
# discriminative channel (see zones.py overall-zone notes); the two depth
# channels next; trunk last.
#
# "depth" (too shallow) and "depth_excess" (too deep) can never both be red on
# the same frame — an angle cannot sit on both sides of the band — so their
# relative order here is nominal, not a real contest.
AUDIO_CORRECTIVE_PRIORITY = ["valgus", "depth", "depth_excess", "trunk"]

# ── System status (hysteresis — two dwell thresholds so it cannot oscillate) ─
# Detection must be BAD continuously for this long before an alert is announced
# (prevents a brief one-frame dropout from triggering).
SYSTEM_STATUS_BAD_DWELL_SEC  = 2.5
# Detection must be GOOD continuously for this long before the alert state
# clears (second, shorter threshold so the alert does not flap on/off).
SYSTEM_STATUS_GOOD_DWELL_SEC = 1.5
# While the alert is active, re-announce no more often than this.  Long, so it
# does not nag if the user genuinely cannot get into frame.
SYSTEM_STATUS_COOLDOWN_SEC   = 12.0

# ── Encouragement (lowest priority; NEVER praises red form) ──────────────────
# Milestone: praise every N completed reps, but only if recent reps were
# acceptable quality (no red).  Edge-triggered on the rep that crosses N.
ENCOURAGE_MILESTONE_INTERVAL = 8
# Clean streak: praise after N consecutive reps with no red zone.
ENCOURAGE_STREAK_LENGTH = 5
# Encouragement (either trigger) cannot repeat more often than this.
ENCOURAGE_COOLDOWN_SEC = 10.0

# ── Clip filenames ───────────────────────────────────────────────────────────
# System status.  The "no pose detected at all" case has no landmarks to tell
# "too far" from "too close", so it STRICTLY ALTERNATES between these two clips
# on successive trigger events.
AUDIO_SYSTEM_NO_POSE_CLIPS = ["system_move_into_frame.mp3", "system_step_back.mp3"]
# Pose present but key lower-body landmarks low-confidence (e.g. turned sideways).
AUDIO_SYSTEM_FACE_CAMERA   = "system_face_camera.mp3"

# Corrective cue clips, keyed by zone.
AUDIO_CORRECT_KNEES_OUT = "correct_knees_out.mp3"   # valgus red
AUDIO_CORRECT_GO_DEEPER = "correct_go_deeper.mp3"   # depth too SHALLOW
# Depth too DEEP — the opposite correction, which had no cue because the depth
# channel used to be graded from one side only.
#
# THIS FILE DOES NOT YET EXIST in assets/audio/.  AudioPlayer.__init__ logs a
# warning for any missing clip and play() no-ops on an uncached name, so until
# it is recorded the fault still shows on the overlay and in the zone strip and
# simply plays no sound — the same degradation the curl clips were shipped with.
AUDIO_CORRECT_NOT_SO_DEEP = "correct_not_so_deep.mp3"   # depth too DEEP
# Trunk red alternates between these two for variety (own rotation pointer).
AUDIO_CORRECT_TRUNK_CLIPS = ["correct_chest_up.mp3", "correct_stay_balanced.mp3"]
# Reserved for a future rep-tempo trigger — NOT wired yet (documented future work).
AUDIO_CORRECT_SLOW_DOWN = "correct_slow_down.mp3"

# Encouragement rotation lists.  Each trigger type cycles its own list so the
# same clip never plays twice in a row.
AUDIO_ENCOURAGE_MILESTONE_CLIPS = [
    "encourage_keep_going.mp3", "encourage_doing_well.mp3",
    "encourage_halfway.mp3",    "encourage_strong_finish.mp3",
]
# NOTE — exercise-neutrality constraint.
# exercises/base.py documents encouragement cues as "generic (shared across
# exercises)", so every clip in this pool must make sense for ANY exercise.
# "encourage_nice_depth.mp3" was removed on 2026-07-27: depth is a squat
# concept, and a bicep-curl set praised for its "nice depth" would be wrong.
# The clip file is retained in assets/audio/ and may be re-attached to squat
# specifically via a per-exercise hook on AudioCueMapping in future.
AUDIO_ENCOURAGE_STREAK_CLIPS = ["encourage_excellent_form.mp3"]

# ── Bicep curl corrective clips ──────────────────────────────────────────────
# These five clips were commissioned alongside the BicepCurlExercise build and
# may not yet exist in assets/audio/.  AudioPlayer.__init__ logs a warning for
# any missing file and AudioPlayer.play() no-ops on an uncached name, so a
# missing clip degrades to silence rather than crashing.
AUDIO_CURL_ELBOWS_PINNED   = "correct_elbows_pinned.mp3"    # elbow drift red
AUDIO_CURL_STOP_SWINGING   = "correct_stop_swinging.mp3"    # body swing red
AUDIO_CURL_MATCH_ARMS      = "correct_match_arms.mp3"       # asymmetry red
AUDIO_CURL_FULL_EXTENSION  = "correct_full_extension.mp3"   # ROM: bottom short
AUDIO_CURL_HIGHER          = "correct_higher.mp3"           # ROM: top short

AUDIO_CURL_CORRECTIVE_CLIPS = [
    AUDIO_CURL_ELBOWS_PINNED, AUDIO_CURL_STOP_SWINGING, AUDIO_CURL_MATCH_ARMS,
    AUDIO_CURL_FULL_EXTENSION, AUDIO_CURL_HIGHER,
]

# Full list of clips the loader expects to find — used for startup validation
# (a missing file logs a warning; it does not crash the program).
EXPECTED_AUDIO_CLIPS = (
    AUDIO_SYSTEM_NO_POSE_CLIPS
    + [AUDIO_SYSTEM_FACE_CAMERA,
       AUDIO_CORRECT_KNEES_OUT, AUDIO_CORRECT_GO_DEEPER,
       AUDIO_CORRECT_NOT_SO_DEEP, AUDIO_CORRECT_SLOW_DOWN]
    + AUDIO_CORRECT_TRUNK_CLIPS
    + AUDIO_ENCOURAGE_MILESTONE_CLIPS
    + AUDIO_ENCOURAGE_STREAK_CLIPS
    + AUDIO_CURL_CORRECTIVE_CLIPS
)


# =============================================================================
# BICEP CURL THRESHOLDS
# =============================================================================
# Level 1 scope: live coaching only.  EVERY value in this section is a
# feel-tuned initial value; not dataset-derived; see future work.
# None of these has been validated against annotated data the way the squat
# thresholds were (REHAB24-6 Ex6, p95/p99 of correct-rep frames).  They are
# plausible starting points for live use and MUST NOT be reported as derived.

# ── Elbow drift (upper arm not pinned to the torso) ──────────────────────────
# Value = outward-oriented (elbow_x - shoulder_x) / shoulder_width_px, MINUS the
# per-user baseline locked at calibration.  Normalised by SHOULDER width — the
# natural upper-limb anthropometric denominator — NOT hip width as the squat
# metrics use, because the relevant body scale for arm geometry is the shoulder
# girdle.  Positive = elbow flared outward FROM ITS OWN RESTING POSITION.
#
# !! THESE TWO VALUES ARE STALE AND MUST BE RE-DERIVED BEFORE USE. !!
# They were feel-tuned on 2026-07-27 against the UN-BASELINED metric, i.e. the
# raw offset including the subject's natural resting elbow position.  Elbows do
# not hang directly under the shoulders: on good/front/subject_024 the raw
# resting offset was ~0.20 of shoulder width, so the old metric sat AT
# ELBOW_DRIFT_RED while the subject stood still with arms down, and the channel
# read red for roughly half the file.  Baseline subtraction (added 2026-07-29)
# moves the at-rest reading to ~0.0, which shifts the entire distribution and
# invalidates both numbers.  They are left unchanged here only so the pipeline
# runs; they carry NO evidential weight until re-derived from data.
ELBOW_DRIFT_YELLOW = 0.10   # STALE - predates baseline subtraction; MUST be re-derived
ELBOW_DRIFT_RED    = 0.20   # STALE - predates baseline subtraction; MUST be re-derived

# ── Body swing (torso english used to throw the weight up) ───────────────────
# Value = abs(trunk_lean - baseline_trunk_lean) in degrees.
BODY_SWING_YELLOW = 5.0     # feel-tuned initial value; not dataset-derived; see future work
BODY_SWING_RED    = 10.0    # feel-tuned initial value; not dataset-derived; see future work

# ── Left/right asymmetry — SUPERSEDED, retained for reference ────────────────
# Value = abs(elbow_angle_l - elbow_angle_r) on a SINGLE FRAME.
# SUPERSEDED on 2026-07-29 and no longer read by BicepCurlExercise.  An
# instantaneous left-vs-right angle difference is meaningless for an ALTERNATING
# curl, where one arm is flexed precisely while the other is extended: on
# good/front/subject_024 this difference reached 160-168 deg mid-rep purely by
# design, pinning the channel red for the whole file and making good and bad
# clips indistinguishable.  Asymmetry is now a PER-REP comparison — see
# CURL_ARM_PEAK_DIFF_* / CURL_ARM_ROM_DIFF_* below.  elbow_symmetry_diff_deg is
# still logged to CSV as a diagnostic; it just no longer drives a zone.
CURL_ASYMMETRY_YELLOW = 10.0  # SUPERSEDED - no longer read
CURL_ASYMMETRY_RED    = 20.0  # SUPERSEDED - no longer read

# ── Left/right asymmetry — per-rep, the real "one arm is doing less work" ────
# Compared between each arm's MOST RECENT COMPLETED rep, so the measure is
# meaningful in both simultaneous and alternating modes.  Green until BOTH arms
# have completed at least one rep (nothing to compare before that).
#   peak-flexion gap = abs(min_elbow_L - min_elbow_R)  — how high each arm curled
#   ROM gap          = abs(rep_range_L - rep_range_R)  — how far each arm travelled
# The zone is the WORSE of the two bands.
CURL_ARM_PEAK_DIFF_YELLOW = 15.0  # feel-tuned; not dataset-derived; must be re-derived
CURL_ARM_PEAK_DIFF_RED    = 30.0  # feel-tuned; not dataset-derived; must be re-derived
CURL_ARM_ROM_DIFF_YELLOW  = 20.0  # feel-tuned; not dataset-derived; must be re-derived
CURL_ARM_ROM_DIFF_RED     = 40.0  # feel-tuned; not dataset-derived; must be re-derived

# ── Range of motion (per-rep verdict, NOT a live zone) ───────────────────────
# A rep whose elbow sweep (max - min) is below this is judged incomplete.
ROM_MIN_ACCEPTABLE = 100.0  # feel-tuned initial value; not dataset-derived; see future work

# ── Rep state machine (degrees, mean of left and right elbow angle) ──────────
# Arm extended at the bottom of the curl -> ~180 deg; fully contracted -> ~40.
# The driver therefore DECREASES from rest, mirroring the squat knee angle.
CURL_EXTENDED   = 155.0   # at/above this the arm counts as extended (rep boundary)
CURL_CURLING    = 140.0   # dropping below this with negative velocity starts the lift
CURL_CONTRACTED = 70.0    # must be reached for the rep to count as a real curl
CURL_LOWERING   = 85.0    # rising back above this begins the eccentric phase

# Minimum frames held at the top before the rep may progress — rejects bounce
# reps thrown up and dropped immediately.
# FRAME-RATE WARNING: this dwell is counted in FRAMES, not seconds, so its
# wall-clock strictness varies with the source frame rate.  8 frames is
# ~265 ms at 30 fps but ~533 ms at 15 fps — and 15 fps is the rate of the
# project's bicep_curl dataset clips.  A slower source therefore silently
# demands a proportionally LONGER real-time hold at the top of the curl.
# The same caveat applies to every *_FRAMES constant in this section.
CURL_CONTRACTED_MIN_DWELL_FRAMES = 8   # feel-tuned initial value; not dataset-derived; see future work

# Calibration guard: the mean elbow angle must stay inside this band for every
# buffered frame, otherwise the user was not in the required arms-down pose and
# the buffer is discarded and restarted.
CURL_CALIB_ELBOW_MIN = 155.0
# UPPER BOUND CANNOT BIND.  angle_at_joint returns degrees(acos(...)), which is
# mathematically bounded to [0, 180], so no observed mean elbow angle can ever
# exceed this and the comparison is always true.  It was 190.0, which read as
# though it guarded against hyperextension; it never did.  Set to 180.0 so the
# constant states the real domain limit rather than implying a live guard.
# Every rejection therefore comes from the LOWER bound: an arm too flexed to be
# a valid arms-down reference.
CURL_CALIB_ELBOW_MAX = 180.0

# How long a bad-ROM verdict is held as a synthetic red zone after a rep
# completes, so the dwell-based audio controller can pick it up.  Must exceed
# CORRECTIVE_DWELL_FRAMES (12) for the cue to ever fire.
CURL_ROM_CUE_HOLD_FRAMES = 15

# Frames the state machine may sit in LOWERING before the descent is judged
# STALLED — the user is hovering mid-range instead of straightening the arm.
# This is the ONLY route by which an incomplete-extension fault can be cued at
# all: a user who never re-extends never completes a rep, so the completion-time
# ROM verdict never runs for them and the failure mode would otherwise be pure
# silence.  Unlike the completion-time cue this one is level-triggered — the
# zone stays red while the stall lasts, which satisfies CORRECTIVE_DWELL_FRAMES
# naturally, and the controller's per-channel cooldown stops it repeating.
# 45 frames ~ 1.5 s at 30 fps (~3.0 s at 15 fps — see the frame-rate warning on
# CURL_CONTRACTED_MIN_DWELL_FRAMES).
CURL_LOWERING_STALL_FRAMES = 45  # feel-tuned initial value; not dataset-derived; see future work

# ── Bicep curl CSV outputs (must not collide with the squat CSVs) ────────────
CURL_CSV        = os.path.join(OUTPUTS_DIR, "bicep_curl_features.csv")
CURL_REP_SUMMARY_CSV = os.path.join(OUTPUTS_DIR, "bicep_curl_rep_summary.csv")

CURL_CSV_HEADER = [
    "frame",
    "shoulder_width_px", "upper_arm_l_px", "upper_arm_r_px",
    "forearm_l_px", "forearm_r_px",
    "elbow_angle_l_deg", "elbow_angle_r_deg", "elbow_mean_deg",
    # Retained as a DIAGNOSTIC only — no longer drives the asymmetry zone.
    "elbow_symmetry_diff_deg",
    # Baseline-subtracted drift, plus the locked baselines themselves so the
    # subtraction is auditable from the log the way baseline_trunk_lean_deg is.
    "elbow_drift_l", "elbow_drift_r",
    "baseline_elbow_drift_l", "baseline_elbow_drift_r",
    "trunk_lean_deg", "baseline_trunk_lean_deg", "trunk_deviation_deg",
    "zone_drift_l", "zone_drift_r", "zone_body_swing", "zone_asymmetry",
    "zone_overall",
    # phase is the representative (most active) arm — what gates the audio cues.
    # phase_l / phase_r expose each independent per-arm machine.
    "phase", "phase_l", "phase_r",
    # rep_count is the TOTAL across both arms (what a person counts aloud).
    "rep_count", "rep_count_l", "rep_count_r",
]

# One row PER ARM, written when that arm completes a rep.  Every field refers to
# the arm named in `arm`.  rep_number is the global completion order across both
# arms; arm_rep_number is that arm's own tally.
CURL_REP_SUMMARY_HEADER = [
    "rep_number", "arm", "arm_rep_number",
    "start_frame", "end_frame", "duration_frames",
    "min_elbow", "max_elbow", "rep_range",
    "peak_drift", "peak_trunk_deviation",
    "rom_incomplete", "rep_quality",
]

# Phase colours for the curl overlay (BGR), mirroring PHASE_COLORS for squat.
CURL_PHASE_COLORS = {
    "EXTENDED":   (100, 210,  90),   # green
    "CURLING":    (  0, 165, 255),   # orange
    "CONTRACTED": ( 80,  80, 255),   # red
    "LOWERING":   (255, 200,  50),   # cyan-blue
}


# =============================================================================
# VIEW DETECTION  (Stage 1 — DETECTION ONLY, pure observer)
# =============================================================================
# Consumed only by view_detection.py.  NOTHING in session.py, any exercise, the
# visibility gate or the audio layer reads these: Stage 1 detects and logs, it
# does not act.  See the module docstring for the geometric basis.
#
# The classifier maps each signal onto a "profileness" in [0, 1] by linear
# interpolation between a FRONT anchor and a SIDE anchor, then takes a weighted
# mean.  Anchors are (front_value, side_value) and may run in either direction:
# hip_shoulder_ratio FALLS toward profile, vis_asym RISES.

# Frames in the rolling smoothing window.  At 15 fps (this dataset's rate) 15
# frames is ~1.0 s; at 30 fps live it is ~0.5 s.  Long enough to ride out a
# single bad pose, short enough to follow a subject who genuinely turns.
VIEW_SMOOTH_FRAMES = 15

# Confidence below which the label is reported as "unknown" rather than guessed.
# Deliberately not low: an ambiguous view should decline to answer, because the
# whole point of the feature is to avoid running frontal-plane metrics on a
# projection that cannot support them.
VIEW_MIN_CONFIDENCE = 0.45

# Score band edges.  score <= FRONT_MAX -> front; >= SIDE_MIN -> side; between
# -> diagonal.
VIEW_FRONT_MAX_SCORE = 0.34
VIEW_SIDE_MIN_SCORE  = 0.66

# (front_anchor, side_anchor) per signal, set to the per-signal MEDIAN of true
# front and true side frames on a SUBJECT-DISJOINT calibration split (subjects
# 001-008, 96 clips) of the Multi-View dataset, measured 2026-07-29.  All
# reported validation figures come from the HELD-OUT subjects (009+), which
# share no subject with this calibration set.
VIEW_ANCHORS = {
    "shoulder_z_ratio":     (0.16, 17.08),
    "nose_shoulder_ratio":  (0.06, 3.38),
    "shoulder_torso_ratio": (0.73, 0.10),
    "vis_asym":             (0.00, 0.41),
}

# Interpolated in LOG space rather than linear — see _profileness().  Both span
# >2 orders of magnitude, and on a linear scale the profile tail swamps the range
# so badly that a true diagonal is indistinguishable from front.
VIEW_LOG_SIGNALS = {"shoulder_z_ratio", "nose_shoulder_ratio"}

# Relative weight in the combined score, set from measured separation on the
# calibration split.  The two log signals carry more weight because they place
# the diagonal median near 0.5 profileness, which is what makes a three-way
# split possible; the linear pair place it nearer 0.25 and act as support.
#
# TWO SIGNALS FROM THE ORIGINAL SPEC ARE DELIBERATELY ABSENT — both are still
# computed and reported by compute_view_signals(), they simply earn no weight:
#   * hip_shoulder_ratio — the expected "hip width collapses in profile" effect
#     did NOT appear.  Calibration medians ran 0.540 front / 0.571 diagonal /
#     0.672 side: a weak move in the OPPOSITE direction to the hypothesis, with
#     front and side bodies overlapping.  MediaPipe infers occluded hips rather
#     than collapsing them, so the projection argument does not survive contact
#     with the pose model.
#   * ear_vis_asym — identically 0.000 at every view; the model reports both
#     ears with equal confidence even in full profile.  No information at all.
VIEW_SIGNAL_WEIGHTS = {
    "shoulder_z_ratio":     2.0,
    "nose_shoulder_ratio":  1.5,
    "shoulder_torso_ratio": 1.0,
    "vis_asym":             1.0,
}


# =============================================================================
# SHOULDER PRESS THRESHOLDS
# =============================================================================
# Level 1 scope: live coaching only.  EVERY value in this section is a
# feel-tuned initial value; not dataset-derived; see future work.
#
# Movement confirmed by inspection (2026-07-30): STANDING DUMBBELL OVERHEAD
# PRESS, both arms simultaneously.  Measured on good/front clips 001-005:
# rack p05 = 50-60 deg, lockout p95 = 157-175 deg, wrists finishing +0.76
# torso-lengths above the nose.  Both-arms-together ratio 10:1 to 14:1, so a
# single mean-driven state machine suffices (unlike the alternating curl).
#
# NO DATASET VALIDATION IS POSSIBLE FOR THE FAULT CHANNELS.  Across 5 good and
# 5 bad front clips no metric separated the folder labels — lockout_p95 was
# actually HIGHER on bad (164.5 vs 160.0 median), and each bad clip failed
# differently (001 torso swing 21.4 deg; 003 genuine partial reps with
# lockout_p95 119.6; 002 and 004 mechanically clean on every measure).  The
# bad/ label means "contains some fault, unspecified".  These thresholds are
# validated by synthetic driver only.

# -- Elbow flare (elbows drifting wide of the press path) ---------------------
# Value = outward-oriented (elbow_x - shoulder_x) / shoulder_width_px, MINUS
# the per-user baseline locked at calibration.  Baseline subtraction is not
# optional here: the bicep curl shipped the equivalent metric un-baselined and
# it read at or above its RED threshold while the subject stood at rest,
# because elbows do not sit directly beneath the shoulder joint.  Positive
# means flared outward from that user's own racked position.
PRESS_ELBOW_FLARE_YELLOW = 0.12  # feel-tuned initial value; not dataset-derived; see future work
PRESS_ELBOW_FLARE_RED    = 0.22  # feel-tuned initial value; not dataset-derived; see future work

# -- Body swing / lower-back arch (torso drive instead of a vertical press) ---
# Value = abs(trunk_lean - baseline_trunk_lean) in degrees.
# RARE-FIRE CHANNEL: measured separation is weak.  Trunk-deviation p95 was
# 2.5 deg on good clips vs 2.1 deg on bad (medians) - i.e. none - and only ONE
# clip in ten (bad/001, 21.4 deg) showed genuine torso drive.  Standing does
# leave the channel live, unlike a seated press, but these subjects brace well,
# so expect it to fire seldom rather than to carry the exercise.
PRESS_BODY_SWING_YELLOW = 6.0    # feel-tuned initial value; not dataset-derived; see future work
PRESS_BODY_SWING_RED    = 12.0   # feel-tuned initial value; not dataset-derived; see future work

# -- Left/right asymmetry ----------------------------------------------------
# Value = abs(elbow_angle_l - elbow_angle_r) in degrees, per frame.
# An INSTANTANEOUS comparison is valid here because the press is simultaneous.
# The bicep curl had to abandon the same per-frame measure for a per-rep one,
# because in an ALTERNATING curl one arm is flexed while the other is extended
# by design, which pinned that channel red on good and bad clips alike.
PRESS_ASYMMETRY_YELLOW = 10.0    # feel-tuned initial value; not dataset-derived; see future work
PRESS_ASYMMETRY_RED    = 20.0    # feel-tuned initial value; not dataset-derived; see future work

# -- Lockout completeness (per-rep verdict, NOT a live zone) ------------------
# A rep must satisfy BOTH conditions: the elbows reach PRESS_LOCKOUT_MIN_ANGLE
# AND the wrists rise at least PRESS_LOCKOUT_MIN_HEIGHT torso-lengths above the
# nose.  Angle alone can be satisfied by pressing forward rather than overhead;
# height alone can be satisfied with bent arms.  Measured good-clip lockout p95
# was 157-175 deg and wrist height p95 +0.75 to +0.83, so both thresholds sit
# just below observed good form.
PRESS_LOCKOUT_MIN_ANGLE  = 155.0  # feel-tuned initial value; not dataset-derived; see future work
PRESS_LOCKOUT_MIN_HEIGHT = 0.55   # feel-tuned initial value; not dataset-derived; see future work

# -- Rep state machine (degrees, mean of left and right elbow angle) ---------
# Racked at the shoulders -> ~50-60 deg; locked out overhead -> ~160-175 deg.
# The driver therefore INCREASES from the start pose, the OPPOSITE of both the
# bicep curl's elbow angle and the squat's knee angle.  Velocity signs follow:
# pressing up is POSITIVE velocity, lowering is negative.
PRESS_RACKED    = 80.0    # at/below this the weights count as racked (rep boundary)
PRESS_PRESSING  = 100.0   # rising above this with positive velocity starts the press
PRESS_LOCKED    = 155.0   # must be reached for the rep to count as a real press
PRESS_LOWERING  = 140.0   # falling back below this begins the eccentric phase

# Minimum frames held at lockout before the rep may progress - rejects bounced
# reps punched up and dropped immediately.
# FRAME-RATE WARNING: this dwell is counted in FRAMES, not seconds, so its
# wall-clock strictness varies with the source frame rate.  6 frames is
# ~200 ms at 30 fps but ~400 ms at 15 fps - and 15 fps is the rate of the
# project's shoulder dataset clips.  A slower source therefore silently
# demands a proportionally LONGER real-time hold at lockout.
# The same caveat applies to every *_FRAMES constant in this section.
PRESS_LOCKED_MIN_DWELL_FRAMES = 6   # feel-tuned initial value; not dataset-derived; see future work

# Frames the machine may sit in LOWERING before the descent is judged STALLED -
# the lifter parked mid-range instead of returning to the rack.  This is the
# ONLY route by which such a rep can be cued at all: it never completes, so the
# completion-time lockout verdict never runs for it and the failure mode would
# otherwise be silence.  Level-triggered, so the zone stays red while the stall
# lasts and satisfies CORRECTIVE_DWELL_FRAMES naturally.  45 frames ~ 1.5 s at
# 30 fps (~3.0 s at 15 fps - see the frame-rate warning above).
PRESS_LOWERING_STALL_FRAMES = 45  # feel-tuned initial value; not dataset-derived; see future work

# How long the per-rep lockout verdict is held as a synthetic red zone after a
# rep completes, so the dwell-based audio controller can observe it.  Must
# exceed CORRECTIVE_DWELL_FRAMES (12) for the cue to ever fire.
# NOTE: not in the original build brief, but required - an edge-triggered
# per-rep verdict cannot reach a controller that only counts CONSECUTIVE red
# frames unless it is held for longer than that controller's dwell window.
PRESS_LOCKOUT_CUE_HOLD_FRAMES = 15

# Calibration guard: the mean elbow angle must stay inside this band for every
# buffered frame, otherwise the lifter was not holding the racked start pose, and
# SessionController discards the buffer and resamples (best-of on timeout).
# Brackets the measured rack p05 of 50-60 deg with margin on both sides.
# Deliberately does NOT gate on knees: knee visibility drops to 0.10-0.16 on
# three of the four clips profiled, because the framing crops the lower legs.
PRESS_CALIB_ELBOW_MIN = 40.0
PRESS_CALIB_ELBOW_MAX = 80.0

# -- Shoulder press CSV outputs (must not collide with squat or curl) ---------
PRESS_CSV             = os.path.join(OUTPUTS_DIR, "shoulder_press_features.csv")
PRESS_REP_SUMMARY_CSV = os.path.join(OUTPUTS_DIR, "shoulder_press_rep_summary.csv")

PRESS_CSV_HEADER = [
    "frame",
    "shoulder_width_px", "torso_length_px",
    "upper_arm_l_px", "upper_arm_r_px", "forearm_l_px", "forearm_r_px",
    "elbow_angle_l_deg", "elbow_angle_r_deg", "elbow_mean_deg",
    "elbow_symmetry_diff_deg",
    # Baseline-subtracted flare, plus the locked baselines themselves so the
    # subtraction is auditable from the log the way baseline_trunk_lean_deg is.
    "elbow_flare_l", "elbow_flare_r",
    "baseline_elbow_flare_l", "baseline_elbow_flare_r",
    "wrist_above_nose_l", "wrist_above_nose_r",
    "trunk_lean_deg", "baseline_trunk_lean_deg", "trunk_deviation_deg",
    "zone_flare_l", "zone_flare_r", "zone_body_swing", "zone_asymmetry",
    "zone_overall", "phase", "rep_count",
]

PRESS_REP_SUMMARY_HEADER = [
    "rep_number", "start_frame", "end_frame", "duration_frames",
    "min_elbow_l", "max_elbow_l", "min_elbow_r", "max_elbow_r",
    "peak_flare_l", "peak_flare_r",
    "peak_trunk_deviation", "max_asymmetry",
    "peak_wrist_above_nose_l", "peak_wrist_above_nose_r",
    "lockout_achieved", "rep_quality",
]

# Phase colours for the press overlay (BGR), mirroring PHASE_COLORS for squat.
PRESS_PHASE_COLORS = {
    "RACKED":   (100, 210,  90),   # green
    "PRESSING": (  0, 165, 255),   # orange
    "LOCKED":   ( 80,  80, 255),   # red
    "LOWERING": (255, 200,  50),   # cyan-blue
}

# -- Shoulder press corrective clips -----------------------------------------
# Present in assets/audio/shoulder press/ and resolved by the RECURSIVE loader
# (23/23 clips cached across 3 directories), so the space in that folder name is
# handled.  A missing clip would degrade to a warning plus silence, not a crash.
AUDIO_PRESS_ELBOWS_IN   = "press_elbows_in.mp3"     # elbow flare red
AUDIO_PRESS_DONT_ARCH   = "press_dont_arch.mp3"     # body swing red
AUDIO_PRESS_MATCH_ARMS  = "press_match_arms.mp3"    # asymmetry red
AUDIO_PRESS_LOCK_OUT    = "press_lock_out.mp3"      # per-rep short lockout
AUDIO_PRESS_ALL_THE_WAY = "press_all_the_way.mp3"   # stalled in LOWERING

AUDIO_PRESS_CORRECTIVE_CLIPS = [
    AUDIO_PRESS_ELBOWS_IN, AUDIO_PRESS_DONT_ARCH, AUDIO_PRESS_MATCH_ARMS,
    AUDIO_PRESS_LOCK_OUT, AUDIO_PRESS_ALL_THE_WAY,
]

# Registering the names here is what makes the loader cache them; without this
# the clips would never be decoded even once the files exist.
EXPECTED_AUDIO_CLIPS = EXPECTED_AUDIO_CLIPS + AUDIO_PRESS_CORRECTIVE_CLIPS


# =============================================================================
# WEB LAYER  (api.py / app.py)
# =============================================================================
# Settings that belong to the HTTP front end only.  No analysis constant lives
# here: thresholds, smoothing windows, audio timings and CSV schemas are all
# above, shared with the desktop app and the evaluation scripts.

# Upload guards.  A 500 MB clip at ~11 fps of processing would run for hours, so
# both a size and a frame ceiling are enforced, and both are reported to the
# user as a plain message rather than a stack trace.
MAX_UPLOAD_BYTES   = 250 * 1024 * 1024        # 250 MB
MAX_FRAMES         = 9000                      # ~5 min at 30 fps
UPLOAD_CHUNK_BYTES = 1024 * 1024

ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
                      ".mpg", ".mpeg"}

# How many videos may be analysed at once.  MediaPipe inference is CPU-bound and
# single-threaded per landmarker, so a second concurrent job mostly steals
# cycles from the first; jobs queue instead.
MAX_CONCURRENT_JOBS = 1

# Finished jobs kept on disk.  Oldest are evicted once this many exist, so a
# long demo session cannot fill the disk with annotated videos.
MAX_RETAINED_JOBS = 12

# Candidate output codecs, best first.  H.264 in MP4 is the only combination
# every browser plays; `mp4v` (MPEG-4 Part 2) is NOT playable in Chrome or
# Firefox and is a last resort that at least leaves a downloadable file.
# Each candidate is probed at runtime — see src/video_io.py.
#
# VP8 sits second rather than first despite compressing ~30% smaller: measured
# on a 120-frame clip it took 19.4 s to H.264's 1.5 s, which would more than
# double the wall time of a job whose analysis is already the slow part.
VIDEO_CODECS = [
    ("avc1", ".mp4",  "H.264 / MP4"),
    ("VP80", ".webm", "VP8 / WebM"),
    ("mp4v", ".mp4",  "MPEG-4 Part 2 / MP4 (may not play in-browser)"),
]

# Longest side of the ANNOTATED VIDEO, in pixels.  Analysis always runs on the
# frame at its original size — this caps only the review copy that is encoded
# and streamed to the browser.
#
# It matters because OpenCV cannot control the encoder's bitrate on any backend
# available here (VIDEOWRITER_PROP_QUALITY is accepted and ignored), so output
# size is governed by pixel count alone: a 30 s portrait phone clip encodes to
# 67 MB at its native 810x1440 and 16 MB at 720.  Playback quality at 720 is
# ample for reviewing an overlay, and a phone clip is downscaled rather than a
# typical 640x480 webcam recording, which is untouched.
VIDEO_MAX_LONG_SIDE = 720

# =============================================================================
# LIVE STREAM  (src/live_session.py — the WebSocket the browser watches)
# =============================================================================
# The analyser emits each frame as it finishes it, so the browser shows the
# overlay, the zones, the charts and the coaching cue while the run is still
# going, instead of waiting for the whole clip.  These settings govern that
# channel only; nothing here can change a measurement.

# JPEG quality for the streamed preview frame.  The ANNOTATED VIDEO is written
# at VIDEO_MAX_LONG_SIDE by the codec above and is unaffected — this is the
# throwaway copy sent down the socket, and it is re-encoded every frame, so it
# is sized for latency rather than for archival quality.  72 keeps a 720x1280
# annotated frame around 45 KB, i.e. ~0.5 MB/s at the ~11 fps the pipeline
# sustains, which is nothing on a loopback or a LAN.
STREAM_JPEG_QUALITY = 72

# Longest side of the streamed preview.  Lower than the recorded video: the live
# canvas is a monitor, the file is the artefact.  Encoding cost scales with
# pixels and is paid inside the frame loop.
STREAM_MAX_LONG_SIDE = 640

# Frames the producer may run ahead of the socket.  This is the BACKPRESSURE
# knob: the analysis thread blocks once the queue is full, so a slow or paused
# client throttles the run rather than growing an unbounded buffer in memory.
# Small on purpose — a deep queue would only let the live view drift further
# behind the frame actually being analysed.
STREAM_QUEUE_FRAMES = 8

# Seconds to wait for a browser to attach its socket to an uploaded video before
# the job is abandoned.  Covers the gap between the upload response and the
# WebSocket handshake, and reclaims the folder if the tab is closed in between.
STREAM_ATTACH_TIMEOUT_S = 30.0
