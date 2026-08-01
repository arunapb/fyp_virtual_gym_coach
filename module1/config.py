# =============================================================================
# Configuration — Virtual Gym Coach  |  Module 1  |  Version 1
# Scope: State detection (null / rest / active) + Exercise recognition
# =============================================================================

# ── Version 1 Scope ───────────────────────────────────────────────────────────
# Supported exercises — the exercise model classifies active windows into one
# of these classes. Add more as you collect data for them.
TARGET_EXERCISES = ['squat', 'bicep_curl', 'deadlift', 'shoulder_press']

# ── Video Processing ──────────────────────────────────────────────────────────
RESIZE_WIDTH  = 640
RESIZE_HEIGHT = 480

# ── State Machine ─────────────────────────────────────────────────────────────
# How long (seconds) to stay in 'rest' before converting to 'null'.
# If no new active movement is detected within this window, the session closes.
TIMEOUT_SECONDS = 60.0

# ── Temporal Window ───────────────────────────────────────────────────────────
WINDOW_SIZE     = 32   # Number of frames per sliding window
WINDOW_OVERLAP  = 0.5  # 50% overlap → step size = 16 frames

# ── Motion Energy Thresholds (for future rule-based or debug use) ─────────────
# Tune these values by observing printed energy values during a live session.
ACTIVE_ENERGY_THRESHOLD = 0.05  # Above this → likely active movement
REST_ENERGY_THRESHOLD   = 0.01  # Below this → essentially still

# ── Pose Quality ──────────────────────────────────────────────────────────────
# Minimum number of landmarks that must have visibility > 0.5 to consider
# the pose valid. Below this, treat the frame as 'no pose detected'.
MIN_VISIBLE_LANDMARKS = 15
