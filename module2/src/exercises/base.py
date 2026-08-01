"""
exercises/base.py — the Exercise interface and the small data objects that
flow between an exercise, the SessionController, drawing.py, and the audio
controller.

Design goal: everything exercise-specific (which landmarks to calibrate, which
metrics to compute, which zones/clip cues exist, how to render the info bar)
lives behind this interface, so the rest of the system is exercise-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from config import SQUAT_CSV, REP_SUMMARY_CSV, REP_SUMMARY_HEADER


# =============================================================================
# DATA OBJECTS
# =============================================================================

@dataclass
class Calibration:
    """
    Opaque per-exercise anthropometric reference, produced by Exercise.calibrate
    from the first BASELINE_FRAMES detected frames of the calibration pose.

    For squats `values` holds {"trunk_lean": float, "hip_mid_y": float}.
    """
    values: dict = field(default_factory=dict)


@dataclass
class RepState:
    """Snapshot of the rep counter after one frame's update."""
    phase: str                       # exercise's phase name, e.g. "DESCENDING"
    rep_count: int
    last_rep_summary: dict | None    # most recent completed-rep summary (or None)
    new_rep_completed: bool          # True only on the frame a rep just finished


@dataclass
class InfoCell:
    """One labelled value in the info bar, with its render colour (BGR)."""
    text: str
    color: tuple = (200, 200, 200)


@dataclass
class DisplaySpec:
    """
    Everything drawing.py needs to render a frame for the active exercise.

    The generic renderer in drawing.py consumes this; the exercise decides the
    content (labels, units, which zone channel colours which body region).
    """
    # Info bar: up to three columns, each a list of InfoCell rows.
    columns: list = field(default_factory=list)
    # Rep / phase row beneath the columns (or None to hide).
    rep_row: dict | None = None
    # Overall-status centred row (or None).
    overall: dict | None = None
    # Aura glow regions: list of (connection_list, BGR_colour).
    aura_regions: list = field(default_factory=list)
    # Per-joint highlight colours by landmark index (e.g. knees by valgus zone).
    joint_colors: dict = field(default_factory=dict)
    # Colour of the torso centre-line (e.g. trunk zone colour).
    torso_color: tuple = (0, 180, 255)
    # Top-left rep badge overlay (or None).
    overlay: dict | None = None


@dataclass
class AudioCueMapping:
    """
    Exercise-specific corrective-cue description consumed by the audio
    controller.  `channels` is a priority-ordered list (highest first) of dicts:

        {
          "name":      str,            # dwell/cooldown key, e.g. "valgus"
          "zone_keys": list[str],      # zones dict keys; channel is "red" if
                                       #   ANY of these is "red"
          "phases":    set[str],       # rep phases in which the cue may fire
          "clips":     list[str],      # one entry, or several to rotate through
        }

    System-status and encouragement cues are generic (shared across exercises)
    and live in config.py, so they are not part of this mapping.
    """
    channels: list = field(default_factory=list)


# =============================================================================
# INTERFACE
# =============================================================================

class Exercise(ABC):
    """
    Abstract base every exercise implements.

    Lifecycle per activation (driven by SessionController):
      1. instance created
      2. calibrate(buffer) once the calibration-pose buffer is full
      3. compute_features / classify_zones / update_rep_counter every frame
      4. get_display_spec / get_csv_row each frame for rendering + logging
    """

    #: Human-readable name, must match the Module-1 signal label and registry key.
    name: str = "Exercise"

    #: Short on-screen instruction shown while the calibration buffer fills.
    calibration_pose_description: str = "Stand still"

    @abstractmethod
    def calibrate(self, landmarks_buffer):
        """
        Produce a Calibration from a buffer of the first BASELINE_FRAMES detected
        frames in the calibration pose.

        landmarks_buffer : list[(lm, w, h)] — collected ONLY while this exercise
        is active and the user holds the calibration pose (no buffering during
        Null/Rest, before activation).
        """

    @abstractmethod
    def compute_features(self, lm, calibration, w, h) -> dict:
        """
        Compute this frame's metrics.  `calibration` is None until calibration
        completes; an exercise must reproduce its pre-calibration provisional
        behaviour in that case.  Returns a flat dict of metric values.
        """

    @abstractmethod
    def classify_zones(self, features: dict) -> dict:
        """Map features to per-channel zone colours + overall (smoothed)."""

    @abstractmethod
    def update_rep_counter(self, features, zones, frame_id, lm) -> RepState:
        """Advance the rep-counter state machine by one frame."""

    @property
    @abstractmethod
    def audio_cue_mapping(self) -> AudioCueMapping:
        """Corrective-cue mapping consumed by the audio controller."""

    @abstractmethod
    def get_display_spec(self, features, zones, rep_state, calib_progress,
                         source_label, frame_id, pose_detected, lm) -> DisplaySpec:
        """Build the structured render description for drawing.py."""

    @abstractmethod
    def csv_header(self) -> list:
        """Column header for this exercise's per-frame feature CSV."""

    @abstractmethod
    def get_csv_row(self, frame_id, features, zones, rep_state, calib_done) -> list:
        """One per-frame feature CSV row matching csv_header()."""

    # ── CSV destinations ──────────────────────────────────────────────────────
    # Concrete (not abstract) so that adding these to the interface cannot break
    # an existing implementation.  The defaults return the squat constants the
    # runner used to hardcode, which is what keeps SquatExercise byte-identical
    # whether or not it overrides them.  An exercise whose rep-summary dict has
    # different keys MUST override all three: writing its rows against another
    # exercise's header raises KeyError, and sharing a path would have one
    # exercise silently overwrite the other's CSV.

    def csv_path(self) -> str:
        """Destination path for this exercise's per-frame feature CSV."""
        return SQUAT_CSV

    def rep_summary_path(self) -> str:
        """Destination path for this exercise's per-rep summary CSV."""
        return REP_SUMMARY_CSV

    def rep_summary_header(self) -> list:
        """
        Ordered keys of the rep-summary dict, forming both the summary CSV
        header and the field order of each row.
        """
        return REP_SUMMARY_HEADER

    # ── Live state accessors (default to the most recent update) ──────────────
    @property
    @abstractmethod
    def phase(self) -> str: ...

    @property
    @abstractmethod
    def rep_count(self) -> int: ...

    @property
    @abstractmethod
    def rep_history(self) -> list: ...
