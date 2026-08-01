"""
exercises/squat.py — SquatExercise.

This is the reference concrete exercise.  It does NOT reimplement any squat
maths: it DELEGATES to the existing measurements.py / zones.py / rep_counter.py
modules (which are also imported by the untouched evaluation scripts).  That
guarantees byte-identical squat behaviour before and after the refactor.

All squat-specific presentation (info-bar columns, phase labels, audio cue
mapping, CSV schema) is gathered here so the generic machinery stays agnostic.
"""

from collections import deque

from config import (
    BASELINE_FRAMES, SMOOTH_N, OVERALL_LABEL, PHASE_COLORS,
    SQUAT_CSV_HEADER, POSE_CONNECTIONS,
    SQUAT_CSV, REP_SUMMARY_CSV, REP_SUMMARY_HEADER,
    IDX_SHOULDER_L, IDX_SHOULDER_R, IDX_HIP_L, IDX_HIP_R,
    IDX_KNEE_L, IDX_KNEE_R,
)
from src.pose_utils import trunk_lean_angle, midpoint, get_px
from src.features import compute_body_measurements, compute_squat_features
from src.zones import compute_zones, get_zone_color, overall_zone
from src.rep_counter import RepCounter
from src.audio_feedback import default_corrective_channels

from src.exercises.base import (
    Exercise, Calibration, RepState, DisplaySpec, InfoCell, AudioCueMapping,
)


# Aura body-region groupings (per-leg / torso / other), coloured by zone.
_LEFT_LEG  = [(23, 25), (25, 27), (27, 29), (29, 31), (27, 31)]
_RIGHT_LEG = [(24, 26), (26, 28), (28, 30), (30, 32), (28, 32)]
_TORSO     = [(11, 12), (11, 23), (12, 24), (23, 24)]
_SPECIAL   = set(_LEFT_LEG) | set(_RIGHT_LEG) | set(_TORSO)
_OTHER     = [(a, b) for a, b in POSE_CONNECTIONS if (a, b) not in _SPECIAL]


class SquatExercise(Exercise):
    name = "Squat"
    calibration_pose_description = "Stand upright, feet shoulder-width apart"

    def __init__(self):
        # Zone smoothing buffers — created once per activation (== once per
        # session), exactly as the pre-refactor main loop created `smoother`.
        self._smoother = {k: deque(maxlen=SMOOTH_N) for k in (
            "valgus_l", "valgus_r", "trunk_dev", "depth",
            "valgus_3d_l", "valgus_3d_r", "knee_angle", "asymmetry")}
        self._rep_counter = RepCounter()
        self._prev_rep_count = 0

    # ── Calibration ──────────────────────────────────────────────────────────
    def calibrate(self, landmarks_buffer):
        """
        Average trunk-lean and hip-midpoint-y over the buffered calibration-pose
        frames.  The buffer holds exactly BASELINE_FRAMES (lm, w, h) tuples,
        collected only while this exercise is active and the user stands still —
        identical sampling to the pre-refactor calibration block.
        """
        trunk_samples, hip_y_samples = [], []
        for lm, w, h in landmarks_buffer:
            sh_mid = midpoint(get_px(lm, IDX_SHOULDER_L, w, h),
                              get_px(lm, IDX_SHOULDER_R, w, h))
            hi_mid = midpoint(get_px(lm, IDX_HIP_L, w, h),
                              get_px(lm, IDX_HIP_R, w, h))
            trunk_samples.append(trunk_lean_angle(sh_mid, hi_mid))
            hip_y_samples.append(hi_mid[1])
        n = len(trunk_samples)
        return Calibration({
            "trunk_lean": sum(trunk_samples) / n,
            "hip_mid_y":  sum(hip_y_samples) / n,
        })

    # ── Per-frame analysis (delegates to the unchanged squat modules) ─────────
    def compute_features(self, lm, calibration, w, h) -> dict:
        meas = compute_body_measurements(lm, w, h)
        # calibration is None until calibration completes.  trunk_lean=0.0 here
        # reproduces the pre-refactor provisional baseline (calib["trunk_lean"]
        # was initialised to 0.0), so trunk_lean_dev == raw lean during the
        # calibration window.
        trunk_lean = calibration.values["trunk_lean"] if calibration else 0.0
        feats = compute_squat_features(lm, meas, {"trunk_lean": trunk_lean}, w, h)
        return {**meas, **feats}      # flat dict; meas/feats keys are disjoint

    def classify_zones(self, features: dict) -> dict:
        return compute_zones(features, self._smoother)

    def update_rep_counter(self, features, zones, frame_id, lm) -> RepState:
        self._rep_counter.update(features, zones, frame_id, landmarks=lm)
        new = self._rep_counter.rep_count > self._prev_rep_count
        self._prev_rep_count = self._rep_counter.rep_count
        return RepState(
            phase=self._rep_counter.phase.value,
            rep_count=self._rep_counter.rep_count,
            last_rep_summary=self._rep_counter.last_rep_summary,
            new_rep_completed=new,
        )

    # ── Live accessors ────────────────────────────────────────────────────────
    @property
    def phase(self) -> str:
        return self._rep_counter.phase.value

    @property
    def rep_count(self) -> int:
        return self._rep_counter.rep_count

    @property
    def rep_history(self) -> list:
        return self._rep_counter.rep_history

    # ── Audio ─────────────────────────────────────────────────────────────────
    @property
    def audio_cue_mapping(self) -> AudioCueMapping:
        # Built from the same config-derived definition the FeedbackController
        # uses as its backward-compatible default -> identical cue decisions.
        return AudioCueMapping(channels=default_corrective_channels())

    # ── CSV ─────────────────────────────────────────────────────────────────
    # Stated explicitly rather than inherited: these are the exact paths and
    # header live_runner.py used to hardcode, so the squat's CSV output is
    # unchanged, and the squat no longer depends on being the interface default.
    def csv_path(self) -> str:
        return SQUAT_CSV

    def rep_summary_path(self) -> str:
        return REP_SUMMARY_CSV

    def rep_summary_header(self) -> list:
        return REP_SUMMARY_HEADER

    def csv_header(self) -> list:
        return SQUAT_CSV_HEADER

    def get_csv_row(self, frame_id, features, zones, rep_state, calib_done) -> list:
        f = features
        phase_value = rep_state.phase if (calib_done and rep_state) else "STANDING"
        rep_count   = rep_state.rep_count if (calib_done and rep_state) else 0
        return [
            frame_id,
            round(f["shoulder_width_px"], 2),
            round(f["hip_width_px"],      2),
            round(f["torso_len_px"],      2),
            round(f["femur_l_px"],        2),
            round(f["femur_r_px"],        2),
            round(f["tibia_l_px"],        2),
            round(f["tibia_r_px"],        2),
            round(f["avg_femur_px"],      2),
            round(f["avg_tibia_px"],      2),
            round(f["stance_width_px"],         2),
            round(f["norm_stance_width"],        4),
            round(f["knee_offset_l_px"],         2),
            round(f["knee_offset_r_px"],         2),
            round(f["norm_knee_offset_l"],       4),
            round(f["norm_knee_offset_r"],       4),
            round(f["hip_height_px"],            2),
            round(f["norm_hip_depth"],           4),
            round(f["trunk_lean_deg"],           3),
            round(f["baseline_trunk_lean_deg"],  3),
            round(f["trunk_lean_dev_deg"],       3),
            round(f["knee_angle_l_deg"],         3),
            round(f["knee_angle_r_deg"],         3),
            round(f["knee_sym_diff_deg"],        3),
            round(f["hip_angle_l_deg"],          3),
            round(f["hip_angle_r_deg"],          3),
            round(f["hip_sym_diff_deg"],         3),
            zones["valgus_l"], zones["valgus_r"],
            zones["trunk"],    zones["depth"],
            zones["overall"],
            phase_value, rep_count,
            round(f["knee_angle_3d_l_deg"],  3),
            round(f["knee_angle_3d_r_deg"],  3),
            round(f["knee_3d_sym_diff_deg"], 3),
            round(f["knee_valgus_3d_l"],     4),
            round(f["knee_valgus_3d_r"],     4),
            round(f["knee_fwd_l"],           4),
            round(f["knee_fwd_r"],           4),
            zones["valgus_3d_l"],
            zones["valgus_3d_r"],
            round(zones["asymmetry_score"], 3),
            zones["asymmetry"],
            zones["phase_label"],
        ]

    # ── Display spec ──────────────────────────────────────────────────────────
    def get_display_spec(self, features, zones, rep_state, calib_progress,
                         source_label, frame_id, pose_detected, lm) -> DisplaySpec:
        spec = DisplaySpec()
        spec.columns = self._build_columns(features, zones, calib_progress,
                                           source_label, frame_id, pose_detected, lm)
        spec.rep_row, spec.overlay = self._build_rep_and_overlay(rep_state, calib_progress[2])
        spec.overall = self._build_overall(zones, calib_progress)
        if pose_detected and zones:
            spec.aura_regions = [
                (_LEFT_LEG,  get_zone_color(zones["valgus_l"])),
                (_RIGHT_LEG, get_zone_color(zones["valgus_r"])),
                (_TORSO,     get_zone_color(zones["trunk"])),
                (_OTHER,     get_zone_color(zones["overall"])),
            ]
            spec.joint_colors = {
                IDX_KNEE_L: get_zone_color(zones["valgus_l"]),
                IDX_KNEE_R: get_zone_color(zones["valgus_r"]),
            }
            spec.torso_color = get_zone_color(zones["trunk"])
        return spec

    def _build_columns(self, features, zones, calib_progress,
                       source_label, frame_id, pose_detected, lm):
        n, total, done, restarts = calib_progress
        col1 = [
            InfoCell(f"Source : {source_label}", (200, 200, 200)),
            InfoCell(f"Frame  : {frame_id:05d}", (200, 200, 200)),
            InfoCell("Pose   : DETECTED" if pose_detected else "Pose   : not detected",
                     (0, 255, 120) if pose_detected else (80, 80, 255)),
        ]
        if pose_detected and lm:
            rk = lm[IDX_KNEE_R]
            col1.append(InfoCell(f"R.Knee : x={rk.x:.3f}  y={rk.y:.3f}", (255, 230, 0)))

        if not (pose_detected and features):
            return [col1, [InfoCell("Waiting for pose...", (100, 100, 150))], []]

        f = features
        col2 = [
            InfoCell(f"Knee  L:{int(round(f['knee_angle_l_deg'])):3d}  "
                     f"R:{int(round(f['knee_angle_r_deg'])):3d}  "
                     f"d:{int(round(f['knee_sym_diff_deg']))}", (0, 230, 255)),
            InfoCell(f"Hip   L:{int(round(f['hip_angle_l_deg'])):3d}  "
                     f"R:{int(round(f['hip_angle_r_deg'])):3d}  "
                     f"d:{int(round(f['hip_sym_diff_deg']))}", (0, 230, 255)),
            InfoCell(f"3D Knee L:{int(round(f['knee_angle_3d_l_deg'])):3d}  "
                     f"R:{int(round(f['knee_angle_3d_r_deg'])):3d}  "
                     f"d:{int(round(f['knee_3d_sym_diff_deg']))}", (0, 200, 220)),
            InfoCell(f"3D Fwd  L:{f['knee_fwd_l']:+.2f}  R:{f['knee_fwd_r']:+.2f}",
                     (160, 200, 200)),
        ]
        valgus_col = get_zone_color(
            overall_zone([zones["valgus_l"], zones["valgus_r"]]) if zones else "green")
        col3 = [
            InfoCell(f"Trunk  : {f['trunk_lean_deg']:+.1f}  dev:{f['trunk_lean_dev_deg']:+.1f}",
                     get_zone_color(zones["trunk"]) if zones else (255, 160, 40)),
            InfoCell(f"Valgus : L:{f['norm_knee_offset_l']:+.2f}  R:{f['norm_knee_offset_r']:+.2f}",
                     valgus_col),
            InfoCell(f"Depth  : {f['norm_hip_depth']:.2f}x femur",
                     get_zone_color(zones["depth"]) if zones else (130, 255, 100)),
            self._calib_cell(n, total, done, zones, restarts),
        ]
        return [col1, col2, col3]

    @staticmethod
    def _calib_cell(n, total, done, zones, restarts=0):
        if done and zones:
            return InfoCell(f"Asym  : {zones.get('asymmetry_score', 0.0):.1f} deg",
                            get_zone_color(zones.get("asymmetry", "green")))
        if done:
            return InfoCell("CALIBRATED", (0, 255, 120))
        # A restart resets the counter, which looks like a fault unless the reason
        # is shown.  ASCII only (cv2.putText / Hershey fonts).
        if restarts:
            return InfoCell(f"Calib: {n}/{total}  retry {restarts}", (255, 200, 0))
        return InfoCell(f"Calib: {n}/{total}  stand still", (255, 200, 0))

    def _build_rep_and_overlay(self, rep_state, done):
        if not (done and rep_state is not None):
            return None, None
        last = rep_state.last_rep_summary
        last_text = (f"Last: {self._q_label(last['rep_quality'])}  "
                     f"min:{last['min_knee_angle']:.0f}deg") if last else None
        last_color = get_zone_color(last["rep_quality"]) if last else None
        phase_color = PHASE_COLORS.get(rep_state.phase, (200, 200, 200))
        common = {"phase": rep_state.phase, "phase_color": phase_color,
                  "rep_count": rep_state.rep_count,
                  "last_text": last_text, "last_color": last_color}
        return dict(common), dict(common)

    @staticmethod
    def _build_overall(zones, calib_progress):
        n, total, done, restarts = calib_progress
        if zones:
            return {"text": f"Overall  :  {OVERALL_LABEL[zones['overall']]}",
                    "color": get_zone_color(zones["overall"])}
        if not done:
            # ASCII only — rendered via cv2.putText (Hershey fonts, no glyphs
            # outside ASCII); an em-dash here rendered as "???".
            note = (f" - restarted x{restarts}, keep still" if restarts else "")
            return {"text": f"Stand still - calibrating  {n} / {total}{note}",
                    "color": (255, 200, 0)}
        return None

    @staticmethod
    def _q_label(quality):
        return {"green": "GOOD", "yellow": "WARN", "red": "RISK"}.get(quality, quality)
