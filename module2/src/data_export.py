"""
src/data_export.py — everything that leaves the pipeline as data rather than pixels.

Two destinations, one source of truth:

  * **CSV** — the three per-session files (`pose_landmarks`, the exercise's
    feature log, and its rep summary).  `CsvLogger` is used by BOTH the desktop
    runner and the offline analyser; they differ only in the directory they
    write to, so parameterising that removes the duplicate writer the two used
    to keep separately.
  * **JSON** — a DisplaySpec turned into the structure the browser renders as a
    live info bar.  Same cells, same colours, as real text.

Schemas are not defined here.  Headers and row order come from the active
Exercise (`csv_header()` / `get_csv_row()` / `rep_summary_header()`), so a new
exercise needs no change to this module.
"""

import csv
import os

# =============================================================================
# CSV
# =============================================================================

LANDMARK_CSV_HEADER = ["frame", "landmark", "x", "y", "z", "visibility"]


class CsvLogger:
    """
    Writes the per-session CSVs into `out_dir`.

    The landmark file opens immediately; an exercise's feature and rep-summary
    files open lazily on the first frame that exercise is active, because their
    headers come from that exercise and it is not known before then.

    File NAMES come from the exercise (`ex.csv_path()`), reduced to their
    basename so the same exercise writes `squat_features.csv` whether the
    destination is `outputs/` for a desktop session or a per-job folder for a
    web upload.  A downloaded file is therefore a drop-in for the evaluation
    scripts that consume it.

    ONE PAIR OF FILES PER EXERCISE, not per session.  The Module-1 signal may
    name a different exercise partway through a session, and each exercise
    defines its own `csv_header()`; writing a bicep-curl row under a squat
    header would produce a file that silently misaligns column to value.
    Handles are kept open and keyed by name, so a session that returns to an
    earlier exercise appends to that exercise's file in frame order instead of
    truncating what it already wrote.
    """

    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self._dir = out_dir
        self._lm_file = open(os.path.join(out_dir, "pose_landmarks.csv"),
                             "w", newline="", encoding="utf-8")
        self._lm = csv.writer(self._lm_file)
        self._lm.writerow(LANDMARK_CSV_HEADER)
        self._files = {}                   # basename -> open handle
        self._writers = {}                 # exercise class -> (feat, rep) writers
        self.landmarks_csv = "pose_landmarks.csv"
        self.feature_csvs = []             # in the order the exercises appeared
        self.rep_csvs = []

    @property
    def feature_csv(self):
        """The first exercise's feature file — what a single-exercise run wrote."""
        return self.feature_csvs[0] if self.feature_csvs else None

    @property
    def rep_csv(self):
        return self.rep_csvs[0] if self.rep_csvs else None

    def log(self, frame_id, ex, fs, calib_done, pose_detected, lm):
        """
        Write this frame's rows.  Nothing is logged before an exercise is active
        or on a frame with no pose, which is what keeps the landmark, feature and
        rep files aligned on frame index.
        """
        if ex is None or not pose_detected:
            return
        feat, rep = self._writers.get(type(ex)) or self._open_exercise_files(ex)
        for idx, p in enumerate(lm):
            self._lm.writerow([frame_id, idx, round(p.x, 6), round(p.y, 6),
                               round(p.z, 6), round(p.visibility, 6)])
        feat.writerow(ex.get_csv_row(frame_id, fs.features, fs.zones,
                                     fs.rep_state, calib_done))
        if fs.rep_state is not None and fs.rep_state.new_rep_completed:
            summary = fs.rep_state.last_rep_summary
            rep.writerow([summary[k] for k in ex.rep_summary_header()])

    def _open_exercise_files(self, ex):
        feature_name = os.path.basename(ex.csv_path())
        rep_name = os.path.basename(ex.rep_summary_path())
        feat = self._open(feature_name, ex.csv_header())
        rep = self._open(rep_name, ex.rep_summary_header())
        self.feature_csvs.append(feature_name)
        self.rep_csvs.append(rep_name)
        self._writers[type(ex)] = (feat, rep)
        return feat, rep

    def _open(self, name, header):
        handle = open(os.path.join(self._dir, name), "w", newline="",
                      encoding="utf-8")
        self._files[name] = handle
        writer = csv.writer(handle)
        writer.writerow(header)
        return writer

    def close(self):
        self._lm_file.close()
        for handle in self._files.values():
            handle.close()


# =============================================================================
# DISPLAYSPEC -> JSON
# =============================================================================

def bgr_to_hex(color) -> str:
    """
    OpenCV colour (B, G, R) -> CSS "#rrggbb".

    Every colour in the pipeline — zone colours from `zones.get_zone_color`,
    phase colours from config — is authored in BGR for cv2, so the web layer
    converts rather than maintaining a second palette that could drift.
    """
    if not color:
        return "#c8c8c8"
    b, g, r = (int(c) for c in color[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _cells(cells):
    return [{"text": c.text, "color": bgr_to_hex(c.color)} for c in cells]


def _row(row):
    """Serialise a rep_row / overlay dict, converting its BGR colours."""
    if row is None:
        return None
    out = {k: v for k, v in row.items() if not k.endswith("_color")}
    for key in ("phase_color", "last_color"):
        if key in row:
            out[key] = bgr_to_hex(row[key])
    return out


def json_default(value):
    """
    Make NumPy scalars serialisable, for `json.dump(..., default=json_default)`.

    Angles, distances and normalised ratios come back from `pose_utils` as
    `np.float64` (via `np.arccos`, `np.linalg.norm`), which `json` rejects.
    Converting at the encoder boundary keeps the pipeline's own types untouched,
    and living here means the file writer and the WebSocket agree on it rather
    than each carrying a copy.
    """
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def spec_to_dict(spec) -> dict:
    """The info bar the desktop app draws, as data the browser can render."""
    return {
        "columns": [_cells(col) for col in spec.columns[:3]],
        "rep_row": _row(spec.rep_row),
        "overall": ({"text": spec.overall["text"],
                     "color": bgr_to_hex(spec.overall["color"])}
                    if spec.overall else None),
    }


# Zone channels are identified by VALUE, not by a hard-coded key list: a zones
# dict also carries non-zone entries (the squat's `asymmetry_score` float and
# `phase_label`, the press's `phase`), and a new exercise may add more.  Reading
# only the traffic-light values keeps this generic for every current and future
# exercise without the web layer knowing any exercise's channel names.
_ZONE_VALUES = ("green", "yellow", "red")


def zone_channels(zones) -> dict:
    """Traffic-light channels only, e.g. {"valgus_l": "red", "trunk": "green"}."""
    if not zones:
        return {}
    return {k: v for k, v in zones.items()
            if isinstance(v, str) and v in _ZONE_VALUES}
