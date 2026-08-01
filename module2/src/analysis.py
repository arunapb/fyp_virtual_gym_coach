"""
src/analysis.py — run the existing pipeline over an uploaded video, headlessly.

This is the web equivalent of `live_runner.run_live`, and it deliberately keeps
the same per-frame order so results are comparable frame for frame:

    read frame -> MediaPipe detect -> session.set_signal -> session.process_frame
    -> get_display_spec -> annotate + encode -> session.update_audio -> log CSVs

It contains NO analysis logic of its own.  Features, zones, rep counting,
calibration and cue decisions all come from `SessionController` and the active
`Exercise`, exactly as in the desktop app.

STREAMING IS THE PRIMARY SHAPE.  `analyse_stream` is a generator that yields one
event per frame the instant that frame is finished, so the browser can draw the
overlay, extend the charts and play the coaching cue while the rest of the clip
is still being analysed.  `analyse` is a thin drain of the same generator, kept
for callers that only want the finished document.  There is deliberately no
second loop: batch and live cannot drift because they are the same code.

Differences from `run_live`, each forced by the offline setting:

  * The video is read straight through once.  `run_live` rewinds at EOF so a demo
    clip loops forever on screen; an analysis job must end.
  * Frames are written to a video file (and optionally JPEG-encoded for the
    socket) instead of `cv2.imshow`, and the info bar is emitted as data instead
    of being drawn into pixels.
  * The audio clock is synthetic (`now = frame_id / fps`).  Cue cooldowns are
    wall-clock seconds by design, and processing runs at roughly 0.44x real time
    (docs/E7_PERFORMANCE.md), so a real clock would stretch every cooldown across
    far more video than the user will experience on playback.  Driving it from
    video time makes the cue schedule match what a live session would produce —
    the same technique `_accept_harness.py` uses for reproducibility.

The Module-1 signal is a per-frame input, not a run-wide setting.  Callers hand
in any `Module1SignalSource`: `ConstantSignalSource` for a scripted run, and
`LiveSignalSource` for the browser (and, when it lands, Module 1 itself).  A
signal that changes mid-clip is handled the way it will have to be in
production — the run is cut into SEGMENTS, one per span of active exercise, each
with its own calibration and its own repetitions.
"""

import os
import time

import cv2

from config import (
    AUDIO_FEEDBACK_ENABLED, BASELINE_FRAMES, MAX_FRAMES,
    STREAM_JPEG_QUALITY, STREAM_MAX_LONG_SIDE,
)
from src import series
from src.audio_bridge import CueRecorder, cue_caption, cue_category
from src.audio_feedback import FeedbackController, NullAudioController
from src.data_export import CsvLogger, spec_to_dict, zone_channels
from src.overlay import annotate_frame
from src.pose_extractor import create_landmarker, detect
from src.session import EXERCISE_REGISTRY, SessionController
from src.signal_source import ConstantSignalSource
from src.video_io import AnnotatedVideoWriter
from src.view_detection import ViewDetector

# Event types yielded by `analyse_stream`.
EVENT_META = "meta"       # once, before the first frame — geometry and fps
EVENT_FRAME = "frame"     # one per analysed frame
EVENT_DONE = "done"       # once, last — the full result document


class AnalysisCancelled(RuntimeError):
    """Raised when the caller's cancel check fires mid-run."""


def probe_video(path):
    """(fps, frame_count, width, height) for an uploaded file, or None if unreadable."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if w <= 0 or h <= 0:
        return None
    return (fps if fps and fps > 0 else 25.0), max(count, 0), w, h


def analyse(video_path, exercise_label, out_dir,
            progress=None, should_cancel=None) -> dict:
    """
    Analyse one uploaded video to completion and return the result document.

    A drain of `analyse_stream` with the signal pinned to one exercise — the
    behaviour every non-interactive caller wants, and identical frame for frame
    to what the live socket produces.
    """
    result = None
    for event in analyse_stream(video_path, ConstantSignalSource(exercise_label),
                                out_dir, progress=progress,
                                should_cancel=should_cancel):
        if event["type"] == EVENT_DONE:
            result = event["result"]
    return result


def analyse_stream(video_path, signal_source, out_dir, *, stream_frames=False,
                   progress=None, should_cancel=None):
    """
    Analyse one uploaded video, yielding each frame's findings as it finishes.

    signal_source  — `Module1SignalSource`, polled once per frame (the seam
                     Module 1 will occupy; see src/signal_source.py).
    stream_frames  — also JPEG-encode the annotated frame into each event, for a
                     client that is drawing it.  Off for batch runs, which would
                     otherwise pay the encode for nothing.
    progress(done, total)  — optional, called periodically.
    should_cancel()        — optional, truthy to abort the run.

    Yields, in order: one EVENT_META, one EVENT_FRAME per analysed frame, and
    one EVENT_DONE carrying the result document (also what `analyse` returns).

    Every artefact — the annotated video, the CSVs — is written exactly as in a
    batch run, so watching an analysis live and running it headlessly leave the
    same files on disk.
    """
    probe = probe_video(video_path)
    if probe is None:
        raise ValueError("The uploaded file could not be opened as a video. "
                         "Try re-exporting it as an MP4 (H.264).")
    fps, declared_frames, width, height = probe

    cap = cv2.VideoCapture(video_path)
    writer = AnnotatedVideoWriter(os.path.join(out_dir, "annotated"), fps,
                                  (width, height))
    csv_log = CsvLogger(out_dir)

    # The real controller with its speaker replaced — see src/audio_bridge.py.
    audio = (FeedbackController(player=CueRecorder())
             if AUDIO_FEEDBACK_ENABLED else NullAudioController())
    sessionc = SessionController(audio)
    # Pure observer, exactly as view_detection.py's Stage-1 contract requires:
    # its verdict is REPORTED to the user and consumed by nothing — no zone,
    # threshold, visibility gate or cue reads it.
    view_detector = ViewDetector()

    label_of = {cls: label for label, cls in EXERCISE_REGISTRY.items()}
    metric_key_cache = {}

    timeline, cues, view_counts = [], [], {}
    zone_counts, pose_frames, frame_id = {}, 0, 0
    segments, open_segment = [], None
    last_signal = None
    calib_done_frame = None
    truncated = False
    started = time.perf_counter()

    yield {
        "type": EVENT_META,
        "fps": round(fps, 3),
        "declared_frames": declared_frames,
        "max_frames": MAX_FRAMES,
        "source_width": width, "source_height": height,
        "width": writer.width, "height": writer.height,
        "codec": writer.codec_label,
        "calibration_frames": BASELINE_FRAMES,
    }

    try:
        with create_landmarker() as landmarker:
            while True:
                if should_cancel is not None and should_cancel():
                    raise AnalysisCancelled()
                ok, frame = cap.read()
                if not ok:
                    break                     # straight through, no rewind
                if frame_id >= MAX_FRAMES:
                    # The read above is what proves a frame was actually left
                    # unprocessed, so a clip of exactly MAX_FRAMES frames is not
                    # mislabelled as cut short.
                    truncated = True
                    break

                h, w = frame.shape[:2]
                pose_detected, lm = detect(landmarker, frame)

                # ── Module-1 signal, and the segment boundary it may open ─────
                signal = signal_source.current()
                if signal != last_signal:
                    # Snapshot BEFORE set_signal: a transition tears the active
                    # exercise down, taking its rep history and its calibration
                    # with it.  Only computed on a change, so a constant signal
                    # pays nothing for the possibility.
                    snapshot = (_segment_snapshot(open_segment, sessionc,
                                                  calib_done_frame, fps,
                                                  frame_id - 1)
                                if open_segment is not None else None)
                    previous = sessionc.active_exercise
                    sessionc.set_signal(signal)
                    last_signal = signal
                    # Whether a segment ended is decided by what the controller
                    # DID, not by the label: `set_signal` ignores an exercise it
                    # has no implementation for and leaves the current one
                    # running, and the segment must survive that untouched.
                    if sessionc.active_exercise is not previous:
                        if open_segment is not None:
                            open_segment.update(snapshot)
                            segments.append(open_segment)
                            open_segment = None
                        calib_done_frame = None
                        entered = label_of.get(type(sessionc.active_exercise))
                        if entered is not None:
                            open_segment = {"exercise": entered,
                                            "start_frame": frame_id,
                                            "start_time": round(frame_id / fps, 3)}

                fs = sessionc.process_frame(lm, pose_detected, w, h, frame_id)
                ex = sessionc.active_exercise
                ex_label = label_of.get(type(ex)) if ex is not None else None

                if ex is not None and sessionc.calib_done and calib_done_frame is None:
                    calib_done_frame = frame_id

                # ── Render ───────────────────────────────────────────────────
                spec = None
                if ex is not None:
                    spec = ex.get_display_spec(fs.features, fs.zones, fs.rep_state,
                                               fs.calib_progress, "Uploaded video",
                                               frame_id, pose_detected, lm)
                    annotate_frame(frame, spec, lm, fs.zones, pose_detected, w, h)
                writer.write(frame)

                # ── Audio ────────────────────────────────────────────────────
                cue = None
                clip = sessionc.update_audio(pose_detected, lm, fs,
                                             now=frame_id / fps)
                if clip:
                    cue = {"frame": frame_id, "time": round(frame_id / fps, 3),
                           "clip": clip, "caption": cue_caption(clip),
                           "category": cue_category(clip)}
                    cues.append(cue)

                csv_log.log(frame_id, ex, fs, sessionc.calib_done, pose_detected, lm)

                # ── Aggregates ───────────────────────────────────────────────
                view = None
                if pose_detected:
                    pose_frames += 1
                    view = view_detector.update(lm, w, h).view
                    view_counts[view] = view_counts.get(view, 0) + 1

                channels = zone_channels(fs.zones)
                for name, value in channels.items():
                    bucket = zone_counts.setdefault(name, {"green": 0, "yellow": 0, "red": 0})
                    bucket[value] = bucket.get(value, 0) + 1

                if ex_label not in metric_key_cache:
                    metric_key_cache[ex_label] = series.metric_keys(ex_label or "")
                entry = {
                    "f": frame_id,
                    "t": round(frame_id / fps, 3),
                    "pose": pose_detected,
                    "zones": channels,
                    "phase": fs.rep_state.phase if fs.rep_state else None,
                    # Read off the EXERCISE, not the frame's rep_state, which is
                    # absent on any frame without a pose or before calibration.
                    # A completed repetition does not un-complete because
                    # tracking dropped for a frame, and taking it from rep_state
                    # made the counter fall to zero and back on every such gap.
                    "reps": ex.rep_count if ex is not None else 0,
                    "view": view,
                    "info": spec_to_dict(spec) if spec is not None else None,
                    "metrics": _metrics(fs.features, metric_key_cache[ex_label]),
                    "signal": signal,
                    "ex": ex_label,
                    # None while an exercise is active; the Null/Rest prompt
                    # otherwise, so scrubbing back through an idle stretch shows
                    # what the user was told at the time instead of a blank.
                    "banner": fs.banner,
                }
                timeline.append(entry)

                # ── Live event ───────────────────────────────────────────────
                # Everything a client needs to update itself for this frame, and
                # nothing it can derive: the frame entry, plus the two things
                # that are edge-triggered rather than per-frame state.
                event = dict(entry)
                event["type"] = EVENT_FRAME
                event["banner"] = fs.banner
                event["calib"] = {"n": fs.calib_progress[0],
                                  "total": fs.calib_progress[1],
                                  "done": bool(fs.calib_progress[2]),
                                  "restarts": fs.calib_progress[3]}
                event["cue"] = cue
                if fs.rep_state is not None and fs.rep_state.new_rep_completed:
                    event["rep"] = dict(fs.rep_state.last_rep_summary)
                    # The column ORDER is the exercise's, not the dict's, and it
                    # differs per exercise — sent with the rep so a live table can
                    # be built without the result document that normally carries
                    # the header.  Only on a completed rep, so it costs nothing
                    # on the ~99% of frames that are mid-repetition.
                    event["rep_header"] = ex.rep_summary_header()
                else:
                    event["rep"] = None
                if stream_frames:
                    event["image"] = _encode_preview(frame)
                yield event

                frame_id += 1
                if progress is not None and frame_id % 5 == 0:
                    progress(frame_id, declared_frames)
    finally:
        cap.release()
        writer.release()
        csv_log.close()
        audio.shutdown()

    if open_segment is not None:
        open_segment.update(_segment_snapshot(open_segment, sessionc,
                                              calib_done_frame, fps, frame_id - 1))
        segments.append(open_segment)

    elapsed = time.perf_counter() - started
    dominant = _dominant_exercise(segments)
    merged = _merge_reps(segments, dominant)
    yield {
        "type": EVENT_DONE,
        "result": {
            "exercise": dominant,
            "video": {
                "file": os.path.basename(writer.path),
                "codec": writer.codec_label,
                "fps": round(fps, 3),
                "frames": frame_id,
                # Playback size.  `source_*` is what the analysis actually
                # measured; they differ only when a large upload was downscaled
                # for encoding.
                "width": writer.width,
                "height": writer.height,
                "source_width": width,
                "source_height": height,
                "scaled": writer.scaled,
                "duration": round(frame_id / fps, 2) if fps else 0.0,
                "truncated": truncated,
            },
            "processing": {
                "seconds": round(elapsed, 2),
                "fps": round(frame_id / elapsed, 2) if elapsed > 0 else 0.0,
                "realtime_ratio": round((frame_id / elapsed) / fps, 3) if elapsed > 0 and fps else 0.0,
            },
            "calibration": (segments[-1]["calibration"] if segments
                            else _calibration_summary(sessionc, None, fps)),
            "detection": {
                "pose_frames": pose_frames,
                "total_frames": frame_id,
                "rate": round(pose_frames / frame_id, 4) if frame_id else 0.0,
            },
            "view": {
                "counts": view_counts,
                "dominant": (max(view_counts, key=view_counts.get) if view_counts else None),
            },
            "reps": merged,
            "zones": {"counts": zone_counts},
            "cues": cues,
            "charts": series.charts_for(dominant or ""),
            # One entry per span of active exercise.  A run driven by
            # ConstantSignalSource has exactly one, and `reps`/`charts` above are
            # then simply that segment's — which is why pinning the exercise
            # (today) and letting Module 1 switch it (tomorrow) produce the same
            # document shape.
            "segments": [{k: v for k, v in s.items()} for s in segments],
            "artifacts": {
                "landmarks_csv": csv_log.landmarks_csv,
                "features_csv": csv_log.feature_csv,
                "rep_summary_csv": csv_log.rep_csv,
                "feature_csvs": list(csv_log.feature_csvs),
                "rep_summary_csvs": list(csv_log.rep_csvs),
            },
            "timeline": timeline,
        },
    }


def _encode_preview(frame):
    """
    The annotated frame as JPEG bytes for the live socket.

    Capped and re-encoded per frame, so it is sized for latency: the recorded
    video (src/video_io.py) is the artefact and is written at its own, higher
    cap from the same pixels.  Returns None if the encoder refuses the frame,
    which the client treats as "hold the previous image" rather than an error —
    a dropped preview must never fail an analysis.
    """
    h, w = frame.shape[:2]
    longest = max(w, h)
    if longest > STREAM_MAX_LONG_SIDE:
        scale = STREAM_MAX_LONG_SIDE / longest
        frame = cv2.resize(frame, (max(int(round(w * scale)), 2),
                                   max(int(round(h * scale)), 2)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _metrics(features, keys):
    """The charted subset of this frame's features, rounded for transport."""
    if not features or not keys:
        return {}
    out = {}
    for key in keys:
        value = features.get(key)
        if isinstance(value, (int, float)):
            out[key] = round(float(value), 4)
    return out


def _segment_snapshot(segment, sessionc, calib_done_frame, fps, end_frame):
    """
    The closing half of a segment record, read off the still-active exercise.

    Must be called while that instance is STILL the active one — a transition
    drops it, and its rep history and calibration go with it.  Returned rather
    than applied, because whether the segment actually ended is only known after
    the transition has been attempted.
    """
    ex = sessionc.active_exercise
    return {
        "end_frame": end_frame,
        "end_time": round(end_frame / fps, 3) if fps else 0.0,
        "frames": max(end_frame - segment["start_frame"] + 1, 0),
        "calibration": _calibration_summary(sessionc, calib_done_frame, fps),
        "reps": (_segment_reps(ex) if ex is not None
                 else {"count": 0, "header": [], "rows": [],
                       "quality": {"green": 0, "yellow": 0, "red": 0}}),
    }


def _segment_reps(ex):
    return {
        "count": ex.rep_count,
        "header": ex.rep_summary_header(),
        "rows": [dict(r) for r in ex.rep_history],
        "quality": _quality_counts(ex.rep_history),
    }


def _dominant_exercise(segments):
    """
    Which exercise the clip is mostly about — the one holding the most frames.

    With a single segment this is just "the exercise", which is the only case
    that exists until Module 1 is driving the signal.  It decides which charts
    and which rep table the summary leads with; every segment is reported in
    full regardless.
    """
    totals = {}
    for seg in segments:
        totals[seg["exercise"]] = totals.get(seg["exercise"], 0) + seg.get("frames", 0)
    return max(totals, key=totals.get) if totals else None


def _merge_reps(segments, exercise):
    """Every repetition of the dominant exercise, across all of its segments."""
    rows, header = [], []
    for seg in segments:
        if seg["exercise"] != exercise:
            continue
        header = header or seg["reps"]["header"]
        rows.extend(seg["reps"]["rows"])
    return {"count": len(rows), "header": header, "rows": rows,
            "quality": _quality_counts(rows)}


def _calibration_summary(sessionc, calib_done_frame, fps):
    """
    What the calibration phase actually did — restarts included.

    Worth surfacing: `SessionController` may discard a contaminated buffer and
    resample, and may fall back to the best buffer seen (`valid=False`) if no
    clean window exists.  On an uploaded clip the user cannot be asked to hold
    still and try again, so the report has to say when the baseline is shaky.
    """
    values = sessionc.calibration.values if sessionc.calibration else {}
    return {
        "done": sessionc.calib_done,
        "frames_required": BASELINE_FRAMES,
        "restarts": sessionc.calib_restarts,
        "completed_frame": calib_done_frame,
        "completed_time": (round(calib_done_frame / fps, 2)
                           if calib_done_frame is not None and fps else None),
        "clean": bool(values.get("valid", True)),
        "rejected_frames": values.get("rejected_frames"),
        "values": {k: (round(v, 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                   for k, v in values.items()},
    }


def _quality_counts(rep_history):
    counts = {"green": 0, "yellow": 0, "red": 0}
    for rep in rep_history:
        quality = rep.get("rep_quality")
        if quality in counts:
            counts[quality] += 1
    return counts
