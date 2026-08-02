"""
backend/merged_analysis.py — the merged per-frame loop: Module 1 and Module 2
run on every frame, in one pass over the uploaded video, with Module 2's
calibration collected from the video's OPENING frames regardless of Module 1's
state, and Module 2's pose-correction (zones/reps/overlay/audio) gated on
Module 1 reporting the user as actually active.

Why this exists instead of driving module2/src/analysis.py's `analyse_stream()`
unmodified
------------------------------------------------------------------------------
`analyse_stream()` ties Module 2's calibration buffer strictly to "while an
exercise is active": `SessionController._enter_exercise()` always resets
`calib_buffer = []` and only starts filling it from that point forward. There
is no way to feed it frames collected before the exercise was signalled
without either editing `analyse_stream()`/`SessionController` (off limits) or
owning the frame loop, so this file reimplements that loop.

It is NOT a rewrite of Module 2's logic. Every non-trivial piece — pose
detection, `SessionController`, the `Exercise` classes (including their
`calibrate()`), zone classification, rep counting, overlay, audio, CSV/video
writing, the end-of-run result document — is imported from Module 2's `src`
and called in the exact order `analyse_stream()` already uses, so the output
document stays byte-for-byte the same shape. Structural comparison against
`module2/src/analysis.py` is intentional: keep the two easy to diff.

Pipeline order
---------------
    video starts
      -> CalibrationBank captures the user's baseline from the first
         BASELINE_FRAMES tracked frames  (backend/calibration_bank.py)
      -> Module 1 classifies state + exercise on every frame
      -> once Module 1 names a supported exercise, Module 2 activates and
         analyses using the ALREADY-CAPTURED baseline

Calibration is therefore its own stage: it does not wait on Module 1's state
and does not belong to Module 2's session. See backend/calibration_bank.py
for why it evaluates every registered exercise's `calibrate()` rather than
waiting to be told which one.

Pose-correction gating
------------------------
Falls out of Module 2's own `SessionController.process_frame()` unchanged:
`fs.zones`/`fs.rep_state` are only populated once `active_exercise` is set
AND `calib_done` is True. Module 1 naming a supported exercise is what sets
the former; the bank has normally already satisfied the latter, so analysis
begins on the same frame the exercise is recognised.
"""

import os
import time

import cv2

from config import AUDIO_FEEDBACK_ENABLED, BASELINE_FRAMES
from src import series
from src.audio_bridge import CueRecorder, cue_caption, cue_category
from src.audio_feedback import FeedbackController, NullAudioController
from src.data_export import CsvLogger, spec_to_dict, zone_channels
from src.exercises.base import DisplaySpec
from src.overlay import annotate_frame, draw_pose
from src.pose_extractor import create_landmarker, detect
from src.session import EXERCISE_REGISTRY, SessionController
from src.video_io import AnnotatedVideoWriter
from src.view_detection import ViewDetector
from src.analysis import (
    EVENT_DONE, EVENT_FRAME, EVENT_META,
    _dominant_exercise, _encode_preview, _merge_reps,
    _metrics, _segment_snapshot, probe_video,
)
# NOTE: module2's `AnalysisCancelled` is deliberately NOT raised here — see the
# cancel check inside the frame loop. `live_session.py` still catches it, which
# is now a dead branch for this generator but harmless, and correct if that
# base class is ever driven by module2's own `analyse_stream()`.

from backend import module1_worker
from backend.calibration_bank import CalibrationBank
from backend.module3 import exercise_log
from backend.signal_mapping import map_signal

_IDLE_MODULE1 = {"state": "null", "exercise": None, "label": None,
                 "confidence": 0.0, "state_probs": None, "early_guess": None}

# Module 2's own `annotate_frame()` only draws once real zone data exists
# (its own docstring: "leaves the raw frame visible during calibration").
# Standalone that's fine — the browser's exercise picker already told the
# user what's about to happen. Here Module 1 auto-detects, so before that
# there's otherwise no visual sign pose tracking is even running. A plain
# `DisplaySpec()` — all defaults, no zone colours — is exactly the
# "nothing to report yet" input `draw_pose()` (exercise-agnostic; it only
# reads torso_color/joint_colors, both defaulted) already supports.
_IDLE_SPEC = DisplaySpec()


def _run_module1(pool, frame, last_result):
    """Module 1's readout for this exact frame — the same frame Module 2's
    pose detection below runs on, read once, used by both."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return last_result  # hold the last known reading rather than lose it
    return pool.submit(module1_worker.process_frame, buf.tobytes()).result()


def analyse_stream_merged(video_path, pool, out_dir, *, stream_frames=False,
                          progress=None, should_cancel=None, max_frames=None,
                          username=None):
    """
    Analyse one uploaded video, yielding each frame's findings as it finishes,
    with Module 1 driving Module 2's signal on the same frame Module 2 itself
    reads — genuinely one pass over the file, not two independently-paced ones.

    Mirrors `module2/src/analysis.py`'s `analyse_stream()` field-for-field in
    its yielded events and final result document (see module docstring for
    why it's a parallel implementation rather than a call into that function).
    """
    probe = probe_video(video_path)
    if probe is None:
        raise ValueError("The uploaded file could not be opened as a video. "
                         "Try re-exporting it as an MP4 (H.264).")
    fps, declared_frames, width, height = probe
    frame_ceiling = max_frames if max_frames is not None else 9000

    cap = cv2.VideoCapture(video_path)
    writer = AnnotatedVideoWriter(os.path.join(out_dir, "annotated"), fps,
                                  (width, height))
    csv_log = CsvLogger(out_dir)

    audio = (FeedbackController(player=CueRecorder())
             if AUDIO_FEEDBACK_ENABLED else NullAudioController())
    sessionc = SessionController(audio)
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

    # Stage 1 of the pipeline, independent of Module 1 and Module 2.
    calib_bank = CalibrationBank()
    m1_last = dict(_IDLE_MODULE1)

    # Module 1's per-exercise duration tally — the measurement its standalone
    # app.py exported to data/session_summary.json ("Export exercise durations
    # for Module 3"), counted in FRAMES here and converted with the video's own
    # fps (app.py used a hardcoded frames / 30.0). `GymCoachSession` keeps its
    # own `exercise_durations`, but off `time.time()`: correct for a live
    # webcam, meaningless here, where frames run as fast as the CPU allows and
    # so would measure server effort rather than exercise. See
    # backend/module3/exercise_log.py.
    m1_active_frames = {}
    # Every frame, by Module 1's verdict — 'active' / 'rest' / 'null'. Rest is
    # what the browser shows as total rest time; 'null' is "nobody tracked",
    # which is not rest and is reported separately.
    m1_state_frames = {"active": 0, "rest": 0, "null": 0}
    logged_exercise = []
    stopped = False

    yield {
        "type": EVENT_META,
        "fps": round(fps, 3),
        "declared_frames": declared_frames,
        "max_frames": frame_ceiling,
        "source_width": width, "source_height": height,
        "width": writer.width, "height": writer.height,
        "codec": writer.codec_label,
        "calibration_frames": BASELINE_FRAMES,
    }

    try:
        with create_landmarker() as landmarker:
            while True:
                # Stop is an ENDING, not a failure. Breaking out (rather than
                # raising AnalysisCancelled, which unwinds past everything
                # below) means a stopped run still produces its result
                # document, annotated video and CSVs for the part that was
                # analysed — so the browser can show the session instead of
                # throwing it away and returning to the upload screen.
                if should_cancel is not None and should_cancel():
                    stopped = True
                    break
                ok, frame = cap.read()
                if not ok:
                    break                     # straight through, no rewind
                if frame_id >= frame_ceiling:
                    truncated = True
                    break

                h, w = frame.shape[:2]
                pose_detected, lm = detect(landmarker, frame)

                # ── Stage 1: calibration, from the video's opening frames ──
                if pose_detected:
                    calib_bank.observe(lm, w, h, frame_id)

                # ── Module 1, on this exact frame ──────────────────────────
                m1 = _run_module1(pool, frame, m1_last)
                m1_last = m1
                signal = map_signal(m1["state"], m1["exercise"], m1.get("early_guess"))

                # Only a CONFIRMED exercise while genuinely active counts
                # towards the log — `early_guess` is explicitly a not-yet-
                # locked hypothesis (it exists to give Module 2's calibration a
                # head start) and must not be billed as exercise time.
                m1_state_frames[m1["state"]] = m1_state_frames.get(m1["state"], 0) + 1
                if m1["state"] == "active" and m1["exercise"]:
                    m1_active_frames[m1["exercise"]] = \
                        m1_active_frames.get(m1["exercise"], 0) + 1

                # ── Module-1 signal, and the segment boundary it may open ──
                if signal != last_signal:
                    snapshot = (_segment_snapshot(open_segment, sessionc,
                                                  calib_done_frame, fps,
                                                  frame_id - 1)
                                if open_segment is not None else None)
                    previous = sessionc.active_exercise
                    sessionc.set_signal(signal)
                    last_signal = signal
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
                            # Stage 3: Module 2 starts on the baseline stage 1
                            # already captured, so analysis begins on this
                            # frame instead of after another 30-frame wait.
                            calib_bank.apply_to(sessionc, entered)

                fs = sessionc.process_frame(lm, pose_detected, w, h, frame_id)
                ex = sessionc.active_exercise
                ex_label = label_of.get(type(ex)) if ex is not None else None

                if ex is not None and sessionc.calib_done and calib_done_frame is None:
                    calib_done_frame = frame_id

                # Calibration progress + banner while no exercise is active.
                #
                # Module 2's own `fs.calib_progress`/`fs.banner` describe an
                # EXERCISE's calibration, so before one is active they report
                # (0, total, False, 0) and a keyboard-era Rest prompt — neither
                # true here. Stage 1's bank is the real calibration state.
                #
                # The banner keys off `pose_detected`, NOT Module 1's state:
                # `WorkoutStateMachine` (module1/src/state_machine.py) returns
                # 'null' for every frame until the model predicts 'active' even
                # once, because `session_open` starts False — so a clearly
                # tracked, standing user reads as state 'null' the whole time
                # they are just standing there. Showing "no user detected" off
                # that would contradict the skeleton drawn on the same frame.
                if ex is None:
                    calib_view = (calib_bank.collected, BASELINE_FRAMES,
                                  calib_bank.done, 0)
                    if not pose_detected:
                        banner_text = "No user detected"
                    elif not calib_bank.done:
                        banner_text = (f"Calibrating... "
                                       f"({calib_bank.collected}/{BASELINE_FRAMES})")
                    else:
                        banner_text = "Calibrated - waiting to identify your exercise"
                else:
                    calib_view = fs.calib_progress
                    banner_text = fs.banner

                # ── Render ───────────────────────────────────────────────────
                spec = None
                if ex is not None:
                    spec = ex.get_display_spec(fs.features, fs.zones, fs.rep_state,
                                               fs.calib_progress, "Uploaded video",
                                               frame_id, pose_detected, lm)
                    annotate_frame(frame, spec, lm, fs.zones, pose_detected, w, h)
                elif pose_detected:
                    # No exercise identified yet — bare skeleton only, so the
                    # user can see tracking is working during Null/Rest and
                    # while the video-start calibration buffer fills.
                    draw_pose(frame, lm, _IDLE_SPEC, w, h)
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
                    "reps": ex.rep_count if ex is not None else 0,
                    "view": view,
                    "info": spec_to_dict(spec) if spec is not None else None,
                    "metrics": _metrics(fs.features, metric_key_cache[ex_label]),
                    "signal": signal,
                    "ex": ex_label,
                    "banner": banner_text,
                }
                timeline.append(entry)

                # ── Live event ───────────────────────────────────────────────
                event = dict(entry)
                event["type"] = EVENT_FRAME
                event["banner"] = banner_text
                event["calib"] = {"n": calib_view[0],
                                  "total": calib_view[1],
                                  "done": bool(calib_view[2]),
                                  "restarts": calib_view[3]}
                event["cue"] = cue
                if fs.rep_state is not None and fs.rep_state.new_rep_completed:
                    event["rep"] = dict(fs.rep_state.last_rep_summary)
                    event["rep_header"] = ex.rep_summary_header()
                else:
                    event["rep"] = None
                event["module1"] = m1
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
        # Hand the session to Module 3: MET-based calories, appended to its
        # exercise_logs.json, which feeds the 7-day average behind the next
        # meal plan.
        #
        # In `finally`, not after the loop, because "the session is over"
        # includes the user pressing Stop — that path raises AnalysisCancelled
        # and unwinds straight past everything below. This block is the only
        # place both endings pass through. `save_session` never raises (see its
        # docstring): a nutrition-log failure must not sink an analysis whose
        # video, CSVs and results are already written.
        logged_exercise = exercise_log.save_session(
            {ex: n / fps for ex, n in m1_active_frames.items()} if fps else {},
            username=username)

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
                "width": writer.width,
                "height": writer.height,
                "source_width": width,
                "source_height": height,
                "scaled": writer.scaled,
                "duration": round(frame_id / fps, 2) if fps else 0.0,
                "truncated": truncated,
                # The user pressed Stop. Everything in this document is real,
                # it just covers fewer frames than the file holds.
                "stopped": stopped,
            },
            "processing": {
                "seconds": round(elapsed, 2),
                "fps": round(frame_id / elapsed, 2) if elapsed > 0 else 0.0,
                "realtime_ratio": round((frame_id / elapsed) / fps, 3) if elapsed > 0 and fps else 0.0,
            },
            # Stage 1's baseline is the calibration for this run. Module 2's
            # own per-segment summary is preferred when it actually ran its
            # own (bank rejected for that exercise); otherwise the bank is
            # the truth, and reporting Module 2's "never calibrated" here
            # would be wrong — the user WAS calibrated, before Module 2
            # started.
            "calibration": (segments[-1]["calibration"]
                            if segments and not calib_bank.done
                            else calib_bank.summary(fps, dominant)),
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
            # What Module 1 measured, and what Module 3 made of it. Both are
            # reported so the browser can show the hand-off rather than the
            # log silently appearing in a JSON file: `durations` is the raw
            # per-exercise active time, `log` is the records actually appended
            # to module3/backend/data/exercise_logs.json (empty when nothing
            # cleared exercise_log.MIN_LOGGED_SECONDS, or if Module 3 was
            # unreachable).
            "nutrition": {
                "durations": ({ex: round(n / fps, 2)
                               for ex, n in m1_active_frames.items()}
                              if fps else {}),
                "active_frames": dict(m1_active_frames),
                # Time spent between sets, and time with nobody tracked. Kept
                # apart because they mean different things: 'rest' is the user
                # recovering, 'null' is the pose model seeing no one.
                "rest_seconds": (round(m1_state_frames["rest"] / fps, 2)
                                 if fps else 0.0),
                "idle_seconds": (round(m1_state_frames["null"] / fps, 2)
                                 if fps else 0.0),
                "state_frames": dict(m1_state_frames),
                "log": logged_exercise,
                "calories_burned": round(
                    sum(e["calories_burned"] for e in logged_exercise), 2),
            },
            "charts": series.charts_for(dominant or ""),
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
