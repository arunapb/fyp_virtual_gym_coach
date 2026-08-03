"""
src/live_runner.py — interactive driver for the exercise-agnostic pipeline.

It owns the live frame loop: poll the Module-1 signal source (keyboard dev stub),
route it through the SessionController, render via the generic DisplaySpec
renderers, feed the audio controller, and log CSVs.

Keyboard (dev stub for Module 1):  1=Null  2=Rest  3=Squat  4=BicepCurl  ESC=quit

The pieces this loop drives all live beside it in `src/`: pose inference in
`pose_extractor`, drawing in `overlay`, CSV writing in `data_export`.  The loop
itself is only sequencing, which is what lets the offline analyser in
`analysis.py` run the identical per-frame order without sharing this file.
"""

import cv2

from config import AUDIO_FEEDBACK_ENABLED, OUTPUTS_DIR, WINDOW_TITLE
from src.audio_feedback import FeedbackController, NullAudioController
from src.data_export import CsvLogger
from src.overlay import annotate_frame, compose_display
from src.pose_extractor import create_landmarker, detect
from src.session import SessionController
from src.signal_source import KeyboardSignalSource


def _read_frame(cap, source, frame_id):
    """
    Read the next frame; rewind looping video at EOF.
    Returns (status, frame, frame_id) where status is "ok" / "rewind" / "stop".
    """
    success, frame = cap.read()
    if not success:
        if source == "video":
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return "rewind", None, 0
        print("Failed to read from webcam.")
        return "stop", None, frame_id
    if source == "webcam":
        frame = cv2.flip(frame, 1)
    return "ok", frame, frame_id


def _render(frame, ex, fs, lm, pose_detected, source_label, frame_id, w, h):
    """Compose the display (video frame + info bar / banner) for one frame."""
    if ex is None:
        return compose_display(frame, None, w, banner=fs.banner or "")

    spec = ex.get_display_spec(fs.features, fs.zones, fs.rep_state,
                               fs.calib_progress, source_label, frame_id,
                               pose_detected, lm)
    annotate_frame(frame, spec, lm, fs.zones, pose_detected, w, h)
    return compose_display(frame, spec, w)


def run_live(cap, source, source_label, wait_ms):
    # A FILE reports its true rate, so dwells and smoothing windows can be
    # derived from it (config.py's TIMEBASE).  A WEBCAM's CAP_PROP_FPS is the
    # sensor's nominal rate, not the rate this loop actually achieves — pose
    # inference sets that, and it varies with the machine — so the reported
    # value would be wrong in a way that is worse than the FPS_REFERENCE
    # default.  Live capture therefore keeps the historical behaviour until
    # there is a measured throughput to use instead.
    fps = cap.get(cv2.CAP_PROP_FPS) if source == "video" else None
    audio = (FeedbackController(fps=fps) if AUDIO_FEEDBACK_ENABLED
             else NullAudioController())
    session = SessionController(audio, fps=fps)
    signal_source = KeyboardSignalSource(default="Rest")
    csv_log = CsvLogger(OUTPUTS_DIR)

    print(f"Source opened  : {source_label}")
    print("Pipeline ACTIVE.  Keys: 1=Null 2=Rest 3=Squat 4=BicepCurl  ESC=quit\n")

    frame_id = 0
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    try:
        with create_landmarker() as landmarker:
            while True:
                status, frame, frame_id = _read_frame(cap, source, frame_id)
                if status == "stop":
                    break
                if status == "rewind":
                    continue
                h, w = frame.shape[:2]
                pose_detected, lm = detect(landmarker, frame)

                session.set_signal(signal_source.current())
                fs = session.process_frame(lm, pose_detected, w, h, frame_id)
                ex = session.active_exercise

                display = _render(frame, ex, fs, lm, pose_detected,
                                  source_label, frame_id, w, h)
                cv2.imshow(WINDOW_TITLE, display)

                session.update_audio(pose_detected, lm, fs)   # real wall-clock
                csv_log.log(frame_id, ex, fs, session.calib_done, pose_detected, lm)

                frame_id += 1
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == 27:                          # ESC
                    break
                signal_source.on_key(key)
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        audio.shutdown()
        csv_log.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nDone.")
