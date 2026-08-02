"""
desktop_app.py — the interactive desktop entry point.

    python desktop_app.py

Opens a Tkinter picker for Webcam or Sample Video, then runs the live OpenCV
window: skeleton, coloured aura, rep badge, info bar and spoken coaching cues,
with the Module-1 signal driven from the keyboard until Module 1 is integrated.

To analyse a recorded video in a browser instead, run `python app.py`.  Both
front ends drive the SAME pipeline in `src/`, so their numbers agree.

Repository layout
─────────────────
  config.py          constants, thresholds, landmark indices, CSV schema, paths
  src/               the pipeline
    pose_extractor   MediaPipe setup + per-frame inference
    session          SessionController: signal routing + per-frame pipeline
    exercises/       Exercise interface + squat / bicep curl / shoulder press
    features         body segments + per-exercise metrics
    zones            risk classification + smoothing
    rep_counter      repetition state machine
    overlay          DisplaySpec renderers + frame composition
    data_export      CSV writers + DisplaySpec-to-JSON
    audio_feedback   three-tier audio coaching controller
    live_runner      this app's frame loop
    analysis         the offline (uploaded-video) frame loop
  frontend/          the web UI
  scripts/           evaluation + diagnostics
  models/            pose model weights
  outputs/           generated CSVs

Module-1 signal (keyboard dev stub):  1=Null  2=Rest  3=Squat  4=BicepCurl
"""

import os

import cv2
from tkinter import messagebox

from config import SAMPLE_VIDEO
from src.gui import show_source_chooser
from src.live_runner import run_live
from src.pose_extractor import ensure_model


def main():
    # Download the pose model on first run (~3 MB, one-time).
    ensure_model()

    source = show_source_chooser()
    if source is None:
        print("No source selected. Exiting.")
        return

    if source == "webcam":
        cap = cv2.VideoCapture(0)
        source_label = "Webcam"
        if not cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Cannot open webcam.\nCheck that a camera is connected.")
            return
    else:
        cap = cv2.VideoCapture(SAMPLE_VIDEO)
        source_label = "Sample Video"
        if not cap.isOpened():
            messagebox.showerror(
                "Video Error", f"Cannot open video:\n{os.path.abspath(SAMPLE_VIDEO)}")
            return

    # Play the sample video at its native rate; process webcam as fast as it can.
    if source == "video":
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_ms = max(1, int(1000 / fps)) if fps > 0 else 33
    else:
        wait_ms = 1

    run_live(cap, source, source_label, wait_ms)


if __name__ == "__main__":
    main()
