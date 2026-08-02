"""
src/pose_extractor.py — MediaPipe Pose Landmarker setup and per-frame inference.

This is the pipeline's only entry to the pose model.  The configuration used to
be written out separately in the desktop entry point, the offline analyser and
each harness; identical every time, but four copies that could drift, and a
drift here would silently change every downstream metric.

Configuration, and why each value is what it is
───────────────────────────────────────────────
  * **Lite model.**  A documented project decision: CPU-only deployment.  The
    accepted cost is noisy z-depth, which is why the 3-D channels are computed
    and logged but excluded from decisions (see `zones.py`).
  * **RunningMode.IMAGE.**  Synchronous, one frame at a time, with NO temporal
    state carried between frames.  That is what makes a run reproducible: the
    same video always yields the same landmarks, which is the property the
    acceptance harness relies on to compare outputs byte for byte.
  * **num_poses=1.**  One subject is the deployment assumption.
  * **All three confidences at 0.5.**  MediaPipe's defaults; never tuned, so
    they are not a hidden parameter of any published result.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import MODEL_PATH, MODEL_URL

MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


def ensure_model(verbose: bool = True) -> str:
    """
    Download the pose model on first use and return its path.

    Kept here rather than in an entry point so every way into the pipeline — the
    desktop app, the web API, a batch script — works in a fresh clone without
    each one repeating the fetch.
    """
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        if verbose:
            print(f"Downloading model to '{MODEL_PATH}' (~3 MB, one-time) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        if verbose:
            print("Download complete.\n")
    return MODEL_PATH


def landmarker_options() -> mp_vision.PoseLandmarkerOptions:
    """The single definition of how the Pose Landmarker is configured."""
    return mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )


def create_landmarker():
    """
    A ready-to-use Pose Landmarker, model downloaded if needed.

    Use it as a context manager so its native resources are released:

        with create_landmarker() as landmarker:
            detected, lm = detect(landmarker, frame)
    """
    ensure_model()
    return mp_vision.PoseLandmarker.create_from_options(landmarker_options())


def detect(landmarker, frame):
    """
    Run pose estimation on one BGR frame.

    Returns `(detected, landmarks)` — `landmarks` is the single pose's landmark
    list, or None when nothing was found.  The BGR→RGB conversion happens here
    because MediaPipe expects RGB while OpenCV hands us BGR; doing it at the
    boundary means no caller has to remember.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    detected = bool(result.pose_landmarks)
    return detected, (result.pose_landmarks[0] if detected else None)
