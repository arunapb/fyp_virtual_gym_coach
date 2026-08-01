"""
src — the Module 2 pose-correction pipeline.

Layered, exercise-agnostic, and driven identically by both front ends
(`desktop_app.py` live, `app.py` over an uploaded video):

    pose_extractor      MediaPipe Pose Landmarker setup + per-frame inference
        │
    session             SessionController — Module-1 signal routing, the
        │               activation lifecycle, calibration, per-frame order
        ├── signal_source   Module-1 signal abstraction (ABC + 2 impls)
        ├── exercises/      Exercise interface + squat / bicep curl / press
        │     ├── features        body segments + per-exercise metrics
        │     ├── zones           risk classification + smoothing
        │     └── rep_counter     repetition state machine
        ├── overlay         DisplaySpec renderers + frame composition
        ├── audio_feedback  three-tier audio coaching controller
        └── data_export     CSV writers + DisplaySpec-to-JSON

    live_runner         the interactive frame loop (desktop)
    analysis            the offline frame loop (uploaded video)
        ├── video_io        codec-probing annotated-video writer
        ├── audio_bridge    cue recorder + clip resolution
        ├── jobs            analysis queue, progress, storage
        ├── series          which metrics each exercise charts
        └── ranged          HTTP range support for video seeking

    view_detection      camera-view detector — a PURE OBSERVER; no zone,
                        threshold, gate or cue reads it
    pose_utils          geometry helpers
    gui                 Tkinter source picker for the desktop app

Constants, thresholds, landmark indices, CSV schemas and paths all live in the
repository-root `config.py`, which every module here imports.
"""
