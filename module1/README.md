# Virtual Gym Coach — Module 1

Real-time exercise recognition pipeline using MediaPipe Pose and a session-aware state machine.
This module processes recorded workout videos (via a web UI) or a local webcam/video (via the CLI) to classify the user's current workout state and exercise.

> **Version 2 Scope:**
> - **State Model:** Detects if the user is doing nothing (`null`), resting between sets (`rest`), or actively exercising (`active`).
> - **Exercise Model:** If `active`, identifies which exercise is being performed (e.g., squat, bicep_curl, deadlift, shoulder_press).
> - *Phase detection (descent, ascent, etc.) has been removed to simplify the baseline pipeline.*

---

## What's New

- **Web app**: a FastAPI backend (`api.py`) + static frontend (`frontend/`) let you upload a video in the browser and watch live state/exercise predictions streamed back over a WebSocket — no need to run the CLI or view an OpenCV window.
- **Trained models included**: `models/best_state_model.pt` and `models/best_exercise_model.pt` are committed, so the app runs out of the box without retraining.
- **Session pipeline module** (`src/session.py`): the per-connection pipeline (pose → features → sliding window → state model → state machine → exercise model) used by the WebSocket handler, decoupled from `app.py`'s CLI loop.
- The old dataset-prep / training scripts (`training/`) and empty `data/` folders have been removed from this branch — training now happens in a separate workflow (see `docs/` and `official_interim_report.md` if present).

---

## Project Structure

```
implementation/
│
├── api.py                  # FastAPI backend — video upload + WebSocket inference, serves frontend/
├── app.py                  # CLI entry point — runs the pipeline against a webcam or video file (OpenCV window)
├── config.py                # Tunable constants (window size, timeout, thresholds)
├── requirements.txt         # Python dependencies
├── .gitignore
│
├── frontend/                 # Static test harness UI served by api.py
│   ├── index.html            # Upload-a-video control panel + live readout
│   ├── app.js                # WebSocket client, drives the upload/inference flow
│   └── style.css
│
├── src/
│   ├── session.py           # GymCoachSession — per-connection pipeline used by api.py
│   ├── video_io.py          # Opens webcam or video file, resizes frames
│   ├── pose_extractor.py    # MediaPipe Pose: extracts 33 3D landmarks per frame
│   ├── data_export.py       # Saves frame-level landmarks to CSV (134 columns)
│   ├── features.py          # Normalisation, joint angles, motion energy, 32-frame sliding window
│   ├── state_rules.py       # Runs inference using best_state_model.pt
│   ├── state_machine.py     # Session-aware state machine (active → rest → null timeout)
│   ├── exercise_rules.py    # Runs inference using best_exercise_model.pt
│   └── overlay.py           # Draws state, exercise, and debug info on frame (CLI mode)
│
├── models/                   # Trained .pt model weights (committed)
│   ├── best_state_model.pt
│   └── best_exercise_model.pt
│
├── uploads/                  # Videos uploaded via the web UI, deleted after each session
└── outputs/                  # CLI landmark CSVs, debug videos, confusion matrices / ROC curves
```

---

## Setup Instructions

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## How to Run

### Option A — Web app (recommended)

Starts the FastAPI backend, which also serves the frontend UI, video upload endpoint, and the `/ws/session` WebSocket used for live inference.

```powershell
uvicorn api:app --reload
```

Then open **http://127.0.0.1:8000** in your browser:

1. Click **Upload Video** and pick a workout clip (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`).
2. The video is uploaded, then streamed frame-by-frame through the pipeline over a WebSocket — the frame the server just analyzed (with the skeleton overlay drawn in) is shown live, along with State, Exercise, Confidence, and per-class probability bars.
3. Click **Stop Session** to end early, or let the video play out — a per-exercise duration summary is shown at the end.

Useful endpoint: `GET /api/health` returns whether the state/exercise models loaded successfully.

### Option B — CLI (webcam or local video file, OpenCV window)

```powershell
# Default: use webcam (index 0)
python app.py

# Use a specific webcam index
python app.py --source 1

# Use a video file instead of a webcam
python app.py --source path\to\video.mp4

# Skip saving the landmarks CSV when the session ends
python app.py --no-save
```

#### What you will see (CLI mode)

| Element | Location | Colour |
|---------|----------|--------|
| **STATE** label | Top-left | Green = Active, Yellow = Rest, Red = Null |
| **Exercise** name | Below state | Green |
| Raw state probabilities | Bottom of frame | White |
| MediaPipe skeleton | Overlaid on body | Orange joints / Pink connections |

Press **`q`** in the video window to quit. Landmarks are saved to `outputs/landmarks.csv`, and per-exercise durations are printed to the console.

---

## State Logic (The "Final Set" Problem)

The `WorkoutStateMachine` in `src/state_machine.py` implements intelligent logic on top of the raw AI model predictions:

```text
No pose / unknown movement          →  NULL
                                         |
Person starts doing an exercise     →  ACTIVE  (session opens)
                                         |
Set ends, person pauses             →  REST  (session still open)
                                         |
   ┌── New set starts                →  ACTIVE  (session continues)
   └── Timeout expires (config.TIMEOUT_SECONDS) →  NULL   (session closes)
```
This ensures the app knows the difference between a pause mid-workout and someone completely walking away.
