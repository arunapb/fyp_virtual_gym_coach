# backend — Virtual Gym Coach shared server

Single FastAPI app that combines all four modules on one port:

- **Module 1 + 2** — live pose analysis (webcam/video → exercise classification + form feedback)
- **Module 3** — meal recommendation & feedback
- **Module 4** — nutrition analysis from a food photo

## Prerequisites

- **Python 3.10–3.12**, verified on **3.12.10**. This range is a hard ceiling *and*
  a hard floor:
  - `mediapipe==0.10.14` publishes no wheel for 3.13+. On 3.13 the install dies with
    `No matching distribution found for mediapipe==0.10.14`. Newer mediapipe releases
    do have 3.13 wheels, but Module 1's pose extractor uses the legacy
    `mp.solutions.pose` API that those releases drop — so bumping the pin trades an
    install-time error for a runtime one.
  - The codebase's `X | None` union syntax (PEP 604) is evaluated eagerly in live
    function signatures, so it will not import on 3.9 or older.

  If your system Python falls outside the range:
  ```bash
  # macOS
  brew install python@3.12

  # Windows
  winget install --id Python.Python.3.12

  # Debian/Ubuntu
  sudo apt install python3.12 python3.12-venv
  ```

## One-time setup

Run these from the **repo root** (the directory containing `backend/`, `module1/`, etc.):

```bash
# 1. Create a virtual environment with Python 3.12
/opt/homebrew/bin/python3.12 -m venv .venv   # macOS
py -3.12 -m venv .venv                       # Windows

# 2. Activate it
source .venv/bin/activate                    # macOS/Linux
.\.venv\Scripts\Activate.ps1                 # Windows (PowerShell)

# 3. Install dependencies (pulls in torch, mediapipe, transformers, faiss, etc. — several minutes)
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

# 4. Download the spaCy model used by Module 3's feedback parser
python -m spacy download en_core_web_sm
```

> If `python` still resolves to another version after activating, the venv wasn't
> built with 3.12 — check `cat .venv/pyvenv.cfg`, and delete/recreate `.venv` if
> the `version =` line isn't `3.12.x`. Recreating is the fix for a venv that was
> already made with the wrong interpreter; changing your PATH won't retarget it.

## Running it (every time)

```bash
source .venv/bin/activate     # macOS/Linux
.\.venv\Scripts\Activate.ps1  # Windows (PowerShell)

python backend/run.py
```

Then open **http://127.0.0.1:8000/** in a browser.

Stop the server with `Ctrl+C`.

> **Do not** run `uvicorn backend.main:app` or `python backend/main.py` directly —
> `backend/run.py` exists specifically to keep Module 1's worker-process
> multiprocessing setup from breaking on import order. See that file's
> docstring for the full explanation.

## Routes

| Route | What it is |
|---|---|
| `/` | Workout coach UI (Modules 1 + 2) |
| `/login` | Demo sign-in screen (no real auth — every route is open) |
| `/nutrition` | Module 4 UI |
| `/meals` | Module 3 UI |
| `/api/docs` | Swagger UI for the full API |

## First-run downloads (needs internet)

These happen automatically the first time each feature is used, then cache locally:

- **Module 2's pose model** (`pose_landmarker_lite.task`) downloads from Google
  on the first video analysis.
- **Module 3/4's ML weights** (BERT ABSA, BART zero-shot, SentenceTransformer,
  NutriIngredientNet — roughly 2–3 GB total) download from Hugging Face on the
  first `/api/meals/feedback` or `/api/nutrition/predict` call, and cache in
  `~/.cache/huggingface`.

## About `backend/.env`

This file is currently **not read by the running server**:

- Module 4's settings only load `.env` from `backend/module4/.env` or
  `module4/.env` (neither exists) — it runs on defaults.
- Module 3's Qdrant config (`module3/scripts/qdrant_config.py`) isn't on the
  import path the shared server uses, so retrieval always falls back to the
  vendored local FAISS index in `module3/vector_store/` regardless of what's
  in `backend/.env`.

Both fall back gracefully, so this doesn't block running the app. To actually
wire up Qdrant Cloud or a custom Hugging Face token/device for Module 4, copy
the relevant values into `backend/module4/.env` (see `module4/.env.example`
for the keys Module 4 reads).
