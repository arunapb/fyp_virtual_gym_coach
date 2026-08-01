# Module 2 — Real-Time Pose Correction

Part of **"A Unified Health and Fitness Coaching Platform"** (team DeltaX,
University of Moratuwa, Level 4 FYP), supervised by Dr. Upeksha Ganegoda.

Existing camera-based fitness apps apply *fixed* joint-angle thresholds drawn
from population averages, which systematically penalise users whose limb
proportions differ from that average. Module 2 measures each user's own body
segments during a short calibration phase and expresses every biomechanical
metric as a **dimensionless ratio against that user's own anthropometry**, so
one set of thresholds is fair across body types. It classifies those metrics
into traffic-light risk zones and delivers visual and audio correction.

Single RGB camera, CPU-only inference, no wearables or depth sensors.

---

## Run it

```bash
cd module2
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:8000
```

Upload a video and **watch it being analysed**: the overlay, risk zones,
biomechanics charts, rep table and coaching audio all appear frame by frame as
the analysis runs, over a WebSocket. When it finishes, the same page becomes a
scrubbable review of the annotated recording it produced.

> **Port note.** Module 1's web UI also defaults to `8000`. To run both at once,
> give one of them another port: `python app.py --port 8100`.
> `--host 0.0.0.0` exposes it on the local network; `--reload` enables
> development auto-reload.

**Desktop** — live OpenCV window from a webcam:

```bash
python desktop_app.py
```

Both front ends drive the same pipeline in `src/`, so their numbers agree.

### First run

The pose model (~6 MB) downloads into `models/` automatically. Nothing else to
fetch — the coaching audio ships in `assets/audio/`.

---

## Layout

```
config.py            constants, thresholds, landmark indices, CSV schema, paths
api.py               FastAPI application (routes, uploads, the stream socket)
app.py               launcher for the web front end
desktop_app.py       launcher for the live OpenCV front end

src/                 the pipeline
  pose_extractor.py    MediaPipe setup + per-frame inference
  session.py           SessionController — signal routing, lifecycle, calibration
  signal_source.py     Module-1 signal abstraction
  exercises/           Exercise interface + squat / bicep curl / shoulder press
  features.py          body segments + per-exercise metrics
  zones.py             risk classification + smoothing
  rep_counter.py       repetition state machine
  overlay.py           DisplaySpec renderers + frame composition
  data_export.py       CSV writers + DisplaySpec-to-JSON
  audio_feedback.py    three-tier audio coaching controller
  view_detection.py    camera-view detector (pure observer)
  pose_utils.py        geometry helpers
  live_runner.py       desktop frame loop
  analysis.py          uploaded-video frame loop, streaming one event per frame
  live_session.py      the WebSocket a browser watches an analysis through
  video_io.py          codec-probing annotated-video writer
  audio_bridge.py      cue recorder + clip resolution
  jobs.py              analysis queue, progress, storage
  series.py            which metrics each exercise charts
  ranged.py            HTTP range support for video seeking
  gui.py               Tkinter source picker

frontend/            the web UI (index.html, style.css, app.js)
assets/audio/        the pre-rendered coaching clips (see its own README)
models/              pose model weights (downloaded on first run)
outputs/             generated CSVs; outputs/jobs/<id>/ per web analysis
uploads/             videos posted to the API, one folder per job
```

This mirrors Module 1's layout so the two modules of the platform read the same
way. Where a name would have been misleading it was kept accurate instead:
`rep_counter.py` counts repetition phases (Module 1's `state_machine.py`
detects Null/Rest/Active), and `zones.py` classifies injury risk (Module 1's
`state_rules.py` classifies session state).

Evaluation scripts, dataset outputs and the report write-ups are **not** in this
folder — they are not needed to run the module and live in the Module 2
development repository.

---

## How a frame is processed

Identical order in both front ends:

```
read frame → pose_extractor.detect → session.set_signal → session.process_frame
           → Exercise.get_display_spec → overlay → audio → data_export
```

`SessionController` translates the Module-1 signal (Null / Rest / `<Exercise>`)
into an active exercise, owns calibration, and enforces the per-frame order
`compute_features → classify_zones → update_rep_counter`. Adding an exercise is
a class plus one `EXERCISE_REGISTRY` entry — it then appears in the web UI, the
info bar, the rep table and the cue log automatically.

## Integrating with Module 1

Module 1 emits one label per frame — `Null`, `Rest`, or an exercise name — and
that is exactly the shape Module 2 consumes. `Module1SignalSource`
(`src/signal_source.py`) is the interface; the frame loop polls `current()` once
per frame and knows nothing about who is answering.

Three implementations exist today, and swapping in Module 1's classifier is a
fourth:

| Implementation | Used by |
|---|---|
| `LiveSignalSource` | the web app — thread-safe, written from the browser's signal control (and, later, from Module 1) |
| `KeyboardSignalSource` | `desktop_app.py` — keys 1–5 |
| `ConstantSignalSource` | scripted / batch runs |

A label that changes mid-session is already handled: `SessionController` tears
the old exercise down and calibrates the new one, and a web run records the
result as **segments**, one per span of active exercise. A label the registry
has no implementation for is ignored and the current exercise keeps running, so
an unrecognised signal degrades rather than breaking.

## Where the numbers come from

Every active threshold in `config.py` carries a provenance comment. The squat
thresholds are percentiles of **correct-rep frames only** in REHAB24-6 Exercise 6
(Černek et al., SISAP 2024, DOI 10.5281/zenodo.13305826) — yellow = p95,
red = p99. Because they are fitted only to correct-form data and never to the
incorrect-rep labels, the fault-detection evaluation is not circular. The bicep
curl and shoulder press thresholds are feel-tuned initial values pending
derivation, and say so at their definitions.
