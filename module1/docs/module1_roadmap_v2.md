# Module 1 — Revised Roadmap v2
**Updated:** 2026-05-04  
**Scope:** State detection (null / rest / active) + Exercise recognition (squat, bicep_curl, etc.)  
**Phase detection:** Removed — not in scope for Module 1  

---

## Where You Are Right Now

| Step | What it is | Status |
|------|-----------|--------|
| Video input | Read webcam or video file | ✅ Done |
| Pose extraction | MediaPipe finds 33 body joints | ✅ Done |
| Landmark saving | Saves joints to CSV | ✅ Done |
| Feature computation | Angles, motion energy, normalization | ✅ Done |
| Sliding window | 32-frame, 50% overlap buffer | ✅ Done |
| Folder structure | data/, landmarks/, annotations/, windows/, labels/ | ✅ Done |
| Annotation template | 6-column CSV template ready | ✅ Done |
| Dataset builder | build_dataset.py creates .npy windows | ✅ Done |
| Train/val/test splitter | split_dataset.py splits by video ID | ✅ Done |
| State machine | Timeout logic (rest → null after 5s) | ✅ Done |
| **State model** | BiLSTM trained on null/rest/active | ❌ Not started |
| **Exercise model** | Classifier trained on active windows | ❌ Not started |
| **Live integration** | Replace random placeholders with models | ❌ Not started |

---

## The New Goal

```
Every 32-frame window → STATE MODEL → null / rest / active
                                              ↓ (only if active)
                                      EXERCISE MODEL → squat / bicep_curl / shoulder_press / deadlift
```

Two models. No phases. Clean and achievable.

---

## Step 1 — Collect Data for NULL (No exercise happening)

### What you do
Download the **Charades dataset** (everyday indoor activities).  
Every Charades clip counts as `null` for your project because no one is doing target exercises.

### Why
Your state model needs examples of null to learn what "not exercising" looks like.  
Charades already has 9,848 videos — you do not need to record null footage yourself.

### How
1. Request access and download Charades from [allenai.org/plato/charades](https://allenai.org/plato/charades)
2. Take a subset — 200–500 short clips is enough to start
3. Run `app.py` on each video to extract MediaPipe landmarks:
   ```powershell
   python app.py --source path/to/charades_clip.mp4
   ```
4. Rename the output: `outputs/landmarks.csv` → `data/landmarks/{video_id}_landmarks.csv`
5. Write the annotation CSV (entire clip = null, one row per clip):
   ```csv
   video_id,start_frame,end_frame,state,exercise,notes
   char_001,0,299,null,none,charades clip - pouring water
   ```

### Output
- `data/landmarks/char_001_landmarks.csv`, `char_002_landmarks.csv`, ...
- `data/annotations/char_001_annotation.csv`, `char_002_annotation.csv`, ...

### Validate
```powershell
python training/validate_annotations.py data/annotations/
```

---

## Step 2 — Collect Data for ACTIVE (Squat exercise)

### What you do
Download the **Penn Action Dataset** which has real squat video clips with frame-level joints.  
Also record yourself doing squats to supplement.

### Why
Your models need examples of someone actually squatting.  
For the state model, the entire clip = `active`.  
For the exercise model, the entire clip = `active` + `exercise=squat`.

### How
1. Download Penn Action: `https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz`
2. Filter the squat clips only (Penn Action has 15 exercise types)
3. Run `app.py` on each squat clip to extract landmarks
4. Write annotation CSVs:
   ```csv
   video_id,start_frame,end_frame,state,exercise,notes
   penn_sq_001,0,149,active,squat,penn action squat clip
   ```
5. Record your own squat sessions for extra variety:
   ```powershell
   python app.py --source data/raw_videos/p01_squat_s01.mp4
   ```

### Output
- Landmark CSVs for squat clips
- Annotation CSVs with `state=active, exercise=squat`

### Validate
```powershell
python training/validate_annotations.py data/annotations/
```

---

## Step 3 — Collect Data for REST and Other Exercises

### What you do
Record yourself **standing still between squat sets** (for REST).  
Record yourself doing **bicep curls, shoulder press** (for exercise recognition).

### Why
- The state model needs rest examples to learn the difference between "doing nothing" (null) and "pausing between sets" (rest).
- The exercise model needs non-squat exercise examples so it can distinguish between exercises.

### How — REST
1. Do a squat set, then stand still and breathe for 30–60 seconds. Repeat 5–10 times.
2. Record this as a video, place in `data/raw_videos/`
3. Run `app.py` to extract landmarks
4. Annotate: entire standing-still segment = `rest`

### How — Other Exercises
1. Record 10–20 minutes of each target exercise (bicep curl, shoulder press, etc.)
2. Run `app.py` to extract landmarks
3. Annotate: entire clip = `active`, `exercise=bicep_curl` etc.

### Output
- Landmark CSVs for rest clips and other exercise clips
- Annotation CSVs with correct state and exercise labels

### Minimum data you need before training

| Label | Minimum clips | Notes |
|-------|--------------|-------|
| `null` | 100–300 clips | Use Charades |
| `rest` | 20–50 clips | Record yourself |
| `active / squat` | 50–150 clips | Penn Action + yourself |
| `active / bicep_curl` | 20–50 clips | Record yourself |
| `active / shoulder_press` | 20–50 clips | Record yourself |

---

## Step 4 — Build the Window Dataset

### What you do
Run the dataset builder to convert all landmark CSVs + annotation CSVs into fixed-size `.npy` windows.

### Why
The model does not train on raw video or even raw landmarks.  
It trains on 32-frame windows. Each window is a `(32, 142)` NumPy array with one label.

### How
```powershell
cd f:\research\implementation
venv\Scripts\activate
python training/build_dataset.py
```

### Output
- `data/windows/all/*.npy` — thousands of window files, each `(32, 142)`
- `data/labels/labels_all.csv` — index of all windows with columns:
  `window_id, video_id, start_frame, end_frame, state, exercise, purity, numpy_path`

### Check
Look at the printed summary — make sure all three state classes (null, rest, active) appear and no single class is missing.

---

## Step 5 — Split into Train / Validation / Test

### What you do
Split the windows into three groups **by video**, not by frame.

### Why
If windows from the same video go into both train and test, the model "cheats" — it recognises the person's body shape rather than the exercise pattern. Splitting by video prevents this.

### How
```powershell
python training/split_dataset.py
```

### Output
- `data/labels/train.csv` — 70% of videos
- `data/labels/val.csv` — 15% of videos  
- `data/labels/test.csv` — 15% of videos

### Check
The printed table must show all three state classes present in all three splits.  
If a class is missing from val or test, collect more data for that class before continuing.

---

## Step 6 — Train the State Model on Google Colab

### What you do
Train a small **BiLSTM** model to classify each 32-frame window as `null`, `rest`, or `active`.

### Why
This is the core of Module 1. The state model decides whether a workout session is happening.  
Everything else (exercise recognition, live display) depends on this being correct first.

### How — Prepare data for Colab
1. Zip your `data/` folder and upload to Google Drive
2. Open a new Colab notebook
3. Mount your Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### How — Training script (to be built: `training/train_state_model.py`)
The script will:
- Load windows from `train.csv` and `val.csv`
- Build a BiLSTM: input `(batch, 32, 142)` → output `(batch, 3)`
- Train for 30–50 epochs with Adam optimizer and cross-entropy loss
- Track accuracy, macro F1, confusion matrix each epoch
- Save the best model to `models/best_state_model.pt`

### Recommended training settings

| Setting | Value |
|---------|-------|
| Model | BiLSTM (2 layers, hidden=128) |
| Input shape | (batch, 32, 142) |
| Output | 3 classes: null, rest, active |
| Batch size | 32 |
| Epochs | 30–50 |
| Optimizer | Adam, lr=0.001 |
| Loss | CrossEntropyLoss |
| Metrics | Accuracy, Macro F1, Confusion matrix |

### Output
- `models/best_state_model.pt` — saved model weights
- Training curves (loss and F1 per epoch)
- Confusion matrix printout

### What to look for
The most important confusion to check is **rest predicted as null** and **null predicted as rest**.  
If this is high, the model cannot tell the difference between resting and not being in a session.

---

## Step 7 — Evaluate the State Model

### What you do
Run the saved model on the validation and test sets.  
Print a confusion matrix and identify where it fails.

### Why
A model that looks good on training data might fail on new videos.  
You need to know exactly which cases it gets wrong before you trust it in the live app.

### How — Evaluation script (to be built: `training/evaluate_state_model.py`)
The script will:
- Load `models/best_state_model.pt`
- Run predictions on `val.csv` and `test.csv`
- Print accuracy, macro F1, and a confusion matrix
- Save a CSV with per-window predictions vs ground truth

### Output
- Printed metrics: accuracy, macro F1
- Confusion matrix (9 cells: 3 true labels × 3 predicted labels)
- `outputs/state_model_predictions.csv`

### Pass criteria before moving to Step 8
- Macro F1 on validation set > 0.75
- Active class F1 > 0.80 (most important)
- Rest vs null confusion is understood and acceptable

---

## Step 8 — Connect State Model to the Live App

### What you do
Replace the random placeholder in `src/state_rules.py` with real model inference.  
The state machine already exists — just feed it model probabilities instead of random numbers.

### Why
The state machine handles the timeout logic (rest → null after 5 seconds).  
The model handles the frame-level classification.  
Together they produce smooth, session-aware predictions.

### How — Update `src/state_rules.py`
```python
# Old: random numbers
# New: load model and run inference on the window
import torch
model = load_state_model('models/best_state_model.pt')

def predict_state(window):
    tensor = window_to_tensor(window)   # shape: (1, 32, 142)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    return {
        'null':   probs[0][0].item(),
        'rest':   probs[0][1].item(),
        'active': probs[0][2].item(),
    }
```

### Output
- `app.py` runs with real model predictions instead of random output
- State machine applies timeout on top of model probabilities
- Screen shows: `STATE: ACTIVE` or `STATE: REST` based on real inference

### Test
Run on a recorded squat video and verify:
- NULL while walking to the rack
- ACTIVE during squats
- REST after stopping
- NULL after 5 seconds with no new squats

---

## Step 9 — Train the Exercise Model on Google Colab

### What you do
Train a second classifier that runs **only when state = active**.  
It classifies which exercise is happening: squat / bicep_curl / shoulder_press / etc.

### Why
Once you know someone is exercising (active), you need to know which exercise.  
This model only sees `active` windows — null and rest windows are never passed to it.

### How — Training script (to be built: `training/train_exercise_model.py`)
- Filter `train.csv` to only rows where `state=active`
- Same BiLSTM architecture as state model
- Output: one class per exercise
- Save to `models/best_exercise_model.pt`

### Recommended training settings

| Setting | Value |
|---------|-------|
| Model | BiLSTM (same as state model) |
| Input | Only active windows from train.csv |
| Output | N classes (one per exercise) |
| Batch size | 32 |
| Epochs | 30–50 |
| Loss | CrossEntropyLoss |

### Output
- `models/best_exercise_model.pt`
- Confusion matrix per exercise class
- Per-exercise F1 scores

---

## Step 10 — Connect Exercise Model to the Live App

### What you do
Update `src/exercise_rules.py` to load `best_exercise_model.pt` and run real inference.

### Why
This completes the two-model pipeline.  
The screen now shows both what state the person is in AND which exercise they are doing.

### How — Update `src/exercise_rules.py`
```python
# Old: random exercise name
# New: real model inference, only runs when state=active
def predict_exercise(window, final_state):
    if final_state != 'active':
        return None
    tensor = window_to_tensor(window)
    with torch.no_grad():
        probs = torch.softmax(exercise_model(tensor), dim=1)
    class_idx = probs.argmax().item()
    return config.TARGET_EXERCISES[class_idx]
```

### Output
- Full live pipeline: video → pose → features → state model → exercise model → display
- Screen shows: `STATE: ACTIVE | Exercise: squat`
- `models/` folder has both `.pt` files committed to the project

### Test
Run `python app.py` on a recorded video with multiple exercises.  
Verify state changes are correct and exercise labels match what is being done.

---

## Step 11 — Full Demo and Review

### What you do
Record a clean demo video showing the live system working end to end.

### Demo should show
1. Person walks into frame → screen shows `NULL`
2. Person starts squatting → screen shows `ACTIVE | squat`
3. Person rests → screen shows `REST`
4. 5 seconds pass with no new squats → screen shows `NULL`
5. Person does bicep curls → screen shows `ACTIVE | bicep_curl`

### Output
- A recorded demo video
- `models/best_state_model.pt` committed to repo
- `models/best_exercise_model.pt` committed to repo
- `verify_pipeline.py` still passes all 5 steps

---

## Summary — What to Build Next (In Order)

| Priority | What to build | Where |
|----------|--------------|-------|
| 1 | Collect null data (Charades) + extract landmarks | Manual + `app.py` |
| 2 | Collect active/squat data (Penn Action) + extract landmarks | Manual + `app.py` |
| 3 | Record rest + other exercises yourself | Camera + `app.py` |
| 4 | Write all annotation CSVs | Text editor |
| 5 | `python training/build_dataset.py` | Terminal |
| 6 | `python training/split_dataset.py` | Terminal |
| 7 | Write `training/train_state_model.py` | Code (Colab) |
| 8 | Write `training/evaluate_state_model.py` | Code (Colab) |
| 9 | Update `src/state_rules.py` to use real model | Code |
| 10 | Write `training/train_exercise_model.py` | Code (Colab) |
| 11 | Update `src/exercise_rules.py` to use real model | Code |
| 12 | Record demo video | Camera |

---

## Files That Need to Be Built (Don't Exist Yet)

| File | What it does |
|------|-------------|
| `training/train_state_model.py` | BiLSTM training for null/rest/active |
| `training/evaluate_state_model.py` | Confusion matrix + per-window predictions |
| `training/train_exercise_model.py` | BiLSTM training for exercise classification |
| `models/best_state_model.pt` | Saved state model weights (after training) |
| `models/best_exercise_model.pt` | Saved exercise model weights (after training) |

---

## Files That Already Exist and Are Ready

| File | Status |
|------|--------|
| `app.py` | ✅ Ready — just needs model loaded in |
| `config.py` | ✅ Ready |
| `src/video_io.py` | ✅ Ready |
| `src/pose_extractor.py` | ✅ Ready |
| `src/features.py` | ✅ Ready |
| `src/data_export.py` | ✅ Ready |
| `src/state_machine.py` | ✅ Ready |
| `src/state_rules.py` | ⚠️ Placeholder — replace with model |
| `src/exercise_rules.py` | ⚠️ Placeholder — replace with model |
| `src/overlay.py` | ✅ Ready |
| `verify_pipeline.py` | ✅ Ready |
| `training/build_dataset.py` | ✅ Ready |
| `training/split_dataset.py` | ✅ Ready |
| `training/validate_annotations.py` | ✅ Ready |
| `training/count_frames.py` | ✅ Ready |
| `data/annotations/template_annotation.csv` | ✅ Ready |
| `data/README.md` | ✅ Ready |
| `docs/v1_label_design.md` | ✅ Ready |
