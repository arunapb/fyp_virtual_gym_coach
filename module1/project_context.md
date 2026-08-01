# Virtual Gym Coach (Module 1) - Project Context

This document contains the complete context, architecture, logic, and fixes for **Module 1: Exercise Recognition** of the Virtual Gym Coach project. It is designed to be fed to another LLM so it can perfectly understand the codebase, the heuristics, and the specific ML implementations we have developed.

## 1. Project Overview
**Goal:** Build an intelligent virtual gym coach that tracks a user's workout state in real-time and classifies specific resistance training exercises using a standard webcam. 
**Supported Exercises:** Squat, Bicep Curl, Shoulder Press, Deadlift.
**Tech Stack:** Python, PyTorch (BiLSTM), MediaPipe (Pose), OpenCV, Scikit-learn.

## 2. Data Pipeline & Feature Extraction
Instead of passing raw images to a CNN, the system uses MediaPipe to extract human pose landmarks, which are converted into a rich 142-dimensional numerical feature vector per frame.

**Feature Vector (142 dimensions):**
*   `[0:132]`: Normalized landmarks (x, y, z, visibility) for all 33 MediaPipe joints.
*   `[132:141]`: 9 Specific Joint Angles in degrees. **Crucial note:** The exact order must be preserved across training and inference: `['l_knee', 'r_knee', 'l_hip', 'r_hip', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'trunk']`.
*   `[141]`: Motion Energy (a single scalar representing the total euclidean distance all joints moved since the previous frame).

**Temporal Sliding Window:**
Because exercises are sequential movements, the system slides a **32-frame window** across the video (at 30 FPS). The models process a tensor of shape `(Batch, 32, 142)`.

## 3. Dual BiLSTM Architecture
The system utilizes two separate Bidirectional LSTM networks:
1.  **State Model (`train_state_model.py`):** Classifies the user's broad state into `null` (nobody in frame), `rest` (standing still), or `active` (exercising).
2.  **Exercise Model (`train_exercise_model.py`):** Runs *only* when the user is `active`. Classifies the exact movement into `squat`, `bicep_curl`, `shoulder_press`, or `deadlift`.

## 4. The Biomechanical Expert System (`exercise_rules.py`)
Standard neural networks suffer from Out-of-Distribution (OOD) errors (e.g., misclassifying an ear-scratch as a bicep curl). To enforce real-world kinesiology constraints, we implemented a rigorous Expert System that masks out physically impossible neural network predictions.

*   **Rule 1 (Lower-Body Constraints):** If total leg energy is `< 0.15` and vertical hip drop is `< 0.015` over the 32 frames, the system forces the probabilities of `Squat` and `Deadlift` to `0.0`.
*   **Rule 3 (Upper-Body Constraints):** If wrists never exceed the shoulder height, `Shoulder Press` is forced to `0.0`. If wrists go entirely overhead, `Bicep Curl` is forced to `0.0`.

## 5. The State Machine & Ghost Memory System
To provide a smooth, flicker-free User Interface, the raw Neural Network probabilities are post-processed through a State Machine.

*   **Inter-Repetition Pauses:** When a user stops moving, the State AI outputs `Rest: 100%`. However, the State Machine delays the UI transition to `REST` by **2 seconds (60 frames)**. This prevents the set from prematurely ending when a user pauses to breathe at the top of a squat.
*   **Ghost Memory & Verification Thresholds:** 
    *   Normally, identifying an exercise requires **0.5 seconds (15 frames)** of continuous prediction.
    *   When a set finishes, a "Ghost Memory" of the locked exercise is preserved for **30 seconds (900 frames)**.
    *   If a *new* exercise is detected during this 30-second rest window, the system becomes highly skeptical (to filter out random arm movements). It demands **5.0 seconds (150 frames)** of continuous prediction before allowing the switch.

## 6. Critical Bugs Fixed During Development
*   **Scrambled Feature Arrays:** Discovered that the angle extraction order in live inference (`state_rules.py` and `exercise_rules.py`) was misaligned with the training script (`build_dataset.py`), and included a typo (`'torso'` instead of `'trunk'`). This caused the live AI to receive garbage features and output `100% NULL`. Fixed by hard-syncing the angle array arrays.
*   **False Lock Breaking during Rest:** The system was mistakenly destroying the exercise "Session Lock" when the user was resting. Fixed by passing `raw_state_probs` directly to `predict_exercise`. If the system knows the user is resting (Rest/Null > 60%), it actively halts exercise verification and preserves the lock.

## 7. Model Evaluation
Evaluated on a holdout test set of unseen video sequences:
*   **Accuracy:** 99% macro-average accuracy.
*   **Precision/Recall:** Precision of 1.00 for Squats and Deadlifts (zero false positives). Recall of 1.00 for upper-body exercises (zero false negatives).
*   **ROC-AUC:** All four exercise classes achieved an Area Under the Curve (AUC) approximating 1.00, demonstrating near-perfect discriminative capability on the test dataset.
