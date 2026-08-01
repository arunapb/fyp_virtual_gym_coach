import os
import torch
import numpy as np
from src.state_rules import _features_to_vector
import torch.nn as nn

EXERCISE_MAP = {0: 'squat', 1: 'bicep_curl', 2: 'shoulder_press', 3: 'deadlift'}

# ── Model Architecture ────────────────────────────────────────────────────────
class ExerciseBiLSTM(nn.Module):
    def __init__(self, input_dim=142, hidden_dim=64, num_layers=2, num_classes=4):
        super(ExerciseBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Load Model Once (shared, read-only after eval()) ───────────────────────────
_model = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _load_exercise_model():
    global _model
    if _model is not None:
        return _model

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_exercise_model.pt')
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        print(f"[exercise_rules] WARNING: Model not found at {model_path}. Using fallback.")
        return None

    model = ExerciseBiLSTM().to(_device)
    model.load_state_dict(torch.load(model_path, map_location=_device))
    model.eval()
    _model = model
    print(f"[exercise_rules] BiLSTM loaded from {model_path}")
    return _model


class ExercisePredictor:
    """
    Classifies the specific exercise being performed from a 32-frame window,
    and applies the biomechanical expert-system masking + session lock /
    ghost-memory logic described in project_context.md.

    All lock/ghost state lives on the instance (not module globals) so that
    multiple concurrent sessions (e.g. separate WebSocket connections) never
    interfere with each other.
    """

    def __init__(self):
        self._last_valid_exercise = None
        self._proposed_exercise = None
        self._proposed_count = 0
        self._ghost_exercise = None
        self._frames_in_rest = 0

    def predict(self, window, current_state, raw_state_probs=None):
        if current_state != 'active':
            if self._last_valid_exercise is not None:
                self._ghost_exercise = self._last_valid_exercise
                self._frames_in_rest = 0

            self._last_valid_exercise = None
            self._proposed_exercise = None
            self._proposed_count = 0

            if self._ghost_exercise is not None:
                self._frames_in_rest += 1
                if self._frames_in_rest > 900:  # 30 seconds
                    self._ghost_exercise = None

            return None

        if len(window) < 32:
            return None

        model = _load_exercise_model()
        if model is None:
            return 'unknown_active'

        # ---------------------------------------------------------
        # EXPERT SYSTEM: BIOMECHANICAL MASKING
        # ---------------------------------------------------------
        min_hip_y = 999.0
        max_hip_y = -999.0
        leg_energy = 0.0
        max_wrist_above_shoulder = 0.0  # Track how high wrists go above shoulders

        # Points 23, 24 = Hips | 25, 26 = Knees | 27, 28 = Ankles
        leg_indices = [23, 24, 25, 26, 27, 28]
        valid_frames_count = 0
        recent_history = list(window)[-32:]

        for i in range(1, len(recent_history)):
            curr_feat = recent_history[i]
            prev_feat = recent_history[i - 1]

            # Skip frames where no pose was detected
            if curr_feat is None:
                continue

            curr_lms = curr_feat.get('normalized_landmarks')
            prev_lms = prev_feat.get('normalized_landmarks') if prev_feat is not None else None

            raw_hip_y = curr_feat.get('raw_hip_y')
            if raw_hip_y is not None:
                if raw_hip_y < min_hip_y: min_hip_y = raw_hip_y
                if raw_hip_y > max_hip_y: max_hip_y = raw_hip_y
                valid_frames_count += 1

            # Track how high wrists go relative to shoulders (lower y = higher position)
            if curr_lms is not None:
                avg_shoulder_y = (curr_lms[11]['y'] + curr_lms[12]['y']) / 2.0
                for wrist_idx in [15, 16]:  # Left and right wrist
                    wrist_above = avg_shoulder_y - curr_lms[wrist_idx]['y']  # Positive = above shoulder
                    if wrist_above > max_wrist_above_shoulder:
                        max_wrist_above_shoulder = wrist_above

            if curr_lms is not None and prev_lms is not None:
                for idx in leg_indices:
                    dx = curr_lms[idx]['x'] - prev_lms[idx]['x']
                    dy = curr_lms[idx]['y'] - prev_lms[idx]['y']
                    dz = curr_lms[idx]['z'] - prev_lms[idx]['z']
                    leg_energy += np.sqrt(dx**2 + dy**2 + dz**2)

        hip_y_drop = max_hip_y - min_hip_y if valid_frames_count > 0 else 0.0

        sequence = []
        for item in window:
            if item is None:
                sequence.append(np.zeros(142))
                continue

            flat_lms = []
            for lm in item['normalized_landmarks']:
                flat_lms.extend([lm['x'], lm['y'], lm['z'], lm['visibility']])

            angle_keys = ['l_knee', 'r_knee', 'l_hip', 'r_hip',
                          'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'trunk']
            flat_angles = [item['angles'].get(k, 0.0) for k in angle_keys]

            frame_features = flat_lms + flat_angles + [item['motion_energy']]
            sequence.append(frame_features)

        X = torch.tensor(np.array([sequence]), dtype=torch.float32).to(_device)

        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()

        mask_rule = "None"

        # RULE 1: If legs are stationary and hips don't drop, it CANNOT be a Squat or Deadlift!
        if leg_energy < 0.15 or hip_y_drop < 0.015:
            probabilities[0] = 0.0  # Block squat
            probabilities[3] = 0.0  # Block deadlift
            mask_rule = "BLOCKED SQUAT+DEADLIFT (No Leg Movement)"

        # RULE 2: If legs are moving massively and hips drop, it CANNOT be Bicep Curl or Shoulder Press!
        elif leg_energy > 0.8 and hip_y_drop > 0.08:
            probabilities[1] = 0.0  # Block bicep_curl
            probabilities[2] = 0.0  # Block shoulder_press
            mask_rule = "LOCKED LOWER-BODY (Massive Leg Movement)"

        # RULE 3: Wrist height check — distinguish Bicep Curl vs Shoulder Press
        # If wrists NEVER went above shoulders → cannot be shoulder press
        if max_wrist_above_shoulder < 0.15:
            probabilities[2] = 0.0  # Block shoulder_press
            mask_rule = mask_rule + " + BLOCKED SHOULDER_PRESS (Wrists Below Shoulders)"
        # If wrists went well ABOVE shoulders (overhead) → cannot be bicep curl
        elif max_wrist_above_shoulder > 0.5:
            probabilities[1] = 0.0  # Block bicep_curl
            mask_rule = mask_rule + " + BLOCKED BICEP_CURL (Wrists Overhead)"

        pred_idx = np.argmax(probabilities)
        confidence = float(probabilities[pred_idx]) * 100.0

        scores_dict = {
            'squat': float(probabilities[0]) * 100.0,
            'bicep_curl': float(probabilities[1]) * 100.0,
            'shoulder_press': float(probabilities[2]) * 100.0,
            'deadlift': float(probabilities[3]) * 100.0
        }

        # ---------------------------------------------------------
        # PAUSE VERIFICATION
        # ---------------------------------------------------------
        # If the state machine is still in ACTIVE (due to timeout delay),
        # but the neural network strongly believes the person is RESTING (or NULL),
        # we should pause exercise verification and NOT break any session locks.
        is_pausing = False
        if raw_state_probs is not None:
            if raw_state_probs.get('rest', 0.0) > 0.6 or raw_state_probs.get('null', 0.0) > 0.6:
                is_pausing = True

        if is_pausing:
            if self._last_valid_exercise is not None:
                return {
                    'exercise': self._last_valid_exercise,
                    'confidence': 0.0,
                    'scores': scores_dict,
                    'hip_drop': hip_y_drop,
                    'leg_energy': leg_energy,
                    'mask_rule': f"{mask_rule} + PAUSING SET (Keeping {self._last_valid_exercise})"
                }
            else:
                return None

        # If all probabilities were masked to 0.0, no valid exercise is detected
        if np.max(probabilities) == 0.0:
            self._proposed_exercise = None
            self._proposed_count = 0
            if self._last_valid_exercise is not None:
                self._last_valid_exercise = None
            return {
                'exercise': None,
                'confidence': 0.0,
                'scores': scores_dict,
                'hip_drop': hip_y_drop,
                'leg_energy': leg_energy,
                'mask_rule': f"{mask_rule} + NO VALID EXERCISE"
            }

        # STRICT SESSION LOCK: If we already locked in an exercise for this set, KEEP IT!
        # Exception: If physics mask rules zeroed out the locked exercise's probability
        # (e.g. Squat locked during static start, but no leg movement), break the false lock!
        if self._last_valid_exercise is not None:
            locked_idx = [k for k, v in EXERCISE_MAP.items() if v == self._last_valid_exercise][0]
            if probabilities[locked_idx] == 0.0:
                self._last_valid_exercise = None
                self._proposed_exercise = None
                self._proposed_count = 0
            else:
                return {
                    'exercise': self._last_valid_exercise,
                    'confidence': confidence,
                    'scores': scores_dict,
                    'hip_drop': hip_y_drop,
                    'leg_energy': leg_energy,
                    'mask_rule': f"{mask_rule} + SESSION LOCKED ({self._last_valid_exercise.upper()})"
                }

        # RULE 4: Initial Lock Confidence
        # We require at least 50% confidence (or a Physics Lock) to start the session lock!
        if confidence < 50.0 and "LOCKED" not in mask_rule:
            self._proposed_exercise = None
            self._proposed_count = 0
            return {
                'exercise': None,
                'confidence': confidence,
                'scores': scores_dict,
                'hip_drop': hip_y_drop,
                'leg_energy': leg_energy,
                'mask_rule': f"{mask_rule} + WAITING FOR HIGH CONFIDENCE"
            }

        # Start Verification Phase!
        pred_exercise = EXERCISE_MAP[pred_idx]

        if self._proposed_exercise == pred_exercise:
            self._proposed_count += 1
        else:
            self._proposed_exercise = pred_exercise
            self._proposed_count = 1

        target_frames = 15
        if self._ghost_exercise is not None and self._proposed_exercise != self._ghost_exercise:
            target_frames = 90  # 3 seconds of verification if switching exercises!

        if self._proposed_count >= target_frames:
            # Lock it in permanently!
            self._last_valid_exercise = pred_exercise
            self._ghost_exercise = None  # Clear ghost memory once locked
            return {
                'exercise': pred_exercise,
                'confidence': confidence,
                'scores': scores_dict,
                'hip_drop': hip_y_drop,
                'leg_energy': leg_energy,
                'mask_rule': f"{mask_rule} + INITIAL LOCK TRIGGERED"
            }
        else:
            # Still verifying - output VERIFYING to the UI
            return {
                'exercise': "VERIFYING...",
                'confidence': confidence,
                'scores': scores_dict,
                'hip_drop': hip_y_drop,
                'leg_energy': leg_energy,
                'mask_rule': f"{mask_rule} + VERIFYING ({self._proposed_count}/{target_frames})"
            }


# ── Backward-compatible module-level function (used by app.py) ────────────────
# app.py is a single-session desktop CLI script, so a single shared default
# instance is fine there. Multi-session server code (src/session.py) must
# instantiate its own ExercisePredictor() per connection instead.
_default_predictor = ExercisePredictor()

def predict_exercise(window, current_state, raw_state_probs=None):
    return _default_predictor.predict(window, current_state, raw_state_probs)
