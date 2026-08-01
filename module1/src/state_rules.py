import os
import numpy as np
import torch
import torch.nn as nn

# ── Model Architecture (must match train_state_model.py exactly) ──────────────
class StateBiLSTM(nn.Module):
    def __init__(self, input_dim=142, hidden_dim=64, num_layers=2, num_classes=3):
        super(StateBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        lstm_out_dim = hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_frame_out = out[:, -1, :]
        return self.fc(last_frame_out)


# ── Load Model Once (on first import) ────────────────────────────────────────
STATE_MAP = {0: 'null', 1: 'rest', 2: 'active'}
_model = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _load_model():
    global _model
    if _model is not None:
        return _model

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_state_model.pt')
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print(f"[state_rules] WARNING: Model not found at {model_path}. Using null fallback.")
        return None

    model = StateBiLSTM().to(_device)
    model.load_state_dict(torch.load(model_path, map_location=_device))
    model.eval()
    _model = model
    print(f"[state_rules] BiLSTM model loaded from {model_path}")
    return _model


# ── Public Function ───────────────────────────────────────────────────────────
def _features_to_vector(frame_features):
    """
    Converts one frame's feature dict into a flat 142-dim numpy array.
    Format must match training: [132 landmark values] + [9 angles] + [1 motion energy]
    """
    import numpy as np

    if frame_features is None:
        return np.zeros(142, dtype=np.float32)

    vec = []

    # 1. Normalized landmarks: 33 joints × 4 values (x, y, z, visibility) = 132
    norm_skel = frame_features.get('normalized_landmarks')
    if norm_skel:
        for lm in norm_skel:
            vec.extend([lm['x'], lm['y'], lm['z'], lm['visibility']])
    else:
        vec.extend([0.0] * 132)

    # 2. Joint angles: 9 values
    angles = frame_features.get('angles', {})
    for key in ['l_knee', 'r_knee', 'l_hip', 'r_hip',
                'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'trunk']:
        vec.append(float(angles.get(key, 0.0)))

    # 3. Motion energy: 1 value
    vec.append(float(frame_features.get('motion_energy', 0.0)))

    return np.array(vec, dtype=np.float32)  # shape: (142,)


def predict_state(window):
    """
    Given a list of feature dicts (one per frame),
    returns a probability dict: {'null': float, 'rest': float, 'active': float}
    """
    if len(window) < 32:
        return {'null': 1.0, 'rest': 0.0, 'active': 0.0}

    model = _load_model()
    if model is None:
        return {'null': 1.0, 'rest': 0.0, 'active': 0.0}

    # Take the last 32 frames and convert each to a 142-dim vector
    frames = list(window)[-32:]
    arr = np.array([_features_to_vector(f) for f in frames], dtype=np.float32)  # (32, 142)
    tensor = torch.tensor(arr).unsqueeze(0).to(_device)                          # (1, 32, 142)

    with torch.no_grad():
        logits = model(tensor)                                                    # (1, 3)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    return {
        'null':   float(probs[0]),
        'rest':   float(probs[1]),
        'active': float(probs[2])
    }
