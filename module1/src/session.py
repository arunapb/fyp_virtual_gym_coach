import base64
import time
import cv2
import numpy as np

import config
from src.pose_extractor import PoseExtractor
from src.features import (
    normalize_skeleton,
    update_sliding_window,
    extract_body_angles,
    calculate_motion_energy
)
from src.state_rules import predict_state
from src.state_machine import WorkoutStateMachine
from src.exercise_rules import ExercisePredictor

# Exercise labels that represent a real, confirmed exercise lock (as opposed
# to None / "VERIFYING..." / the no-model fallback "unknown_active").
_CONFIRMED_LABELS = set(config.TARGET_EXERCISES)

# If no frame arrives for longer than this, don't count the gap as active
# exercise time (covers network stalls / paused uploads / etc).
_MAX_FRAME_GAP_SEC = 1.0


class GymCoachSession:
    """
    Per-connection pipeline: wraps everything app.py's main loop does per
    frame (pose extraction -> features -> sliding window -> state model ->
    state machine -> exercise model) behind a single process_frame() call,
    so a WebSocket handler can feed it frames one at a time regardless of
    whether they came from a browser webcam or a server-side video file.

    Owns its own PoseExtractor / WorkoutStateMachine / ExercisePredictor
    instances so concurrent sessions never share mutable state.
    """

    def __init__(self):
        self.pose_ext = PoseExtractor()
        self.state_machine = WorkoutStateMachine()
        self.exercise_predictor = ExercisePredictor()

        self.sliding_window = []
        self.prev_landmarks = None

        self.frame_index = 0
        self.start_time = time.time()
        self._last_frame_time = None

        # exercise name -> accumulated active seconds
        self.exercise_durations = {}
        # accumulated seconds spent in the 'rest' state this session
        self.rest_time = 0.0

    def process_frame_bytes(self, jpeg_bytes, include_preview=False):
        """Decode a JPEG-encoded frame (e.g. from a WebSocket binary message)
        and run it through the pipeline."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return self.process_frame(frame, include_preview=include_preview)

    def process_frame(self, frame, include_preview=False):
        """Run one BGR frame through the full pipeline and return the
        JSON-serializable state update dict.

        If include_preview is True, the returned dict also carries a
        base64-encoded JPEG ("frame") of the (skeleton-annotated) frame that
        was just analyzed, so a browser client can render it directly
        instead of relying on native <video> playback of an arbitrary
        uploaded file (which may use a codec/container the browser's own
        demuxer can't handle even though OpenCV decoded it fine)."""
        now = time.time()
        frame = cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))

        landmarks_list, frame = self.pose_ext.process_frame(frame, draw=include_preview)
        self.frame_index += 1

        if landmarks_list:
            norm_skel = normalize_skeleton(landmarks_list)
            angles = extract_body_angles(landmarks_list)
            motion_energy = calculate_motion_energy(self.prev_landmarks, landmarks_list)
            self.prev_landmarks = landmarks_list

            features = {
                'normalized_landmarks': norm_skel,
                'angles': angles,
                'motion_energy': motion_energy,
                'raw_hip_y': (landmarks_list[23]['y'] + landmarks_list[24]['y']) / 2.0
            }
            self.sliding_window = update_sliding_window(self.sliding_window, features, config.WINDOW_SIZE)
        else:
            self.sliding_window = update_sliding_window(self.sliding_window, None, config.WINDOW_SIZE)
            self.prev_landmarks = None

        raw_state_probs = predict_state(self.sliding_window)
        final_state = self.state_machine.update(raw_state_probs)
        exercise_data = self.exercise_predictor.predict(self.sliding_window, final_state, raw_state_probs)

        raw_label = exercise_data['exercise'] if exercise_data else None
        confidence = exercise_data['confidence'] if exercise_data else 0.0
        confirmed_exercise = raw_label if raw_label in _CONFIRMED_LABELS else None

        # Wall-clock based duration tracking (robust to variable frame rates
        # from a browser webcam, unlike a fixed frames/30.0 assumption).
        dt = 0.0
        if self._last_frame_time is not None:
            dt = min(now - self._last_frame_time, _MAX_FRAME_GAP_SEC)
        self._last_frame_time = now

        if final_state == 'active' and confirmed_exercise is not None:
            self.exercise_durations[confirmed_exercise] = (
                self.exercise_durations.get(confirmed_exercise, 0.0) + dt
            )
        elif final_state == 'rest':
            self.rest_time += dt

        result = {
            "type": "state_update",
            "state": final_state,
            "exercise": confirmed_exercise,
            "label": raw_label,
            "confidence": round(confidence, 1),
            "elapsed_active_sec": round(self.exercise_durations.get(confirmed_exercise, 0.0), 2)
                if confirmed_exercise else 0.0,
            "state_probs": {k: round(v, 3) for k, v in raw_state_probs.items()},
            "frame_index": self.frame_index,
        }

        if include_preview:
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                result["frame"] = base64.b64encode(buf).decode('ascii')

        return result

    def get_summary(self):
        return {
            "durations": {k: round(v, 2) for k, v in self.exercise_durations.items()},
            "rest_time": round(self.rest_time, 2),
            "total_time": round(time.time() - self.start_time, 2),
        }
