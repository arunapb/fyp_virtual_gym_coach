import cv2
import mediapipe as mp

class PoseExtractor:
    def __init__(self):
        """Initialize MediaPipe Pose."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame, draw=False):
        """
        Takes a frame, runs MediaPipe Pose, and returns the 33 landmarks.
        
        MediaPipe outputs 33 3D landmarks. Each landmark contains:
        - x and y: Normalized image coordinates (0.0 to 1.0).
        - z: Estimated depth (smaller is closer to camera).
        - visibility: A value (0.0 to 1.0) indicating how likely the joint is visible.
        
        Returns:
            landmarks_list: A list of 33 dictionaries if pose is found, else None.
            frame: The annotated frame if draw=True, else the original frame.
        """
        # Convert the BGR image to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark image as not writeable
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        landmarks_list = None

        if results.pose_landmarks:
            landmarks_list = []
            # Extract exactly 33 landmarks
            for landmark in results.pose_landmarks.landmark:
                landmarks_list.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            # Optionally draw the skeleton on the frame
            if draw:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
        return landmarks_list, frame
