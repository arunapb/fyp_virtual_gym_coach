import cv2
import config

def get_video_capture(source=0):
    """
    Open a video capture object.
    :param source: 0 for webcam, or a string path to a video file.
    :return: cv2.VideoCapture object
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
    return cap

def read_frame(cap):
    """
    Read a frame and resize it to consistent dimensions.
    """
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
    return success, frame

def release_capture(cap):
    """Cleanly release the video capture object."""
    cap.release()
