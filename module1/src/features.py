import math

class SlidingWindowBuffer:
    """
    A buffer that collects frames and groups them into discrete 'windows'.
    Used for creating cleanly separated blocks of frames with a specific overlap.
    """
    def __init__(self, window_size=32, overlap_percent=0.5):
        self.window_size = window_size
        # Step size is how many frames we slide forward after returning a window
        self.step_size = int(window_size * (1.0 - overlap_percent))
        self.buffer = []
        
    def add_frame(self, features):
        """
        Adds a frame. If the buffer hits the window_size, it returns the 
        full 32-frame window and then slides forward by the step_size.
        """
        self.buffer.append(features)
        
        if len(self.buffer) >= self.window_size:
            # Make a copy of the exact 32 frames to return
            full_window = list(self.buffer[:self.window_size])
            
            # Slide the window forward (e.g. drop the oldest 16 frames for 50% overlap)
            self.buffer = self.buffer[self.step_size:]
            
            return full_window
            
        # Return None until we have gathered at least 32 frames
        return None

def update_sliding_window(window, new_features, window_size):
    """
    Add the newest frame's features to the sliding window.
    Removes the oldest frame if the window gets too large.
    """
    window.append(new_features)
    
    if len(window) > window_size:
        window.pop(0) # Remove oldest frame
        
    return window

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points a, b, c, where b is the middle joint.
    We use the 2D (x, y) coordinates for a simple angle.
    
    Math explanation for beginners:
    1. We use the 'atan2' function, which gives us the angle of a line relative to the x-axis.
    2. We subtract the angles of the two lines (line from b to c, and line from b to a).
    3. We convert the result from radians (math speak) to degrees (human speak).
    """
    # Grab the x and y coordinates of each point
    ax, ay = a['x'], a['y']
    bx, by = b['x'], b['y']
    cx, cy = c['x'], c['y']
    
    # Calculate radians using the inverse tangent function
    # math.atan2(y, x) computes the angle in radians between the positive x-axis and the point.
    radians = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    
    # Convert radians to degrees (1 radian = ~57.3 degrees)
    angle = math.degrees(radians)
    
    # Angles can be negative. We want the absolute angle inside the joint (0 to 360).
    angle = abs(angle)
    
    # If the angle is > 180, we want the smaller inner angle, so we subtract from 360
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

def normalize_skeleton(landmarks):
    """
    Normalizes the skeleton making it relative to the person's body rather than the camera view.
    
    Math explanation for beginners:
    1. Hip Center (Translation): We find the exact middle point between the left and right hip. 
       We subtract this point from all joints. This "moves" the skeleton so the hips are always 
       at the center of our graph (0,0), regardless of where the person stands in the camera.
    2. Torso Size (Scaling): We measure the distance from the hips to the shoulders. 
       We divide all joint coordinates by this distance. This makes the skeleton the exact same 
       size whether the person is close to the camera or far away.
    """
    if not landmarks:
        return None
        
    # MediaPipe Indices: 23=left hip, 24=right hip, 11=left shoulder, 12=right shoulder
    l_hip, r_hip = landmarks[23], landmarks[24]
    l_shldr, r_shldr = landmarks[11], landmarks[12]
    
    # Calculate the center of the hips
    hip_center_x = (l_hip['x'] + r_hip['x']) / 2.0
    hip_center_y = (l_hip['y'] + r_hip['y']) / 2.0
    
    # Calculate the center of the shoulders
    shldr_center_x = (l_shldr['x'] + r_shldr['x']) / 2.0
    shldr_center_y = (l_shldr['y'] + r_shldr['y']) / 2.0
    
    # Calculate torso length using the Pythagorean distance formula: sqrt(a^2 + b^2)
    dx = shldr_center_x - hip_center_x
    dy = shldr_center_y - hip_center_y
    torso_length = math.sqrt(dx**2 + dy**2)
    
    # Avoid dividing by zero if the torso length is strangely tiny
    if torso_length < 0.001:
        torso_length = 0.001
        
    normalized_landmarks = []
    
    for lm in landmarks:
        # Subtract hip center (Translate feature) and divide by torso length (Scale feature)
        norm_x = (lm['x'] - hip_center_x) / torso_length
        norm_y = (lm['y'] - hip_center_y) / torso_length
        norm_z = lm['z'] / torso_length 
        
        normalized_landmarks.append({
            'x': norm_x,
            'y': norm_y,
            'z': norm_z,
            'visibility': lm['visibility']
        })
        
    return normalized_landmarks

def calculate_motion_energy(prev_landmarks, curr_landmarks):
    """
    Calculates the 'energy' or 'amount of movement' between two frames.
    
    Math explanation for beginners:
    1. We want a single number that tells us "how much did the person move?"
    2. We look at every single joint one by one.
    3. For each joint, we calculate how far it traveled since the last frame.
    4. We add up all these tiny distances. High total = fast movement, low total = staying still.
    """
    if not prev_landmarks or not curr_landmarks:
        return 0.0
        
    total_energy = 0.0
    
    for i in range(len(curr_landmarks)):
        p1 = prev_landmarks[i]
        p2 = curr_landmarks[i]
        
        # Calculate the distance between the old location and new location
        dist = math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
        total_energy += dist
        
    return total_energy

def extract_body_angles(landmarks):
    """
    Calculates specific joint angles that are important for exercise form.
    MediaPipe Index Map: Shoulder(11,12), Elbow(13,14), Wrist(15,16), Hip(23,24), Knee(25,26), Ankle(27,28)
    """
    if not landmarks:
        return {}
        
    # Calculate Left Angles
    left_elbow = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
    left_shoulder = calculate_angle(landmarks[23], landmarks[11], landmarks[13])
    left_hip = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
    left_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
    
    # Calculate Right Angles
    right_elbow = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
    right_shoulder = calculate_angle(landmarks[24], landmarks[12], landmarks[14])
    right_hip = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
    right_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
    
    # Calculate Trunk Angle (How much the body is leaning relative to a perfect straight vertical line)
    # We create a fake "straight up" point above the hips to measure against
    hip_center = {
        'x': (landmarks[23]['x'] + landmarks[24]['x']) / 2.0,
        'y': (landmarks[23]['y'] + landmarks[24]['y']) / 2.0
    }
    shoulder_center = {
        'x': (landmarks[11]['x'] + landmarks[12]['x']) / 2.0,
        'y': (landmarks[11]['y'] + landmarks[12]['y']) / 2.0
    }
    # Create the vertical point directly above the hip (decrease Y because Y is upside down in images)
    vertical_point = {'x': hip_center['x'], 'y': hip_center['y'] - 0.1}
    
    trunk_angle = calculate_angle(shoulder_center, hip_center, vertical_point)

    return {
        'l_elbow': left_elbow, 'r_elbow': right_elbow,
        'l_shoulder': left_shoulder, 'r_shoulder': right_shoulder,
        'l_hip': left_hip, 'r_hip': right_hip,
        'l_knee': left_knee, 'r_knee': right_knee,
        'trunk': trunk_angle
    }
