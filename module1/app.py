import cv2
import argparse
import config
from src.video_io import get_video_capture, read_frame, release_capture
from src.pose_extractor import PoseExtractor
from src.features import (
    normalize_skeleton,
    update_sliding_window,
    extract_body_angles,
    calculate_motion_energy
)
from src.state_rules import predict_state
from src.state_machine import WorkoutStateMachine
from src.exercise_rules import predict_exercise
from src.overlay import draw_info
from src.data_export import save_landmarks_to_csv
import time


def parse_args():
    """
    Parse command-line arguments.

    Examples
    --------
    # Use the default webcam (index 0):
        python app.py

    # Use webcam at a specific index:
        python app.py --source 1

    # Use a sample video file:
        python app.py --source data/sample_videos/squat_demo.mp4

    # Run without saving the landmarks CSV at the end:
        python app.py --no-save
    """
    parser = argparse.ArgumentParser(
        description='Virtual Gym Coach — Module 1 Prototype'
    )
    parser.add_argument(
        '--source',
        default='0',
        help=(
            'Video source. Use "0" (or any digit) for a webcam index, '
            'or a file path such as data/sample_videos/squat.mp4. '
            'Default: 0 (built-in webcam).'
        )
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving the landmarks CSV when the session ends.'
    )
    args = parser.parse_args()

    # Convert source to int if it is a digit string (webcam index)
    if args.source.isdigit():
        args.source = int(args.source)

    return args

def main():
    args = parse_args()

    source_label = f"webcam (index {args.source})" if isinstance(args.source, int) else args.source
    print(f"Initializing Virtual Gym Coach — Module 1 Prototype")
    print(f"Source : {source_label}")
    print(f"Save CSV: {'no' if args.no_save else 'yes (outputs/landmarks.csv)'}")
    print("Press 'q' in the video window to quit.\n")

    # 1. Initialize Objects
    cap = get_video_capture(args.source)
    pose_ext = PoseExtractor()
    state_machine = WorkoutStateMachine()
    
    # Empty list to hold our sliding window of features
    sliding_window = []
    
    # List to collect landmarks for CSV export
    collected_landmarks = []
    frame_index = 0
    start_time = time.time()
    prev_landmarks = None
    
    # Track the number of frames spent performing each exercise
    exercise_durations_frames = {}

    while True:
        # 2. Get Video Frame
        success, frame = read_frame(cap)
        if not success:
            break
            
        # 3. Extract Pose Landmarks (MediaPipe)
        # We pass draw=True to optionally draw the skeleton
        landmarks_list, frame = pose_ext.process_frame(frame, draw=True)
        
        # Append data for CSV export
        current_time_ms = (time.time() - start_time) * 1000
        collected_landmarks.append({
            'frame_index': frame_index,
            'timestamp': current_time_ms,
            'landmarks': landmarks_list
        })
        frame_index += 1
        
        # 4. Extract Features and Update Window
        if landmarks_list:
             norm_skel = normalize_skeleton(landmarks_list)
             angles = extract_body_angles(landmarks_list)
             motion_energy = calculate_motion_energy(prev_landmarks, landmarks_list)
             
             # (Removed old print to combine it below)
             # Store current landmarks for next frame's motion energy calculation
             prev_landmarks = landmarks_list
             
             # Combine everything into one features dictionary
             features = {
                 'normalized_landmarks': norm_skel,
                 'angles': angles,
                 'motion_energy': motion_energy,
                 'raw_hip_y': (landmarks_list[23]['y'] + landmarks_list[24]['y']) / 2.0
             }
             sliding_window = update_sliding_window(sliding_window, features, config.WINDOW_SIZE)
        else:
             # Still update window with "None" to maintain time accuracy
             sliding_window = update_sliding_window(sliding_window, None, config.WINDOW_SIZE)
             prev_landmarks = None

        # 5. Predict Raw State
        raw_state_probs = predict_state(sliding_window)

        # (Old squat heuristic filter removed to support standing upper-body exercises)

        # (Removed old print to combine it below)
        # 6. Apply Context Rules (Handle timeouts and Rest logic)
        final_state = state_machine.update(raw_state_probs)
        
        # 7. Predict Exercise (ONLY if state is 'active')
        exercise_data = predict_exercise(sliding_window, final_state, raw_state_probs)
        exercise = exercise_data['exercise'] if exercise_data else None
        
        # Track exercise duration
        if final_state == 'active' and exercise is not None and exercise != "N/A":
            if exercise not in exercise_durations_frames:
                exercise_durations_frames[exercise] = 0
            exercise_durations_frames[exercise] += 1
            
        # --- DETAILED SCIENTIFIC LOGGING FOR MEETING ---
        if frame_index % 30 == 0 and len(sliding_window) >= 32:
            print("\n" + "=" * 65)
            if landmarks_list:
                print(f"[Frame {frame_index:04d} | Energy: {motion_energy:.2f}] Biomechanical Analysis")
                print(f"  |- Legs : R-Knee ({angles.get('r_knee',0):.0f} deg) | L-Knee ({angles.get('l_knee',0):.0f} deg) | Hip ({angles.get('r_hip',0):.0f} deg)")
                print(f"  |- Arms : R-Elbow ({angles.get('r_elbow',0):.0f} deg) | L-Elbow ({angles.get('l_elbow',0):.0f} deg)")
            else:
                print(f"[Frame {frame_index:04d}] No Subject Detected")
                
            print(f"  |- State AI : [Null: {raw_state_probs['null']*100:02.0f}%] [Rest: {raw_state_probs['rest']*100:02.0f}%] [Active: {raw_state_probs['active']*100:02.0f}%] -> Output: {final_state.upper()}")
            if exercise_data and 'scores' in exercise_data:
                scores = exercise_data['scores']
                ex_scores_str = f"[Squat: {scores['squat']:02.0f}%] [Bicep: {scores['bicep_curl']:02.0f}%] [Shoulder: {scores['shoulder_press']:02.0f}%] [Deadlift: {scores['deadlift']:02.0f}%]"
                ex_str = exercise_data.get('exercise').upper() if exercise_data.get('exercise') else "N/A"
                print(f"  |- Exer. AI : {ex_scores_str} -> Output: {ex_str}")
                print(f"       |- Physics: Hip Drop ({exercise_data['hip_drop']:.3f}) | Leg Energy ({exercise_data['leg_energy']:.2f}) | Mask: {exercise_data['mask_rule']}")
            else:
                print(f"  |- Exer. AI : N/A")
        # -----------------------------------------------
        # 8. Render Visual Interface
        # (Skeleton was already drawn if draw=True above)
        frame = draw_info(frame, final_state, exercise, raw_state_probs)
        
        # 9. Display Frame
        cv2.imshow('Virtual Gym Coach - Module 1 Demo', frame)
        
        # If 'q' is pressed on the keyboard, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources smoothly
    release_capture(cap)
    cv2.destroyAllWindows()
    
    # Export data to CSV (unless --no-save was passed)
    if collected_landmarks and not args.no_save:
        print("Exporting landmark data to CSV...")
        save_landmarks_to_csv(collected_landmarks)
    elif args.no_save:
        print("Skipping CSV export (--no-save flag active).")

    # Export exercise durations for Module 3
    if exercise_durations_frames:
        import json
        exercise_durations_seconds = {ex: frames / 30.0 for ex, frames in exercise_durations_frames.items()}
        print("\n=== Session Exercise Durations ===")
        for ex, duration in exercise_durations_seconds.items():
            print(f"  {ex.capitalize()}: {duration:.2f} seconds")
            
        with open('data/session_summary.json', 'w') as f:
            json.dump(exercise_durations_seconds, f, indent=4)
        print("Saved session durations to data/session_summary.json")

    print("Application closed.")

if __name__ == "__main__":
    main()
