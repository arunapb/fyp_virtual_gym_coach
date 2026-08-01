import pandas as pd
import os

def save_landmarks_to_csv(landmarks_data, output_path="outputs/landmarks.csv"):
    """
    Saves collected pose landmarks to a CSV file using pandas.
    
    Args:
        landmarks_data (list of dicts): A list where each dictionary represents a frame and contains:
            - 'frame_index': int, the index of the frame.
            - 'timestamp': float, the timestamp of the frame.
            - 'landmarks': list of 33 dicts with 'x', 'y', 'z', 'visibility', or None if no pose is detected.
        output_path (str): The file path where the CSV will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    for frame_data in landmarks_data:
        # Base row data
        row = {
            'frame_index': frame_data['frame_index'],
            'timestamp': frame_data['timestamp']
        }
        
        # Flatten the 33 landmarks into individual columns
        if frame_data.get('landmarks'):
            for i, lm in enumerate(frame_data['landmarks']):
                row[f'lm_{i}_x'] = lm['x']
                row[f'lm_{i}_y'] = lm['y']
                row[f'lm_{i}_z'] = lm['z']
                row[f'lm_{i}_v'] = lm['visibility']
        else:
            # If no pose was detected, populate with empty values
            for i in range(33):
                row[f'lm_{i}_x'] = None
                row[f'lm_{i}_y'] = None
                row[f'lm_{i}_z'] = None
                row[f'lm_{i}_v'] = None
                
        rows.append(row)
        
    # Create a pandas DataFrame and export to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved landmarks to {output_path}")
