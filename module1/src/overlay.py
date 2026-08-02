import cv2

def draw_info(frame, state, exercise, raw_probs):
    """
    Draw real-time state and exercise information onto the video frame.

    Colour coding:
        active  -> green
        rest    -> yellow
        null    -> red
    """
    # Color coding based on state
    if state == 'active':
        color = (0, 255, 0)    # Green
    elif state == 'rest':
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)    # Red

    # State label (always shown)
    cv2.putText(frame, f"STATE: {state.upper()}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Exercise label (only when active)
    if state == 'active' and exercise:
        cv2.putText(frame, f"Exercise: {exercise}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Debug: raw model probabilities at the bottom of the frame
    prob_text = (
        f"Raw Probs -> "
        f"N:{raw_probs.get('null', 0):.2f}  "
        f"R:{raw_probs.get('rest', 0):.2f}  "
        f"A:{raw_probs.get('active', 0):.2f}"
    )
    cv2.putText(frame, prob_text,
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame
