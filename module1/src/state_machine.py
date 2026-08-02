import time
import config

class WorkoutStateMachine:
    def __init__(self):
        """Keep track of recent history to manage the 'Rest' timeout logic."""
        self.current_state = 'null'
        self.session_open = False
        self.frames_since_active = 0
    
    def update(self, state_probs):
        """
        Take the raw predictions from the biological/temporal model and
        apply the logical session rules (like timeouts).
        """
        # 1. Get highest probability class
        raw_prediction = max(state_probs, key=state_probs.get)
        
        # 2. Apply rules
        if raw_prediction == 'active':
            self.current_state = 'active'
            self.session_open = True
            self.frames_since_active = 0
            
        elif self.session_open:
            # If the AI predicts 'rest' or 'null' but a session is open,
            # we check how many frames it's been since they were truly active.
            self.frames_since_active += 1

            if self.frames_since_active < 45:
                 # It's been less than 1.5 seconds (45 frames) since they moved.
                 # They are probably just pausing at the top of a squat to breathe! Keep it active.
                 self.current_state = 'active'
            elif self.frames_since_active < config.TIMEOUT_SECONDS * 30:
                 # It's been over 1.5 seconds, so the set is definitely over and they are resting.
                 self.current_state = 'rest'
            else:
                 # It's been over TIMEOUT_SECONDS. Close the session.
                 self.current_state = 'null'
                 self.session_open = False
        else:
            # No session is open and the prediction is not 'active'
            self.current_state = 'null'
            
        return self.current_state
