import os, sys
EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
COMBINE_DIR = os.path.dirname(EVAL_DIR)
BACKEND_DIR = os.path.join(COMBINE_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from services.feedback_service import analyze_feedback

print("=" * 54)
print("  FEEDBACK TESTER")
print("=" * 54)

while True:
    feedback = input("\nYour feedback (or q to quit): ").strip()
    if feedback.lower() == "q":
        break
    analyze_feedback("", feedback, 3, 50)

print("\nDone.")
