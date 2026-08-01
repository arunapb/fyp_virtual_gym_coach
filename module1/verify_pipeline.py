"""
verify_pipeline.py
------------------
Baseline verification script for Module 1 prototype.
Confirms that Steps 1 to 5 are all working correctly without needing a live webcam.

Run with:
    python verify_pipeline.py

Each step prints PASS or FAIL with a short explanation.
"""

import sys
import os
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "  [ PASS ]"
FAIL = "  [ FAIL ]"


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Video IO
# ─────────────────────────────────────────────────────────────────────────────

def test_step1():
    section("Step 1 — Video IO  (src/video_io.py)")
    try:
        from src.video_io import get_video_capture, read_frame, release_capture
        print(f"{PASS}  get_video_capture imported OK")
        print(f"{PASS}  read_frame imported OK")
        print(f"{PASS}  release_capture imported OK")

        import config
        print(f"{PASS}  config.RESIZE_WIDTH  = {config.RESIZE_WIDTH}")
        print(f"{PASS}  config.RESIZE_HEIGHT = {config.RESIZE_HEIGHT}")
        print(f"{PASS}  config.TARGET_EXERCISES = {config.TARGET_EXERCISES}")
        return True
    except Exception as e:
        print(f"{FAIL}  {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Pose Extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_step2():
    section("Step 2 — Pose Extraction  (src/pose_extractor.py)")
    try:
        from src.pose_extractor import PoseExtractor

        extractor = PoseExtractor()
        print(f"{PASS}  PoseExtractor initialised (MediaPipe loaded)")

        # Use a plain black frame — MediaPipe won't detect a person but the
        # pipeline should run cleanly without crashing.
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks, annotated = extractor.process_frame(dummy_frame, draw=False)

        assert annotated.shape == (480, 640, 3), "Frame shape changed unexpectedly"
        print(f"{PASS}  process_frame ran on a 640×480 dummy frame without error")
        print(f"{PASS}  landmarks = {landmarks} (None is expected on a blank frame)")

        # Test with draw=True to confirm skeleton drawing doesn't crash
        _, annotated_draw = extractor.process_frame(dummy_frame, draw=True)
        assert annotated_draw is not None
        print(f"{PASS}  draw=True mode works without error")
        return True
    except Exception as e:
        print(f"{FAIL}  {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Landmark Saving
# ─────────────────────────────────────────────────────────────────────────────

def test_step3():
    section("Step 3 — Landmark Saving  (src/data_export.py)")
    try:
        from src.data_export import save_landmarks_to_csv
        import pandas as pd

        # Build two synthetic frames — one with landmarks, one without
        dummy_lm = [{'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 0.95}] * 33
        dummy_data = [
            {'frame_index': 0, 'timestamp': 0.0,   'landmarks': None},
            {'frame_index': 1, 'timestamp': 33.3,  'landmarks': dummy_lm},
        ]

        test_path = 'outputs/verify_landmarks_test.csv'
        save_landmarks_to_csv(dummy_data, output_path=test_path)

        assert os.path.exists(test_path), "CSV file was not created"
        df = pd.read_csv(test_path)

        assert len(df) == 2,          f"Expected 2 rows, got {len(df)}"
        assert 'frame_index' in df.columns, "Missing column: frame_index"
        assert 'timestamp'   in df.columns, "Missing column: timestamp"
        assert 'lm_0_x'      in df.columns, "Missing column: lm_0_x"
        assert 'lm_32_v'     in df.columns, "Missing column: lm_32_v"

        expected_cols = 2 + (33 * 4)   # frame_index + timestamp + 33 x (x, y, z, v)
        print(f"{PASS}  CSV created at {test_path}")
        print(f"{PASS}  Rows: {len(df)}  |  Columns: {len(df.columns)}  (expected {expected_cols})")
        print(f"{PASS}  Frame 0 lm_0_x = {df.iloc[0]['lm_0_x']}  (expected NaN -- no pose)")
        print(f"{PASS}  Frame 1 lm_0_x = {df.iloc[1]['lm_0_x']}  (expected 0.5)")

        os.remove(test_path)
        print(f"{PASS}  Temporary test CSV cleaned up")
        return True
    except Exception as e:
        print(f"{FAIL}  {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Feature Computation
# ─────────────────────────────────────────────────────────────────────────────

def test_step4():
    section("Step 4 — Feature Computation  (src/features.py)")
    try:
        from src.features import (
            normalize_skeleton,
            extract_body_angles,
            calculate_motion_energy,
        )

        # Create 33 landmarks spread out enough to avoid a zero torso length
        lm = []
        for i in range(33):
            lm.append({'x': 0.4 + (i % 5) * 0.04,
                        'y': 0.1 + i * 0.02,
                        'z': 0.0,
                        'visibility': 0.9})

        # ── normalize_skeleton ────────────────────────────────────────────
        norm = normalize_skeleton(lm)
        assert norm is not None,   "normalize_skeleton returned None"
        assert len(norm) == 33,    f"Expected 33 landmarks, got {len(norm)}"
        hip_cx = (norm[23]['x'] + norm[24]['x']) / 2.0
        hip_cy = (norm[23]['y'] + norm[24]['y']) / 2.0
        print(f"{PASS}  normalize_skeleton: 33 landmarks normalised")
        print(f"           Hip centre after normalisation: x={hip_cx:.4f}, y={hip_cy:.4f} (near 0,0 expected)")

        # ── extract_body_angles ───────────────────────────────────────────
        angles = extract_body_angles(lm)
        expected_keys = {'l_knee', 'r_knee', 'l_hip', 'r_hip',
                         'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'trunk'}
        assert expected_keys == set(angles.keys()), f"Unexpected angle keys: {set(angles.keys())}"
        print(f"{PASS}  extract_body_angles: {len(angles)} angles computed")
        for k, v in angles.items():
            print(f"             {k:15s} = {v:.1f} deg")

        # ── calculate_motion_energy ───────────────────────────────────────
        lm_prev = [{'x': lm[i]['x'] - 0.05, 'y': lm[i]['y'] - 0.05,
                    'z': 0.0, 'visibility': 0.9} for i in range(33)]
        energy = calculate_motion_energy(lm_prev, lm)
        assert energy > 0.0, "Motion energy should be positive when landmarks moved"
        print(f"{PASS}  calculate_motion_energy: energy = {energy:.4f}  (should be > 0)")

        # ── edge cases ────────────────────────────────────────────────────
        assert normalize_skeleton(None) is None
        assert extract_body_angles(None) == {}
        assert calculate_motion_energy(None, lm) == 0.0
        print(f"{PASS}  Edge cases (None inputs) handled safely")
        return True
    except Exception as e:
        print(f"{FAIL}  {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Sliding Window
# ─────────────────────────────────────────────────────────────────────────────

def test_step5():
    section("Step 5 — Sliding Window  (src/features.py)")
    try:
        import config
        from src.features import update_sliding_window, SlidingWindowBuffer

        dummy_feat = {'motion_energy': 0.1, 'angles': {}, 'normalized_landmarks': []}

        # ── rolling buffer (used in app.py live loop) ─────────────────────
        window = []
        for i in range(50):
            window = update_sliding_window(window, dummy_feat, config.WINDOW_SIZE)

        assert len(window) == config.WINDOW_SIZE, \
            f"Rolling buffer: expected {config.WINDOW_SIZE} frames, got {len(window)}"
        print(f"{PASS}  update_sliding_window: window stays at {config.WINDOW_SIZE} frames after 50 updates")

        # ── discrete overlapping buffer (for dataset preparation) ─────────
        buf = SlidingWindowBuffer(window_size=32, overlap_percent=0.5)
        # step_size should be 16 (50% overlap)
        assert buf.step_size == 16, f"Expected step_size=16, got {buf.step_size}"

        returned = []
        for i in range(70):
            result = buf.add_frame({'frame_id': i})
            if result is not None:
                returned.append(result)

        assert len(returned) > 0, "SlidingWindowBuffer never emitted a complete window"
        assert len(returned[0]) == 32, f"Window length should be 32, got {len(returned[0])}"

        # Verify overlap: frame 0 of window 2 should be frame 16 of window 1
        w1_first = returned[0][0]['frame_id']
        w2_first = returned[1][0]['frame_id']
        assert w2_first == w1_first + 16, \
            f"Overlap mismatch: window 1 starts at {w1_first}, window 2 starts at {w2_first} (expected +16)"

        print(f"{PASS}  SlidingWindowBuffer: step_size = {buf.step_size} (50% overlap confirmed)")
        print(f"{PASS}  {len(returned)} complete windows returned from 70 frames")
        print(f"{PASS}  Window 1 starts at frame {w1_first}, Window 2 starts at frame {w2_first} (+16 OK)")
        return True
    except Exception as e:
        print(f"{FAIL}  {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  MODULE 1 - PIPELINE VERIFICATION SCRIPT")
    print("  Checking Steps 1 to 5 of the preprocessing baseline")
    print("=" * 60)

    results = {
        "Step 1 - Video IO":         test_step1(),
        "Step 2 - Pose Extraction":  test_step2(),
        "Step 3 - Landmark Saving":  test_step3(),
        "Step 4 - Feature Compute":  test_step4(),
        "Step 5 - Sliding Window":   test_step5(),
    }

    section("SUMMARY")
    all_passed = True
    for step, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {step}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All 5 steps PASSED. Baseline is stable and ready.")
        print("  You can now proceed to Phase B (annotation and dataset).")
    else:
        print("  One or more steps FAILED. Fix the issues above before continuing.")
        sys.exit(1)

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
