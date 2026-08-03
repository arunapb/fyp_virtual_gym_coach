"""
replay_check.py — threshold/rep-counter verification against an analysed job.

Replays the landmark CSV of an already-analysed job through the squat pipeline
(SquatExercise + RepCounter + FeedbackController) at the source's real frame
rate, reporting the reps, zone counts and audio cues it now produces.

This is a faithful reproduction of a real run, not an approximation: the pose
model is configured RunningMode.IMAGE with no temporal state between frames
(see src/pose_extractor.py), so the landmarks recorded for a frame are exactly
the landmarks a re-run would compute for it.  Everything downstream of pose
detection is therefore exercised for real — which makes this the cheap way to
check whether a threshold change actually catches the fault it was meant to,
on a clip whose faults are known, without re-encoding the video.

It does NOT cover Module 1's signalling, the overlay or the video writer.

Usage:  python replay_check.py <job_dir> <fps>
        python replay_check.py outputs/jobs/75493c29e9d3 15
"""

import csv
import sys
from collections import defaultdict, namedtuple

sys.path.insert(0, ".")

from src.audio_bridge import CueRecorder
from src.audio_feedback import FeedbackController
from src.exercises.squat import SquatExercise

LM = namedtuple("LM", "x y z visibility")

# Source frames were 810x1440; the analysed frame is the same size (analysis
# always runs at the original resolution — only the review copy is scaled).
W, H = 810, 1440


def load(job_dir):
    frames = defaultdict(dict)
    with open(f"{job_dir}/pose_landmarks.csv") as fh:
        for r in csv.DictReader(fh):
            frames[int(r["frame"])][int(r["landmark"])] = LM(
                float(r["x"]), float(r["y"]), float(r["z"]), float(r["visibility"]))
    return {f: [d[i] for i in range(33)] for f, d in sorted(frames.items())}


def main(job_dir, fps):
    frames = load(job_dir)
    ids = sorted(frames)
    ex = SquatExercise(fps=fps)
    audio = FeedbackController(player=CueRecorder(), fps=fps)
    audio.configure(cue_mapping=ex.audio_cue_mapping, signal_state="EXERCISE")

    calib = ex.calibrate([(frames[f], W, H) for f in ids[:30]])
    print(f"calibration: {  {k: round(v, 4) for k, v in calib.values.items()} }")

    cues, reps, zone_counts = [], [], defaultdict(lambda: defaultdict(int))
    for f in ids:
        lm = frames[f]
        feats = ex.compute_features(lm, calib, W, H)
        zones = ex.classify_zones(feats)
        rep = ex.update_rep_counter(feats, zones, f, lm)
        for k, v in zones.items():
            if v in ("green", "yellow", "red"):
                zone_counts[k][v] += 1
        if rep.new_rep_completed:
            reps.append(dict(rep.last_rep_summary))
        clip = audio.update(True, lm, zones, rep.phase, rep.rep_count,
                            ex.rep_history, now=f / fps)
        if clip:
            cues.append((f, round(f / fps, 2), clip, rep.phase))

    print(f"\nreps counted: {len(reps)}")
    for r in reps:
        print(f"  #{r['rep_num']} f{r['start_frame']}-{r['end_frame']} "
              f"min_knee={r['min_knee_angle']:.1f} shallow={r['shallow']} "
              f"worst_depth={r['worst_depth']} quality={r['rep_quality']}")

    print("\nzone counts:")
    for k in ("valgus_l", "valgus_r", "trunk", "depth", "overall"):
        print(f"  {k:9s} {dict(zone_counts[k])}")

    print(f"\ncues fired: {len(cues)}")
    for f, t, clip, phase in cues:
        print(f"  f{f:4d} t={t:5.2f}s  {clip}  (phase {phase})")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
