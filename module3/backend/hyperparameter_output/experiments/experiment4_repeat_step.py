"""
experiment4_repeat_step.py
============================
Standalone version of Experiment 4 from hyperparameter_search.py.
Finds the best repeat-mention growth step -- how fast a repeated,
same-direction preference statement grows toward the "liked" threshold.

Fully self-contained -- run directly with no other project scripts:
    python experiment4_repeat_step.py

Upstream dependency: uses BEST_ALPHA, the result of Experiment 1
(experiment1_alpha.py). Hardcoded below from that experiment's confirmed
result -- edit BEST_ALPHA if you want to test against a different value.
"""

import json
import os
import sys
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))          # .../backend/hyperparameter_output/experiments
OUTPUT_DIR  = os.path.dirname(THIS_DIR)                            # .../backend/hyperparameter_output
BACKEND_DIR = os.path.dirname(OUTPUT_DIR)                          # .../backend

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import services.feedback_service as feedback_service
import services.retrieval_service as retrieval_service
from services.feedback_service import analyze_feedback, load_preferences, save_preferences

retrieval_service.USE_QDRANT = False  # offline/local FAISS for speed & determinism

# ── Upstream "best" value from Experiment 1 ────────────────────────────────────
BEST_ALPHA = 0.1

# ── Data ───────────────────────────────────────────────────────────────────────
REPEAT_STEP_VALUES = [0.05, 0.10, 0.15, 0.20]
N_REPEATS       = 5
LIKED_THRESHOLD = 0.30   # "liked" threshold used in retrieval (threshold1 default)
HIGH_DELTA      = 0.60   # delta value for a single HIGH-intensity statement
TEST_TEXT       = "eggs were fine"  # LOW-intensity (VADER compound ~0.20), positive, aspect="eggs"


def reset_user_preferences():
    """Clear user_preference.json before each candidate run."""
    pref_file = os.path.join(BACKEND_DIR, "data", "user_preference.json")
    if os.path.exists(pref_file):
        os.remove(pref_file)
    save_preferences({})


def main():
    print("=" * 60)
    print("EXPERIMENT 4: Finding Best Repeat-Mention Growth Step")
    print("=" * 60)

    repeat_step_scores           = []  # list of [s1..s5] per candidate
    repeat_step_repeats_to_liked = []
    repeat_step_overshoot        = []

    for step in REPEAT_STEP_VALUES:
        print(f"Testing repeat_mention_step={step:.2f} ...")
        reset_user_preferences()
        feedback_service.EMA_ALPHA           = BEST_ALPHA
        feedback_service.REPEAT_MENTION_STEP = step

        scores = []
        for _ in range(N_REPEATS):
            analyze_feedback("", TEST_TEXT, 4, 80)
            prefs = load_preferences()
            scores.append(prefs.get("eggs", 0.0))

        repeats_to_liked = next((i + 1 for i, s in enumerate(scores) if s >= LIKED_THRESHOLD), None)
        overshoot = scores[-1] > HIGH_DELTA

        repeat_step_scores.append(scores)
        repeat_step_repeats_to_liked.append(repeats_to_liked)
        repeat_step_overshoot.append(overshoot)

        rtl_display = repeats_to_liked if repeats_to_liked is not None else f">{N_REPEATS}"
        print(f"  Scores after 5 mentions: {[round(s, 4) for s in scores]} -> "
              f"repeats_to_liked={rtl_display}, overshoot={overshoot}")

    # Restore REPEAT_MENTION_STEP and EMA_ALPHA to their original defaults so
    # nothing leaks if this module is imported elsewhere afterward.
    feedback_service.REPEAT_MENTION_STEP = 0.10
    feedback_service.EMA_ALPHA           = 0.5

    # Pick best step: lowest repeats_to_liked among non-overshooting candidates;
    # ties broken by preferring the smallest (most conservative) step value.
    candidates = [
        (
            repeat_step_repeats_to_liked[i] if repeat_step_repeats_to_liked[i] is not None else float("inf"),
            REPEAT_STEP_VALUES[i],
        )
        for i in range(len(REPEAT_STEP_VALUES))
        if not repeat_step_overshoot[i]
    ]
    if not candidates:
        candidates = [
            (repeat_step_repeats_to_liked[i] if repeat_step_repeats_to_liked[i] is not None else float("inf"),
             REPEAT_STEP_VALUES[i])
            for i in range(len(REPEAT_STEP_VALUES))
        ]
    candidates.sort(key=lambda c: (c[0], c[1]))
    best_repeat_step = candidates[0][1]
    best_step_idx    = REPEAT_STEP_VALUES.index(best_repeat_step)

    print("\n" + "=" * 68)
    print(f"{'Step':<8}{'Scores (rep1..rep5)':<38}{'Repeats->Liked':<16}{'Overshoot':<10}")
    print("-" * 68)
    for step, scores, rtl, ov in zip(REPEAT_STEP_VALUES, repeat_step_scores, repeat_step_repeats_to_liked, repeat_step_overshoot):
        rtl_str    = str(rtl) if rtl is not None else f">{N_REPEATS}"
        scores_str = "[" + ", ".join(f"{s:.2f}" for s in scores) + "]"
        print(f"{step:<8.2f}{scores_str:<38}{rtl_str:<16}{str(ov):<10}")
    print("-" * 68)
    print(f"Best repeat_mention_step: {best_repeat_step:.2f}")
    print(
        f"Reasoning: step={best_repeat_step:.2f} reaches the 'liked' threshold ({LIKED_THRESHOLD}) in the "
        f"fewest repeats ({repeat_step_repeats_to_liked[best_step_idx]}) among candidates that do NOT "
        f"overshoot the single HIGH-intensity delta ({HIGH_DELTA}) after {N_REPEATS} repeats "
        f"(final score {repeat_step_scores[best_step_idx][-1]:.2f})."
    )

    results_path = os.path.join(OUTPUT_DIR, "exp4_repeat_step_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_alpha_used": BEST_ALPHA,
            "values_tested": REPEAT_STEP_VALUES,
            "scores_per_repeat": repeat_step_scores,
            "repeats_to_liked": [r if r is not None else f">{N_REPEATS}" for r in repeat_step_repeats_to_liked],
            "overshoot_flag": repeat_step_overshoot,
            "best_repeat_step": best_repeat_step,
        }, f, indent=2)
    print(f"\nSaved JSON results to {results_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    repeat_numbers = list(range(1, N_REPEATS + 1))
    line_colors    = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for step, scores, color in zip(REPEAT_STEP_VALUES, repeat_step_scores, line_colors):
        ax.plot(repeat_numbers, scores, marker="o", label=f"step={step:.2f}", color=color)
    ax.axhline(y=LIKED_THRESHOLD, color="gray", linestyle="--", linewidth=1, label="Liked threshold (0.30)")
    ax.axhline(y=HIGH_DELTA, color="black", linestyle=":", linewidth=1, label="HIGH delta (0.60)")
    ax.set_title("Exp 4: Repeat-Mention Step vs Score Progression")
    ax.set_xlabel("Mention Number")
    ax.set_ylabel("Preference Score")
    ax.set_xticks(repeat_numbers)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "exp4_repeat_step_results.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved plot: {chart_path}")


if __name__ == "__main__":
    main()
