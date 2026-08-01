"""
experiment1_alpha.py
=====================
Standalone version of Experiment 1 from hyperparameter_search.py.
Finds the best EMA learning rate (alpha) for preference updates.

Fully self-contained -- run directly with no other project scripts:
    python experiment1_alpha.py

No upstream "best_*" values are needed (this is the first experiment in the
chain), so nothing here depends on any other experiment's output.
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

# ── Data ───────────────────────────────────────────────────────────────────────
SIMULATED_LIKED    = ["salmon", "spices", "chicken"]
SIMULATED_DISLIKED = ["garlic", "oil"]

FEEDBACK_SEQUENCE = [
    ("too much garlic",                    2, 40),
    ("spices were absolutely amazing",     5, 100),
    ("too much oil",                       2, 35),
    ("salmon was perfect",                 5, 100),
    ("chicken was great but too much oil", 3, 60),
    ("too much garlic in the dish",        2, 30),
    ("spices were amazing",                4, 90),
]

ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]


def reset_user_preferences():
    """Clear user_preference.json before each candidate run."""
    pref_file = os.path.join(BACKEND_DIR, "data", "user_preference.json")
    if os.path.exists(pref_file):
        os.remove(pref_file)
    save_preferences({})


def preference_alignment(prefs: dict) -> float:
    pos_score = sum(prefs.get(ing, 0.0) for ing in SIMULATED_LIKED) / len(SIMULATED_LIKED)
    neg_score = sum(abs(prefs.get(ing, 0.0)) for ing in SIMULATED_DISLIKED) / len(SIMULATED_DISLIKED)
    return round((pos_score + neg_score) / 2.0, 4)


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Finding Best Alpha (EMA Learning Rate)")
    print("=" * 60)

    alpha_scores = []
    for alpha in ALPHA_VALUES:
        print(f"Testing alpha={alpha} ...")
        reset_user_preferences()
        feedback_service.EMA_ALPHA = alpha

        for text, rating, consumed in FEEDBACK_SEQUENCE:
            analyze_feedback("", text, rating, consumed)

        prefs = load_preferences()
        score = preference_alignment(prefs)
        alpha_scores.append(score)
        print(f"  Completed 7 feedback sessions -> Alignment score: {score:.4f}")

    best_alpha_idx = alpha_scores.index(max(alpha_scores))
    best_alpha     = ALPHA_VALUES[best_alpha_idx]

    print("\n" + "=" * 32)
    print(f"{'Alpha':<10}{'Alignment Score':<20}")
    print("-" * 32)
    for a, s in zip(ALPHA_VALUES, alpha_scores):
        print(f"{a:<10.1f}{s:<20.4f}")
    print("-" * 32)
    print(f"Best alpha: {best_alpha}")

    # Restore EMA_ALPHA to its original default so nothing leaks if this module
    # is imported elsewhere afterward.
    feedback_service.EMA_ALPHA = 0.5

    results_path = os.path.join(OUTPUT_DIR, "exp1_alpha_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "values_tested": ALPHA_VALUES,
            "scores": alpha_scores,
            "best_alpha": best_alpha,
        }, f, indent=2)
    print(f"\nSaved JSON results to {results_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(a) for a in ALPHA_VALUES], alpha_scores, color="#4C72B0")
    ax.set_title("Exp 1: Alpha vs Preference Alignment")
    ax.set_xlabel("Alpha (EMA Learning Rate)")
    ax.set_ylabel("Alignment Score")
    ax.set_ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "exp1_alpha_results.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved plot: {chart_path}")


if __name__ == "__main__":
    main()
