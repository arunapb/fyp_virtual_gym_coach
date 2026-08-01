"""
experiment2_threshold1.py
==========================
Standalone version of Experiment 2 from hyperparameter_search.py.
Finds the best query-building threshold (threshold1).

Fully self-contained -- run directly with no other project scripts:
    python experiment2_threshold1.py

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
from services.retrieval_service import get_top_recipes

retrieval_service.USE_QDRANT = False  # offline/local FAISS for speed & determinism

# ── Upstream "best" value from Experiment 1 ────────────────────────────────────
BEST_ALPHA = 0.1

# ── Data ───────────────────────────────────────────────────────────────────────
MEAL_TARGETS = {
    "Breakfast": {"calories": 568, "protein": 35},
    "Lunch":     {"calories": 909, "protein": 56},
    "Dinner":    {"calories": 795, "protein": 49},
}

N_FEEDBACK = 10

FEEDBACK_SEQUENCE = [
    ("too much garlic",                    2, 40),
    ("spices were absolutely amazing",     5, 100),
    ("too much oil",                       2, 35),
    ("salmon was perfect",                 5, 100),
    ("chicken was great but too much oil", 3, 60),
    ("too much garlic in the dish",        2, 30),
    ("spices were amazing",                4, 90),
]

TH1_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40]


def reset_user_preferences():
    """Clear user_preference.json before each candidate run."""
    pref_file = os.path.join(BACKEND_DIR, "data", "user_preference.json")
    if os.path.exists(pref_file):
        os.remove(pref_file)
    save_preferences({})


def retrieval_relevance(top3_recipes: list, prefs: dict, threshold1: float) -> float:
    liked = [k for k, v in prefs.items() if v > threshold1]
    if not liked:
        return 0.0
    total = len(top3_recipes)
    matched = 0
    for recipe in top3_recipes:
        title = recipe.get("title", "")
        ings  = recipe.get("ingredients", [])
        text  = (title + " " + " ".join(str(i) for i in ings)).lower()
        if any(l.lower() in text for l in liked):
            matched += 1
    return round(matched / total, 4) if total else 0.0


def main():
    print("=" * 60)
    print("EXPERIMENT 2: Finding Best Threshold 1 (Query Building)")
    print("=" * 60)

    th1_scores = []
    for th1 in TH1_VALUES:
        print(f"Testing threshold1={th1:.2f} ...")
        reset_user_preferences()
        feedback_service.EMA_ALPHA = BEST_ALPHA

        for text, rating, consumed in FEEDBACK_SEQUENCE:
            analyze_feedback("", text, rating, consumed)

        prefs = load_preferences()

        all_recipes = []
        for meal_type, targets in MEAL_TARGETS.items():
            top3 = get_top_recipes(
                meal_type=meal_type,
                targets=targets,
                prefs=prefs,
                n_feedback=N_FEEDBACK,
                top_n=3,
                threshold1=th1,
                threshold2=0.40,
            )
            all_recipes.extend(top3)

        score = retrieval_relevance(all_recipes, prefs, th1)
        th1_scores.append(score)
        print(f"  Retrieval completed -> Relevance score: {score:.4f}")

    best_th1_idx    = th1_scores.index(max(th1_scores))
    best_threshold1 = TH1_VALUES[best_th1_idx]

    print("\n" + "=" * 32)
    print(f"{'Threshold1':<12}{'Relevance Score':<20}")
    print("-" * 32)
    for t1, s in zip(TH1_VALUES, th1_scores):
        print(f"{t1:<12.2f}{s:<20.4f}")
    print("-" * 32)
    print(f"Best threshold1: {best_threshold1:.2f}")

    # Restore EMA_ALPHA to its original default so nothing leaks if this module
    # is imported elsewhere afterward.
    feedback_service.EMA_ALPHA = 0.5

    results_path = os.path.join(OUTPUT_DIR, "exp2_threshold1_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_alpha_used": BEST_ALPHA,
            "values_tested": TH1_VALUES,
            "scores": th1_scores,
            "best_threshold1": best_threshold1,
        }, f, indent=2)
    print(f"\nSaved JSON results to {results_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f"{t:.2f}" for t in TH1_VALUES], th1_scores, color="#55A868")
    ax.set_title("Exp 2: Threshold 1 vs Retrieval Relevance")
    ax.set_xlabel("Threshold 1 (Query Building)")
    ax.set_ylabel("Relevance Score")
    ax.set_ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "exp2_threshold1_results.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved plot: {chart_path}")


if __name__ == "__main__":
    main()
