"""
experiment3_threshold2.py
==========================
Standalone version of Experiment 3 from hyperparameter_search.py.
Finds the best hard-filter threshold (threshold2) -- how strongly a strongly
disliked ingredient gets excluded from recommendations.

Fully self-contained -- run directly with no other project scripts:
    python experiment3_threshold2.py

Upstream dependency: uses BEST_ALPHA (Experiment 1) and BEST_THRESHOLD1
(Experiment 2). Hardcoded below from those experiments' confirmed results --
edit the constants if you want to test against different upstream values.
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

# ── Upstream "best" values from Experiments 1-2 ────────────────────────────────
BEST_ALPHA      = 0.1
BEST_THRESHOLD1 = 0.20

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

TH2_VALUES = [0.30, 0.35, 0.40, 0.45, 0.50]


def reset_user_preferences():
    """Clear user_preference.json before each candidate run."""
    pref_file = os.path.join(BACKEND_DIR, "data", "user_preference.json")
    if os.path.exists(pref_file):
        os.remove(pref_file)
    save_preferences({})


def disliked_still_present(top3_all_meals: list, threshold2: float, prefs: dict) -> bool:
    disliked = [k for k, v in prefs.items() if v < -threshold2]
    for recipes in top3_all_meals:
        for recipe in recipes:
            title = recipe.get("title", "")
            ings  = recipe.get("ingredients", [])
            text  = (title + " " + " ".join(str(i) for i in ings)).lower()
            for d in disliked:
                if d.lower() in text:
                    return True
    return False


def main():
    print("=" * 60)
    print("EXPERIMENT 3: Finding Best Threshold 2 (Hard Filter)")
    print("=" * 60)

    th2_clean_sessions = []
    th2_warnings       = []

    for th2 in TH2_VALUES:
        print(f"Testing threshold2={th2:.2f} ...")
        reset_user_preferences()
        feedback_service.EMA_ALPHA = BEST_ALPHA

        clean_session = None
        zero_warning  = False

        for session_num, (text, rating, consumed) in enumerate(FEEDBACK_SEQUENCE, 1):
            analyze_feedback("", text, rating, consumed)
            prefs = load_preferences()

            top3_all_meals = []
            for meal_type, targets in MEAL_TARGETS.items():
                top3 = get_top_recipes(
                    meal_type=meal_type,
                    targets=targets,
                    prefs=prefs,
                    n_feedback=N_FEEDBACK,
                    top_n=3,
                    threshold1=BEST_THRESHOLD1,
                    threshold2=th2,
                )
                if not top3:
                    zero_warning = True
                top3_all_meals.append(top3)

            still_present = disliked_still_present(top3_all_meals, th2, prefs)
            if not still_present and clean_session is None:
                clean_session = session_num

        session_result = clean_session if clean_session is not None else 8
        th2_clean_sessions.append(session_result)
        th2_warnings.append("Zero candidates warning" if zero_warning else "None")
        print(f"  Disliked ingredients removed after session: {clean_session if clean_session is not None else '>7'}")

    # Pick best threshold2 (lowest sessions until clean, penalizing warnings)
    valid_th2 = [
        (s, w != "None", t2)
        for t2, s, w in zip(TH2_VALUES, th2_clean_sessions, th2_warnings)
    ]
    valid_th2.sort(key=lambda x: (x[0], x[1]))
    best_threshold2 = valid_th2[0][2]

    print("\n" + "=" * 45)
    print(f"{'Threshold2':<12}{'Sessions Until Clean':<22}{'Warning':<15}")
    print("-" * 45)
    for t2, s, w in zip(TH2_VALUES, th2_clean_sessions, th2_warnings):
        s_str = f"{s} sessions" if s <= 7 else ">7 sessions"
        print(f"{t2:<12.2f}{s_str:<22}{w:<15}")
    print("-" * 45)
    print(f"Best threshold2: {best_threshold2:.2f}")

    # Restore EMA_ALPHA to its original default so nothing leaks if this module
    # is imported elsewhere afterward.
    feedback_service.EMA_ALPHA = 0.5

    results_path = os.path.join(OUTPUT_DIR, "exp3_threshold2_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_alpha_used": BEST_ALPHA,
            "best_threshold1_used": BEST_THRESHOLD1,
            "values_tested": TH2_VALUES,
            "sessions_until_clean": [s if s <= 7 else ">7" for s in th2_clean_sessions],
            "warnings": th2_warnings,
            "best_threshold2": best_threshold2,
        }, f, indent=2)
    print(f"\nSaved JSON results to {results_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    sessions_plot_vals = [s if s <= 7 else 8 for s in th2_clean_sessions]
    bars = ax.bar([f"{t:.2f}" for t in TH2_VALUES], sessions_plot_vals, color="#C44E52")
    ax.set_title("Exp 3: Threshold 2 vs Sessions Until Clean")
    ax.set_xlabel("Threshold 2 (Hard Filter)")
    ax.set_ylabel("Sessions Until Clean")
    ax.set_ylim(0, 8)
    for bar, val in zip(bars, th2_clean_sessions):
        height = bar.get_height()
        label = f"{val} sess" if val <= 7 else ">7 sess"
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "exp3_threshold2_results.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved plot: {chart_path}")


if __name__ == "__main__":
    main()
