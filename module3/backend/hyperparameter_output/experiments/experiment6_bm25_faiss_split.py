"""
experiment6_bm25_faiss_split.py
=================================
Standalone version of Experiment 6 from hyperparameter_search.py.
Sweeps the BM25/FAISS split ratio inside get_top_recipes() Stage 4 adaptive
RRF blend (`a = (1.0 - g) * 0.45`, `b = (1.0 - g) * 0.55`).

Fully self-contained -- run directly with no other project scripts:
    python experiment6_bm25_faiss_split.py

REPORTING ONLY: the 0.45/0.55 split is a hardcoded inline literal inside
get_top_recipes() (services/retrieval_service.py), not a function parameter
or module-level constant. To sweep alternate splits without touching that
file on disk, this script compiles an in-memory patched copy of the function
(reading two module-level globals instead of the literals, executed with the
real module's own __dict__ as globals so it still sees the live
bm25/FAISS/recipe_metadata/embedding state). It never modifies the live
0.45/0.55 default in retrieval_service.py, regardless of which candidate
scores highest.

Upstream dependency: uses BEST_ALPHA (Experiment 1), BEST_THRESHOLD1
(Experiment 2), BEST_THRESHOLD2 (Experiment 3). Hardcoded below from those
experiments' confirmed results -- edit the constants if you want to test
against different upstream values.
"""

import inspect
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

# ── Upstream "best" values from Experiments 1-3 ────────────────────────────────
BEST_ALPHA      = 0.1
BEST_THRESHOLD1 = 0.20
BEST_THRESHOLD2 = 0.30

# ── In-memory patch: sweep the BM25/FAISS split ratio ──────────────────────────
_SPLIT_ORIGINAL_LINES = "    a = (1.0 - g) * 0.45\n    b = (1.0 - g) * 0.55"
_SPLIT_PATCHED_LINES  = "    a = (1.0 - g) * _DIAG_BM25_RATIO\n    b = (1.0 - g) * _DIAG_FAISS_RATIO"

_split_original_src = inspect.getsource(retrieval_service.get_top_recipes)
if _SPLIT_ORIGINAL_LINES not in _split_original_src:
    raise RuntimeError(
        "Expected hardcoded a/b split lines not found in get_top_recipes() -- "
        "retrieval_service.py may have changed since this experiment was written."
    )
_split_patched_src = _split_original_src.replace(_SPLIT_ORIGINAL_LINES, _SPLIT_PATCHED_LINES)

_split_exec_ns = {}
exec(compile(_split_patched_src, "<experiment6 patched get_top_recipes>", "exec"),
     retrieval_service.__dict__, _split_exec_ns)
_patched_get_top_recipes_for_split = _split_exec_ns["get_top_recipes"]
retrieval_service._DIAG_BM25_RATIO = 0.45
retrieval_service._DIAG_FAISS_RATIO = 0.55

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

BM25_FAISS_SPLITS = [(0.30, 0.70), (0.40, 0.60), (0.45, 0.55), (0.50, 0.50), (0.55, 0.45), (0.70, 0.30)]


def reset_user_preferences():
    """Clear user_preference.json before each candidate run."""
    pref_file = os.path.join(BACKEND_DIR, "data", "user_preference.json")
    if os.path.exists(pref_file):
        os.remove(pref_file)
    save_preferences({})


def nutrition_closeness(recipe: dict, target_cal: float, target_prot: float) -> float:
    """Average of calorie-closeness and protein-closeness, each in [0, 1]."""
    cal  = recipe.get("calories", 0)
    prot = recipe.get("protein", 0)
    cal_close  = max(0.0, 1.0 - abs(cal - target_cal) / target_cal) if target_cal else 0.0
    prot_close = max(0.0, 1.0 - abs(prot - target_prot) / target_prot) if target_prot else 0.0
    return (cal_close + prot_close) / 2.0


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
    print("EXPERIMENT 6: Finding Best BM25/FAISS Split Ratio")
    print("(REPORTING ONLY -- live 0.45/0.55 default is never modified)")
    print("=" * 60)

    split_rows = []  # (bm25_ratio, faiss_ratio, nutrition_fit, preference_fit, overall_score)

    for bm25_ratio, faiss_ratio in BM25_FAISS_SPLITS:
        print(f"Testing bm25_ratio={bm25_ratio:.2f} / faiss_ratio={faiss_ratio:.2f} ...")
        reset_user_preferences()
        feedback_service.EMA_ALPHA = BEST_ALPHA
        retrieval_service._DIAG_BM25_RATIO = bm25_ratio
        retrieval_service._DIAG_FAISS_RATIO = faiss_ratio

        for text, rating, consumed in FEEDBACK_SEQUENCE:
            analyze_feedback("", text, rating, consumed)

        prefs = load_preferences()

        all_top3 = []
        for meal_type, targets in MEAL_TARGETS.items():
            top3 = _patched_get_top_recipes_for_split(
                meal_type=meal_type,
                targets=targets,
                prefs=prefs,
                n_feedback=N_FEEDBACK,
                top_n=3,
                threshold1=BEST_THRESHOLD1,
                threshold2=BEST_THRESHOLD2,
            )
            all_top3.extend((meal_type, r) for r in top3)

        nut_scores = [
            nutrition_closeness(r, MEAL_TARGETS[mt]["calories"], MEAL_TARGETS[mt]["protein"])
            for mt, r in all_top3
        ]
        nutrition_fit = sum(nut_scores) / len(nut_scores) if nut_scores else 0.0

        preference_fit = retrieval_relevance([r for _, r in all_top3], prefs, BEST_THRESHOLD1)

        overall_score = (nutrition_fit + preference_fit) / 2.0
        split_rows.append((bm25_ratio, faiss_ratio, nutrition_fit, preference_fit, overall_score))
        print(f"  nutrition_fit={nutrition_fit:.4f}  preference_fit={preference_fit:.4f}  overall_score={overall_score:.4f}")

    # Restore the split globals and EMA_ALPHA to their real defaults so nothing
    # leaks if this module is imported elsewhere afterward.
    retrieval_service._DIAG_BM25_RATIO = 0.45
    retrieval_service._DIAG_FAISS_RATIO = 0.55
    feedback_service.EMA_ALPHA = 0.5

    best_split_idx = max(range(len(split_rows)), key=lambda i: split_rows[i][4])
    best_bm25_ratio, best_faiss_ratio, _, _, best_split_score = split_rows[best_split_idx]

    print("\n" + "=" * 78)
    print(f"{'BM25 Ratio':<13}{'FAISS Ratio':<13}{'Nutrition Fit':<16}{'Preference Fit':<17}{'Overall Score':<14}")
    print("-" * 78)
    for bm25_ratio, faiss_ratio, nutrition_fit, preference_fit, overall_score in split_rows:
        print(f"{bm25_ratio:<13.2f}{faiss_ratio:<13.2f}{nutrition_fit:<16.4f}{preference_fit:<17.4f}{overall_score:<14.4f}")
    print("-" * 78)
    _matches_default = abs(best_bm25_ratio - 0.45) < 1e-9
    print(f"Best BM25/FAISS split: bm25_ratio={best_bm25_ratio:.2f} / faiss_ratio={best_faiss_ratio:.2f} "
          f"(overall_score={best_split_score:.4f}) -- "
          f"{'MATCHES' if _matches_default else 'DIFFERS FROM'} the current default (0.45/0.55).")

    results_path = os.path.join(OUTPUT_DIR, "exp6_bm25_faiss_split_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "note": "Verification only -- not applied to live defaults. Live 0.45/0.55 split in retrieval_service.py was NOT modified.",
            "best_alpha_used": BEST_ALPHA,
            "best_threshold1_used": BEST_THRESHOLD1,
            "best_threshold2_used": BEST_THRESHOLD2,
            "splits_tested": BM25_FAISS_SPLITS,
            "rows": [
                {
                    "bm25_ratio": br, "faiss_ratio": fr,
                    "nutrition_fit": nf, "preference_fit": pf, "overall_score": os_,
                }
                for br, fr, nf, pf, os_ in split_rows
            ],
            "best_bm25_ratio": best_bm25_ratio,
            "best_faiss_ratio": best_faiss_ratio,
            "best_overall_score": best_split_score,
            "matches_current_default": bool(abs(best_bm25_ratio - 0.45) < 1e-9),
        }, f, indent=2)
    print(f"\nSaved JSON results to {results_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    split_labels = [f"{br:.2f}/{fr:.2f}" for br, fr, *_ in split_rows]
    bars = ax.bar(split_labels, [r[4] for r in split_rows], color="#64B5CD")
    ax.set_title("Exp 6: BM25/FAISS Split vs Overall Score (verification only)")
    ax.set_xlabel("BM25 Ratio / FAISS Ratio")
    ax.set_ylabel("Overall Score")
    ax.set_ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "exp6_bm25_faiss_split_results.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved plot: {chart_path}")


if __name__ == "__main__":
    main()
