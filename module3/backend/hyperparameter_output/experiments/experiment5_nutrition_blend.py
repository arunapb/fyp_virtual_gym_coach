"""
experiment5_nutrition_blend.py
================================
Standalone version of Experiment 5 from hyperparameter_search.py.
Sweeps the nutrition/RRF blend weight inside get_top_recipes() Stage 4-5
scoring (`final = nut_score * 0.85 + rrf_norm * 0.15`).

Fully self-contained -- run directly with no other project scripts:
    python experiment5_nutrition_blend.py

REPORTING ONLY: the blend weight is a hardcoded inline literal inside
get_top_recipes() (services/retrieval_service.py), not a function parameter
or module-level constant. The exact literal has been observed changing on
disk (someone actively editing retrieval_service.py, toggling between a
0.85/0.15 line and other values via comments), so this script does NOT
hardcode one exact string -- it locates whichever `final = nut_score * X +
rrf_norm * Y` line is CURRENTLY active (not commented out) via regex and
patches THAT line, whatever X/Y happen to be right now. To sweep alternate
weights without touching that file on disk, this script compiles an
in-memory patched copy of the function (reading two module-level globals
instead of the literals, executed with the real module's own __dict__ as
globals so it still sees the live bm25/FAISS/recipe_metadata/embedding
state). It never modifies the live default in retrieval_service.py,
regardless of which candidate scores highest.

Upstream dependency: uses BEST_ALPHA (Experiment 1), BEST_THRESHOLD1
(Experiment 2), BEST_THRESHOLD2 (Experiment 3). Hardcoded below from those
experiments' confirmed results -- edit the constants if you want to test
against different upstream values.
"""

import inspect
import json
import os
import re
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

# ── In-memory patch: sweep the nutrition/RRF blend weight ─────────────────────
# Detects whichever `final = nut_score * X + rrf_norm * Y` line is currently
# active (not commented out) instead of hardcoding one exact literal, since
# the live file's weight values have been observed changing between runs.
_BLEND_LINE_RE = re.compile(
    r"^[ \t]*final\s*=\s*nut_score\s*\*\s*([0-9]*\.?[0-9]+)\s*\+\s*rrf_norm\s*\*\s*([0-9]*\.?[0-9]+)[ \t]*$",
    re.MULTILINE,
)

_blend_original_src = inspect.getsource(retrieval_service.get_top_recipes)
_blend_match = _BLEND_LINE_RE.search(_blend_original_src)
if _blend_match is None:
    raise RuntimeError(
        "Could not find an active (uncommented) `final = nut_score * X + rrf_norm * Y` "
        "line in get_top_recipes() -- retrieval_service.py may have changed structurally "
        "since this experiment was written, or the line is currently commented out."
    )
_BLEND_ORIGINAL_LINE   = _blend_match.group(0)
_LIVE_NUTRITION_WEIGHT = float(_blend_match.group(1))
_LIVE_RRF_WEIGHT       = float(_blend_match.group(2))
_BLEND_PATCHED_LINE    = "        final      = nut_score * _DIAG_NUTRITION_WEIGHT + rrf_norm * _DIAG_RRF_WEIGHT"

print(f"[setup] Detected live nutrition/RRF blend line: "
      f"nutrition_weight={_LIVE_NUTRITION_WEIGHT} / rrf_weight={_LIVE_RRF_WEIGHT}")

_blend_patched_src = _blend_original_src.replace(_BLEND_ORIGINAL_LINE, _BLEND_PATCHED_LINE)

_blend_exec_ns = {}
exec(compile(_blend_patched_src, "<experiment5 patched get_top_recipes>", "exec"),
     retrieval_service.__dict__, _blend_exec_ns)
_patched_get_top_recipes_for_blend = _blend_exec_ns["get_top_recipes"]
retrieval_service._DIAG_NUTRITION_WEIGHT = _LIVE_NUTRITION_WEIGHT
retrieval_service._DIAG_RRF_WEIGHT = _LIVE_RRF_WEIGHT

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

BLEND_NUTRITION_WEIGHTS = [0.70, 0.80, 0.85, 0.90, 0.95]


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


def contains_liked_ingredient(recipe: dict, liked_words: list) -> bool:
    title = recipe.get("title", "")
    ings  = recipe.get("ingredients", [])
    text  = (title + " " + " ".join(str(i) for i in ings)).lower()
    return any(w.lower() in text for w in liked_words)


def main():
    print("=" * 60)
    print("EXPERIMENT 5: Finding Best Nutrition/RRF Blend Weight")
    print("(REPORTING ONLY -- the live default is never modified)")
    print("=" * 60)

    blend_rows = []  # (nut_w, rrf_w, nutrition_fit, preference_fit, overall_score)

    for nut_w in BLEND_NUTRITION_WEIGHTS:
        rrf_w = round(1.0 - nut_w, 2)
        print(f"Testing nutrition_weight={nut_w:.2f} / rrf_weight={rrf_w:.2f} ...")
        reset_user_preferences()
        feedback_service.EMA_ALPHA = BEST_ALPHA
        retrieval_service._DIAG_NUTRITION_WEIGHT = nut_w
        retrieval_service._DIAG_RRF_WEIGHT = rrf_w

        for text, rating, consumed in FEEDBACK_SEQUENCE:
            analyze_feedback("", text, rating, consumed)

        prefs = load_preferences()
        liked_words = [k for k, v in prefs.items() if v > BEST_THRESHOLD1]

        all_top3 = []
        for meal_type, targets in MEAL_TARGETS.items():
            top3 = _patched_get_top_recipes_for_blend(
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

        pref_hits = sum(1 for _, r in all_top3 if contains_liked_ingredient(r, liked_words))
        preference_fit = pref_hits / len(all_top3) if all_top3 else 0.0

        overall_score = (nutrition_fit + preference_fit) / 2.0
        blend_rows.append((nut_w, rrf_w, nutrition_fit, preference_fit, overall_score))
        print(f"  nutrition_fit={nutrition_fit:.4f}  preference_fit={preference_fit:.4f}  overall_score={overall_score:.4f}")

    # Restore the blend globals and EMA_ALPHA to their real defaults so nothing
    # leaks if this module is imported elsewhere afterward.
    retrieval_service._DIAG_NUTRITION_WEIGHT = _LIVE_NUTRITION_WEIGHT
    retrieval_service._DIAG_RRF_WEIGHT = _LIVE_RRF_WEIGHT
    feedback_service.EMA_ALPHA = 0.5

    best_blend_idx = max(range(len(blend_rows)), key=lambda i: blend_rows[i][4])
    best_nutrition_weight, best_rrf_weight, _, _, best_blend_score = blend_rows[best_blend_idx]

    print("\n" + "=" * 78)
    print(f"{'Nutrition W':<13}{'RRF W':<9}{'Nutrition Fit':<16}{'Preference Fit':<17}{'Overall Score':<14}")
    print("-" * 78)
    for nut_w, rrf_w, nutrition_fit, preference_fit, overall_score in blend_rows:
        print(f"{nut_w:<13.2f}{rrf_w:<9.2f}{nutrition_fit:<16.4f}{preference_fit:<17.4f}{overall_score:<14.4f}")
    print("-" * 78)
    print(f"Best nutrition/RRF blend weight: {best_nutrition_weight:.2f} / {best_rrf_weight:.2f} "
          f"(overall_score={best_blend_score:.4f})")

    results_path = os.path.join(OUTPUT_DIR, "exp5_blend_weight_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "note": f"Reporting/verification only -- live default in retrieval_service.py (nutrition_weight={_LIVE_NUTRITION_WEIGHT}/rrf_weight={_LIVE_RRF_WEIGHT} at time of this run) was NOT modified.",
            "best_alpha_used": BEST_ALPHA,
            "best_threshold1_used": BEST_THRESHOLD1,
            "best_threshold2_used": BEST_THRESHOLD2,
            "nutrition_weights_tested": BLEND_NUTRITION_WEIGHTS,
            "rows": [
                {
                    "nutrition_weight": nw, "rrf_weight": rw,
                    "nutrition_fit": nf, "preference_fit": pf, "overall_score": os_,
                }
                for nw, rw, nf, pf, os_ in blend_rows
            ],
            "best_nutrition_weight": best_nutrition_weight,
            "best_rrf_weight": best_rrf_weight,
            "best_overall_score": best_blend_score,
        }, f, indent=2)
    print(f"\nSaved JSON results to {results_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f"{w:.2f}" for w in BLEND_NUTRITION_WEIGHTS],
                  [r[4] for r in blend_rows], color="#8172B2")
    ax.set_title("Exp 5: Nutrition Weight vs Overall Score (reporting only)")
    ax.set_xlabel("Nutrition Weight (RRF weight = 1 - nutrition weight)")
    ax.set_ylabel("Overall Score")
    ax.set_ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "exp5_blend_weight_results.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved plot: {chart_path}")


if __name__ == "__main__":
    main()
