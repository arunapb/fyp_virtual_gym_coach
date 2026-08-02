"""
experiment7_combined_sweep.py
===============================
Standalone version of Experiment 7 from hyperparameter_search.py.
Sweeps the nutrition/search weight (5 levels) AND the BM25/FAISS split
(5 levels) simultaneously -- a 5x5 grid, 25 combinations -- to see whether
lowering the nutrition weight makes the BM25/FAISS split matter more for
which recipes get recommended.

Fully self-contained -- run directly with no other project scripts:
    python experiment7_combined_sweep.py

REPORTING ONLY: combines the Experiment 5 patch (nutrition/RRF blend line)
and the Experiment 6 patch (BM25/FAISS split lines) into a SINGLE in-memory-
patched copy of get_top_recipes(), built from the same pristine source with
BOTH line replacements applied (they touch different, non-overlapping lines,
so they compose without conflict). The nutrition/RRF blend line's exact
literal has been observed changing on disk (someone actively editing
retrieval_service.py, toggling between a 0.85/0.15 line and other values via
comments), so this script does NOT hardcode one exact string for it -- it
locates whichever `final = nut_score * X + rrf_norm * Y` line is CURRENTLY
active (not commented out) via regex. Never modifies the live defaults in
retrieval_service.py (nutrition/RRF blend line, whatever it currently is;
BM25/FAISS split, 0.45/0.55).

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

# ── In-memory patch: nutrition/RRF blend line ──────────────────────────────────
# Detects whichever `final = nut_score * X + rrf_norm * Y` line is currently
# active (not commented out) instead of hardcoding one exact literal, since
# the live file's weight values have been observed changing between runs.
_BLEND_LINE_RE = re.compile(
    r"^[ \t]*final\s*=\s*nut_score\s*\*\s*([0-9]*\.?[0-9]+)\s*\+\s*rrf_norm\s*\*\s*([0-9]*\.?[0-9]+)[ \t]*$",
    re.MULTILINE,
)

_original_src = inspect.getsource(retrieval_service.get_top_recipes)
_blend_match = _BLEND_LINE_RE.search(_original_src)
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

# ── In-memory patch: BM25/FAISS split lines ────────────────────────────────────
_SPLIT_ORIGINAL_LINES = "    a = (1.0 - g) * 0.45\n    b = (1.0 - g) * 0.55"
_SPLIT_PATCHED_LINES  = "    a = (1.0 - g) * _DIAG_BM25_RATIO\n    b = (1.0 - g) * _DIAG_FAISS_RATIO"

if _SPLIT_ORIGINAL_LINES not in _original_src:
    raise RuntimeError(
        "Expected hardcoded a/b split lines not found in get_top_recipes() -- "
        "retrieval_service.py may have changed since this experiment was written."
    )

# Combine BOTH replacements into a single patched copy -- the two target
# different, non-overlapping lines in the same function, so they compose
# without conflict.
_combined_patched_src = _original_src.replace(_BLEND_ORIGINAL_LINE, _BLEND_PATCHED_LINE)
_combined_patched_src = _combined_patched_src.replace(_SPLIT_ORIGINAL_LINES, _SPLIT_PATCHED_LINES)
if _combined_patched_src == _original_src:
    raise RuntimeError("Combined patch did not apply -- check line targets.")

_combined_exec_ns = {}
exec(compile(_combined_patched_src, "<experiment7 patched get_top_recipes (combined)>", "exec"),
     retrieval_service.__dict__, _combined_exec_ns)
_patched_get_top_recipes_for_combined = _combined_exec_ns["get_top_recipes"]

# Live defaults, restored after every candidate run so nothing leaks.
_LIVE_BM25_RATIO  = 0.45
_LIVE_FAISS_RATIO = 0.55

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

NUTRITION_SEARCH_WEIGHTS = [(0.85, 0.15), (0.80, 0.20), (0.75, 0.25), (0.70, 0.30), (0.65, 0.35)]
BM25_FAISS_SPLITS        = [(0.20, 0.80), (0.30, 0.70), (0.45, 0.55), (0.60, 0.40), (0.70, 0.30)]

# Search-sensitivity threshold (out of 9 slots) that marks the split as
# "meaningfully non-trivial" for the best-trade-off-point finding below.
SENSITIVITY_THRESHOLD = 4


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
    print("EXPERIMENT 7: Nutrition/Search Weight x BM25/FAISS Split")
    print("(REPORTING ONLY -- live defaults are never modified)")
    print("=" * 60)

    exp7_metrics = {}  # (nut_w, bm25_ratio) -> (nutrition_fit, preference_fit, overall_score)
    exp7_titles  = {}  # (nut_w, bm25_ratio) -> flat list of 9 recipe titles (fixed slot order)

    for nut_w, search_w in NUTRITION_SEARCH_WEIGHTS:
        for bm25_ratio, faiss_ratio in BM25_FAISS_SPLITS:
            print(f"Testing nutrition_weight={nut_w:.2f}/search_weight={search_w:.2f}  "
                  f"bm25_ratio={bm25_ratio:.2f}/faiss_ratio={faiss_ratio:.2f} ...")
            reset_user_preferences()
            feedback_service.EMA_ALPHA = BEST_ALPHA
            retrieval_service._DIAG_NUTRITION_WEIGHT = nut_w
            retrieval_service._DIAG_RRF_WEIGHT = search_w
            retrieval_service._DIAG_BM25_RATIO = bm25_ratio
            retrieval_service._DIAG_FAISS_RATIO = faiss_ratio

            for text, rating, consumed in FEEDBACK_SEQUENCE:
                analyze_feedback("", text, rating, consumed)

            prefs = load_preferences()
            liked_words = [k for k, v in prefs.items() if v > BEST_THRESHOLD1]

            all_top3 = []
            titles   = []
            for meal_type, targets in MEAL_TARGETS.items():
                top3 = _patched_get_top_recipes_for_combined(
                    meal_type=meal_type,
                    targets=targets,
                    prefs=prefs,
                    n_feedback=N_FEEDBACK,
                    top_n=3,
                    threshold1=BEST_THRESHOLD1,
                    threshold2=BEST_THRESHOLD2,
                )
                all_top3.extend((meal_type, r) for r in top3)
                titles.extend(r.get("title", "") for r in top3)

            nut_scores = [
                nutrition_closeness(r, MEAL_TARGETS[mt]["calories"], MEAL_TARGETS[mt]["protein"])
                for mt, r in all_top3
            ]
            nutrition_fit = sum(nut_scores) / len(nut_scores) if nut_scores else 0.0

            pref_hits = sum(1 for _, r in all_top3 if contains_liked_ingredient(r, liked_words))
            preference_fit = pref_hits / len(all_top3) if all_top3 else 0.0

            overall_score = (nutrition_fit + preference_fit) / 2.0
            exp7_metrics[(nut_w, bm25_ratio)] = (nutrition_fit, preference_fit, overall_score)
            exp7_titles[(nut_w, bm25_ratio)]  = titles
            print(f"  nutrition_fit={nutrition_fit:.4f}  preference_fit={preference_fit:.4f}  overall_score={overall_score:.4f}")

    # Restore all four diagnostic globals and EMA_ALPHA to their real defaults
    # so nothing leaks if this module is imported elsewhere afterward.
    retrieval_service._DIAG_NUTRITION_WEIGHT = _LIVE_NUTRITION_WEIGHT
    retrieval_service._DIAG_RRF_WEIGHT = _LIVE_RRF_WEIGHT
    retrieval_service._DIAG_BM25_RATIO = _LIVE_BM25_RATIO
    retrieval_service._DIAG_FAISS_RATIO = _LIVE_FAISS_RATIO
    feedback_service.EMA_ALPHA = 0.5

    # 5x5 table: rows = nutrition/search weight pairs, columns = BM25/FAISS splits
    _table_width = 16 * (1 + len(BM25_FAISS_SPLITS))
    print("\n" + "=" * _table_width)
    col_labels = [f"{br:.2f}/{fr:.2f}" for br, fr in BM25_FAISS_SPLITS]
    print(f"{'Nut/Search W':<16}" + "".join(f"{label:<16}" for label in col_labels))
    print("-" * _table_width)
    for nut_w, search_w in NUTRITION_SEARCH_WEIGHTS:
        row_label = f"{nut_w:.2f}/{search_w:.2f}"
        row_cells = "".join(
            f"{exp7_metrics[(nut_w, br)][2]:<16.4f}" for br, fr in BM25_FAISS_SPLITS
        )
        print(f"{row_label:<16}{row_cells}")
    print("-" * _table_width)

    # Search sensitivity: for each nutrition/search weight, how many of the 9
    # recipe slots actually change title when the BM25/FAISS split changes.
    def _slots_differing(nut_w):
        variants = [exp7_titles[(nut_w, br)] for br, fr in BM25_FAISS_SPLITS]
        n_variants = len(variants)
        return sum(1 for slot in range(9) if len({variants[v][slot] for v in range(n_variants)}) > 1)

    print("\nSearch sensitivity (of 9 recommended slots, how many change title when the BM25/FAISS split changes):")
    exp7_sensitivity = {}
    for nut_w, search_w in NUTRITION_SEARCH_WEIGHTS:
        sens = _slots_differing(nut_w)
        exp7_sensitivity[nut_w] = sens
        print(f"  nutrition_weight={nut_w:.2f} (search_weight={search_w:.2f}): {sens}/9 slots differ across BM25/FAISS splits")

    # Best overall combination across the full 3x3 grid
    best_combo_key = max(exp7_metrics, key=lambda k: exp7_metrics[k][2])
    best_combo_nut_w, best_combo_bm25 = best_combo_key
    best_combo_search_w = next(sw for nw, sw in NUTRITION_SEARCH_WEIGHTS if nw == best_combo_nut_w)
    best_combo_faiss    = next(fr for br, fr in BM25_FAISS_SPLITS if br == best_combo_bm25)
    best_combo_score    = exp7_metrics[best_combo_key][2]

    print(f"\n(1) Best overall combination across all {len(NUTRITION_SEARCH_WEIGHTS) * len(BM25_FAISS_SPLITS)}: "
          f"nutrition/search={best_combo_nut_w:.2f}/{best_combo_search_w:.2f}, "
          f"bm25/faiss={best_combo_bm25:.2f}/{best_combo_faiss:.2f} (overall_score={best_combo_score:.4f})")

    # (2) Best trade-off point: the FIRST nutrition/search weight level (in the
    # order tested, highest nutrition weight first) where search sensitivity
    # reaches the "meaningfully non-trivial" threshold, paired with its best
    # BM25/FAISS split at that level.
    trade_off_level = next(
        ((nw, sw) for nw, sw in NUTRITION_SEARCH_WEIGHTS if exp7_sensitivity[nw] >= SENSITIVITY_THRESHOLD),
        None,
    )
    if trade_off_level is None:
        print(f"(2) Best trade-off point: none of the tested nutrition/search weight levels reached "
              f"search sensitivity >= {SENSITIVITY_THRESHOLD}/9 -- the BM25/FAISS split stayed "
              f"low-impact across the whole range tested.")
    else:
        to_nut_w, to_search_w = trade_off_level
        best_split_at_tradeoff = max(BM25_FAISS_SPLITS, key=lambda s: exp7_metrics[(to_nut_w, s[0])][2])
        print(f"(2) Best trade-off point: nutrition_weight={to_nut_w:.2f}/search_weight={to_search_w:.2f} "
              f"is the first level where search sensitivity reaches >= {SENSITIVITY_THRESHOLD}/9 "
              f"({exp7_sensitivity[to_nut_w]}/9 slots differ). Best BM25/FAISS split at that level: "
              f"bm25/faiss={best_split_at_tradeoff[0]:.2f}/{best_split_at_tradeoff[1]:.2f} "
              f"(overall_score={exp7_metrics[(to_nut_w, best_split_at_tradeoff[0])][2]:.4f}).")

    results_path = os.path.join(OUTPUT_DIR, "exp7_combined_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "note": "Reporting/verification only -- live defaults in retrieval_service.py were NOT modified.",
            "best_alpha_used": BEST_ALPHA,
            "best_threshold1_used": BEST_THRESHOLD1,
            "best_threshold2_used": BEST_THRESHOLD2,
            "nutrition_search_weights_tested": NUTRITION_SEARCH_WEIGHTS,
            "bm25_faiss_splits_tested": BM25_FAISS_SPLITS,
            "grid": [
                {
                    "nutrition_weight": nw, "search_weight": sw,
                    "bm25_ratio": br, "faiss_ratio": fr,
                    "nutrition_fit": exp7_metrics[(nw, br)][0],
                    "preference_fit": exp7_metrics[(nw, br)][1],
                    "overall_score": exp7_metrics[(nw, br)][2],
                }
                for nw, sw in NUTRITION_SEARCH_WEIGHTS
                for br, fr in BM25_FAISS_SPLITS
            ],
            "search_sensitivity_slots_out_of_9": {
                f"{nw:.2f}": exp7_sensitivity[nw] for nw, sw in NUTRITION_SEARCH_WEIGHTS
            },
            "best_combination": {
                "nutrition_weight": best_combo_nut_w,
                "search_weight": best_combo_search_w,
                "bm25_ratio": best_combo_bm25,
                "faiss_ratio": best_combo_faiss,
                "overall_score": best_combo_score,
            },
            "sensitivity_threshold_used": SENSITIVITY_THRESHOLD,
            "best_trade_off_point": (
                None if trade_off_level is None else {
                    "nutrition_weight": to_nut_w,
                    "search_weight": to_search_w,
                    "sensitivity_slots_out_of_9": exp7_sensitivity[to_nut_w],
                    "best_bm25_ratio": best_split_at_tradeoff[0],
                    "best_faiss_ratio": best_split_at_tradeoff[1],
                    "overall_score": exp7_metrics[(to_nut_w, best_split_at_tradeoff[0])][2],
                }
            ),
        }, f, indent=2)
    print(f"\nSaved JSON results to {results_path}")

    # Chart: Nutrition/Search Weight x BM25/FAISS Split heatmap (5x5)
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    grid_matrix = [
        [exp7_metrics[(nw, br)][2] for br, fr in BM25_FAISS_SPLITS]
        for nw, sw in NUTRITION_SEARCH_WEIGHTS
    ]
    flat_scores = [v for row in grid_matrix for v in row]
    vmin, vmax = min(flat_scores), max(flat_scores)
    vmid = (vmin + vmax) / 2
    im = ax.imshow(grid_matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(BM25_FAISS_SPLITS)))
    ax.set_xticklabels([f"{br:.2f}/{fr:.2f}" for br, fr in BM25_FAISS_SPLITS], fontsize=9)
    ax.set_yticks(range(len(NUTRITION_SEARCH_WEIGHTS)))
    ax.set_yticklabels([f"{nw:.2f}/{sw:.2f}" for nw, sw in NUTRITION_SEARCH_WEIGHTS], fontsize=9)
    ax.set_xlabel("BM25 Ratio / FAISS Ratio")
    ax.set_ylabel("Nutrition Weight / Search Weight")
    ax.set_title("Exp 7: Combined Sweep (5x5) -- Overall Score (reporting only)")
    for i, row in enumerate(grid_matrix):
        for j, val in enumerate(row):
            text_color = "white" if val < vmid else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(im, ax=ax, label="Overall Score")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "exp7_combined_sweep_results.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved plot: {chart_path}")


if __name__ == "__main__":
    main()
