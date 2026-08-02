"""
run_retrieval_test.py
=====================
End-to-end test of the retrieval pipeline.

Usage:
    python "filtering_module - Copy/run_retrieval_test.py"

Make sure build_recipe_index.py has been run first.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import os
import traceback
from datetime import date as today_date_type

EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
COMBINE_DIR = os.path.dirname(EVAL_DIR)
BACKEND_DIR = os.path.join(COMBINE_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

PREF_FILE = os.path.join(BACKEND_DIR, "data", "user_preference.json")
PROF_FILE = os.path.join(BACKEND_DIR, "data", "user_profile.json")

print("Loading preferences ...")
if os.path.exists(PREF_FILE):
    with open(PREF_FILE, "r") as f:
        prefs = json.load(f)
else:
    prefs = {}

print("Loading user profile ...")
with open(PROF_FILE, "r") as f:
    user_profile = json.load(f)

from services.exercise_service import load_exercise_logs, get_previous_7_day_average

ACTIVITY_MULTIPLIERS = {
    "sedentary":         1.2,
    "lightly_active":    1.375,
    "moderately_active": 1.55,
    "very_active":       1.725,
}

MEAL_SPLITS = {
    "Breakfast": 0.25,
    "Lunch":     0.40,
    "Dinner":    0.35,
}


def calculate_targets(p):
    weight   = p["weight_kg"]
    height   = p["height_cm"]
    age      = p["age"]
    gender   = p["gender"]
    activity = p["activity_level"]
    goal     = p["goal"]

    if gender == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

    multiplier = ACTIVITY_MULTIPLIERS[activity]
    tdee = bmr * multiplier

    if goal == "weight_loss":
        base_daily_calories = tdee - 300
    elif goal == "weight_gain":
        base_daily_calories = tdee + 300
    else:
        base_daily_calories = tdee

    today = today_date_type.today()
    exercise_avg_7 = get_previous_7_day_average(today)

    if goal == "weight_gain":
        final_daily_calories = base_daily_calories + exercise_avg_7
    elif goal == "weight_loss":
        final_daily_calories = base_daily_calories
    else:
        final_daily_calories = base_daily_calories + exercise_avg_7

    total_calories = round(final_daily_calories)
    total_protein  = round(2 * weight)

    meals = {}
    for meal, split in MEAL_SPLITS.items():
        meals[meal] = {
            "calories": round(total_calories * split),
            "protein":  round(total_protein  * split),
        }

    return {
        "bmr":                 round(bmr, 2),
        "tdee":                round(tdee, 2),
        "base_daily_calories": round(base_daily_calories, 2),
        "exercise_avg_7":      exercise_avg_7,
        "final_daily_calories": total_calories,
        "total_calories":      total_calories,
        "total_protein":       total_protein,
        "meal_targets":        meals,
        "meals":               meals,
    }


result = calculate_targets(user_profile)

BMR                = result["bmr"]
TDEE               = result["tdee"]
BASE_CALORIES      = result["base_daily_calories"]
EXERCISE_AVG_7     = result["exercise_avg_7"]
TOTAL_CALORIES     = result["total_calories"]
TOTAL_PROTEIN      = result["total_protein"]
meal_targets       = result["meal_targets"]

print(f"\nCalorie Targets:")
print(f"  BMR                  : {BMR} kcal")
print(f"  TDEE                 : {TDEE} kcal")
print(f"  Base Daily Calories  : {BASE_CALORIES} kcal")
print(f"  Exercise Avg (7-day) : {EXERCISE_AVG_7} kcal")
print(f"  Final Daily Calories : {TOTAL_CALORIES} kcal")
print(f"  Total Protein        : {TOTAL_PROTEIN}g")

try:
    from services.retrieval_service import get_top_recipes
except Exception:
    print("\n[FATAL] Could not import services.retrieval_service:")
    traceback.print_exc()
    raise

from services.history_service import (
    get_today_excluded_titles,
    save_recommendations,
)

# ── Print header ──────────────────────────────────────────────────────────────
pref_str_parts = []
for k, v in prefs.items():
    pref_str_parts.append(f"{k}{'+' if v >= 0 else ''}{v:.2f}")
pref_display = "  ".join(pref_str_parts)

print("\nRETRIEVAL RESULTS")
print(f"Preferences: {pref_display}")
print("-" * 60)

# ── Run retrieval and collect results ─────────────────────────────────────────
all_top_recipes = {}   # meal_type -> list of recipe dicts

USER_ID = user_profile.get("user_id", "user_001")

# load titles already shown to this user today (same-day exclusion only)
# recipes are not permanently blocked -- next day they can appear again
today_history_excluded = set(get_today_excluded_titles(USER_ID))

# also track cross-meal duplicates within this single run
used_titles = set()

# merge both exclusion sources into one set passed to the retriever
exclude_titles = today_history_excluded | used_titles

print(f"\n[History] Titles excluded from today's history: {len(today_history_excluded)}")

for meal_type, targets in meal_targets.items():
    try:
        top3 = get_top_recipes(
            meal_type=meal_type,
            targets=targets,
            prefs=prefs,
            n_feedback=10,
            top_n=3,
            exclude_titles=exclude_titles,
        )
        all_top_recipes[meal_type] = top3

        if top3:
            save_recommendations(
                user_id=USER_ID,
                meal_plan={
                    meal_type: {
                        "target_calories": targets["calories"],
                        "recipes": [top3[0]],
                    }
                }
            )

        # add top-1 title to cross-meal and combined exclusion sets
        if top3:
            t = top3[0]["title"]
            used_titles.add(t)
            exclude_titles = today_history_excluded | used_titles

    except Exception:
        print(f"\n[ERROR] Retrieval failed for {meal_type}:")
        traceback.print_exc()
        all_top_recipes[meal_type] = []

# ── Display per-meal results ──────────────────────────────────────────────────
def find_ingredient_matches(recipe, words):
    ingredients = recipe.get("ingredients", [])
    title = recipe.get("title", "")
    text = (title + " " + " ".join(ingredients)).lower()
    matches = []
    for word in words:
        if word.lower() in text:
            matches.append(word)
    return matches


LIKED_CHECK = [
    ingredient for ingredient, score in prefs.items() if score > 0.30
]
DISLIKED_CHECK = [
    ingredient for ingredient, score in prefs.items() if score < -0.30
]

for meal_type, targets in meal_targets.items():
    recipes = all_top_recipes[meal_type]
    cal_tgt = targets["calories"]
    pro_tgt = targets["protein"]

    print(f"\n{meal_type.upper()}  (target: {cal_tgt} cal | {pro_tgt}g protein)")
    print("-" * 60)

    liked_match_count    = 0
    disliked_avoid_count = 0

    for rank, recipe in enumerate(recipes, start=1):
        title       = recipe.get("title",      "Unknown")
        category    = recipe.get("category",   "")
        state       = recipe.get("state",      "Unknown")
        calories    = recipe.get("calories",   0.0)
        protein     = recipe.get("protein",    0.0)
        rating      = recipe.get("rating",     0.0)
        rating_cnt  = recipe.get("rating_count", 0)
        ingredients = recipe.get("ingredients", [])
        rrf_score   = recipe.get("rrf_score",   0.0)
        final_score = recipe.get("final_score", rrf_score)

        cal_delta  = calories - cal_tgt
        prot_delta = protein  - pro_tgt
        cal_sign   = "+" if cal_delta  >= 0 else ""
        prot_sign  = "+" if prot_delta >= 0 else ""

        liked_matches    = find_ingredient_matches(recipe, LIKED_CHECK)
        disliked_matches = find_ingredient_matches(recipe, DISLIKED_CHECK)

        if liked_matches:
            liked_match_count += 1
        if not disliked_matches:
            disliked_avoid_count += 1

        print(f"\nRank {rank}: {title}")
        print(f"  Category:     {category}")
        print(f"  State:        {state}")
        print(f"  Calories:     {calories:.0f} kcal  "
              f"(target {cal_tgt}  {cal_sign}{cal_delta:.0f})")
        print(f"  Protein:      {protein:.1f}g  "
              f"(target {pro_tgt}g  {prot_sign}{prot_delta:.1f}g)")
        print(f"  Rating:       {rating:.1f}  ({rating_cnt} ratings)")
        print(
            "  Liked ingredient matches:    "
            + (", ".join(liked_matches) if liked_matches else "none")
        )
        print(
            "  Disliked ingredient matches: "
            + (", ".join(disliked_matches) if disliked_matches else "none")
        )
        print(f"  RRF Score:    {rrf_score:.4f}")
        print(f"  Final Score:  {final_score:.4f}")

        explanation = recipe.get("explanation", [])
        if explanation:
            print(f"  Why recommended:")
            for reason in explanation:
                print(f"    - {reason}")

    print()
    print(f"  Recipes containing liked ingredients:   {liked_match_count}/{len(recipes)}")
    print(f"  Recipes avoiding disliked ingredients:  {disliked_avoid_count}/{len(recipes)}")


# ── Summary table ─────────────────────────────────────────────────────────────
print("\nDAILY MEAL PLAN")
print("-" * 65)
print(f"{'Meal':<12} {'Recipe':<25} {'Cal':>6}  {'Target':>6}  {'dcal':>6}  {'Prot':>5}")
print("-" * 65)

total_cal = 0.0
total_pro = 0.0

for meal_type, tgt in meal_targets.items():
    recipes = all_top_recipes[meal_type]
    tgt_cal = tgt["calories"]
    if recipes:
        best    = recipes[0]
        r_title = best["title"][:23]
        r_cal   = best["calories"]
        r_pro   = best["protein"]
    else:
        r_title = "N/A"
        r_cal   = 0.0
        r_pro   = 0.0

    delta  = r_cal - tgt_cal
    d_sign = "+" if delta >= 0 else ""
    total_cal += r_cal
    total_pro += r_pro
    print(f"{meal_type:<12} {r_title:<25} {r_cal:>6.0f}  {tgt_cal:>6}  {d_sign}{delta:>5.0f}  {r_pro:>5.1f}g")

print("-" * 65)
print(f"{'Total':<12} {'':25} {total_cal:>6.0f}  {total_pro:>6.1f}g")
print(f"{'Target':<12} {'':25} {TOTAL_CALORIES:>6}  {TOTAL_PROTEIN:>6}g")

cal_match = (total_cal / TOTAL_CALORIES * 100) if TOTAL_CALORIES else 0
pro_match = (total_pro / TOTAL_PROTEIN  * 100) if TOTAL_PROTEIN  else 0
print(f"{'Match %':<12} {'':25} {cal_match:>5.0f}%  {pro_match:>5.0f}%")
print("-" * 65)

# ── Preference summary for selected meals ─────────────────────────────────────
plan_liked         = 0
plan_dislike_avoid = 0
plan_total         = 0
for mt in meal_targets:
    best_list = all_top_recipes.get(mt, [])
    if best_list:
        best = best_list[0]
        plan_total += 1
        if find_ingredient_matches(best, LIKED_CHECK):
            plan_liked += 1
        if not find_ingredient_matches(best, DISLIKED_CHECK):
            plan_dislike_avoid += 1

print(f"  Selected recipes with liked ingredients:   {plan_liked}/{plan_total}")
print(f"  Selected recipes avoiding disliked:        {plan_dislike_avoid}/{plan_total}")

# ── State diversity info ───────────────────────────────────────────────────────
# print("\nStates in recommendations:")
# for meal, recipes in all_top_recipes.items():
#     if recipes:
#         state = recipes[0].get("state", "Unknown")
#         print(f"  {meal}: {state}")
#     else:
#         print(f"  {meal}: N/A")

# ── Diversity check ───────────────────────────────────────────────────────────
print("\nDiversity check")

titles = [
    all_top_recipes[m][0]["title"]
    for m in meal_targets
    if all_top_recipes.get(m)
]
states = [
    all_top_recipes[m][0].get("state", "")
    for m in meal_targets
    if all_top_recipes.get(m)
]

# if len(set(titles)) == len(meal_targets):
#     print("  Titles: all different [OK]")
# else:
#     dupes = [t for t in titles if titles.count(t) > 1]
#     print(f"  Titles: duplicates found [FAIL]  ({set(dupes)})")

# if len(set(states)) >= 2:
#     print("  States: regional diversity [OK]")
# else:
#     print(f"  States: same region [WARN]  ({set(states)})")
