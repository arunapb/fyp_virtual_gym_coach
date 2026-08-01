"""
recommender_service.py
======================
Daily meal plan assembly — pure functions, no script-level execution.

Reads the user profile and preferences from disk, calculates nutrition
targets, then calls retrieval_service.get_top_recipes() for each meal slot.
"""

import logging
from collections import defaultdict
from datetime import date as _date_type

from . import exercise_service, goal_service, history_service, retrieval_service
from .feedback_service import load_preferences
from .profile_service import get_profile

logger = logging.getLogger(__name__)

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


# ── Nutrition target calculation ───────────────────────────────────────────────

def calculate_targets(profile: dict) -> dict:
    """
    Compute BMR → TDEE → daily calorie / protein targets and per-meal splits.

    Returns:
        {
            bmr, tdee, base_daily_calories, exercise_avg_7,
            total_calories, total_protein,
            meal_targets: {Breakfast: {calories, protein}, ...}
        }
    """
    weight   = profile["weight_kg"]
    height   = profile["height_cm"]
    age      = profile["age"]
    gender   = profile["gender"]
    activity = profile["activity_level"]
    goal     = profile["goal"]

    # Harris-Benedict BMR
    if gender.lower() == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

    multiplier = ACTIVITY_MULTIPLIERS.get(activity, 1.2)
    tdee       = bmr * multiplier

    # Daily kcal deficit/surplus comes from the user's selected Goal Weight & Pace
    # option (gentle/balanced/faster -- see goal_service.PACE_OPTIONS), defaulting
    # to "balanced" when no pace has been chosen yet. This is the single source of
    # truth for the adjustment applied here.
    pace_adjustment = goal_service.get_pace_kcal(profile.get("selected_pace"))

    if goal == "weight_loss":
        base_daily_calories = tdee - pace_adjustment
    elif goal == "weight_gain":
        base_daily_calories = tdee + pace_adjustment
    else:
        base_daily_calories = tdee

    exercise_avg_7 = exercise_service.get_previous_7_day_average()

    if goal == "weight_gain":
        final_calories = base_daily_calories + exercise_avg_7
    elif goal == "weight_loss":
        final_calories = base_daily_calories
    else:
        final_calories = base_daily_calories + exercise_avg_7

    total_calories = round(final_calories)
    total_protein  = round(2 * weight)

    meal_targets = {
        meal: {
            "calories": round(total_calories * split),
            "protein":  round(total_protein  * split),
        }
        for meal, split in MEAL_SPLITS.items()
    }

    return {
        "bmr":                 round(bmr, 2),
        "tdee":                round(tdee, 2),
        "base_daily_calories": round(base_daily_calories, 2),
        "exercise_avg_7":      exercise_avg_7,
        "total_calories":      total_calories,
        "total_protein":       total_protein,
        "meal_targets":        meal_targets,
    }


# ── Daily meal plan ────────────────────────────────────────────────────────────

def get_daily_meal_plan(top_n: int = 3) -> dict:
    """
    Build a full daily meal plan for the current user profile + preferences.

    Returns:
        {
            "user_name":         str,
            "goal":              str,
            "nutrition_targets": {...},
            "meal_plan": {
                "Breakfast": {"target_calories": int, "target_protein": int, "recipes": [...]},
                "Lunch":     {...},
                "Dinner":    {...},
            },
            "active_preferences": {ingredient: score, ...},
        }
    """
    profile   = get_profile()
    targets   = calculate_targets(profile)
    prefs     = load_preferences()
    user_id   = profile.get("user_id", "user_001")

    # n_feedback drives the adaptive RRF weight (more feedback → prefer-based weight grows)
    n_feedback = len([v for v in prefs.values() if v != 0.0])

    # Exclude titles already recommended today (same-day deduplication)
    today_excluded = history_service.get_today_excluded_titles(user_id)
    used_titles: set = set(today_excluded)

    # Shared across all three get_top_recipes() calls below so the preferred-ingredient
    # cap (max 2 per ingredient) is enforced across the full day's plan, not per slot.
    daily_ingredient_counts: dict = {}

    meal_plan = {}

    for meal_type, slot_targets in targets["meal_targets"].items():
        recipes = retrieval_service.get_top_recipes(
            meal_type                = meal_type,
            targets                  = slot_targets,
            prefs                    = prefs,
            n_feedback               = n_feedback,
            top_n                    = top_n,
            exclude_titles           = used_titles,
            daily_ingredient_counts  = daily_ingredient_counts,
        )
        for r in recipes:
            used_titles.add(r.get("title", ""))

        meal_plan[meal_type] = {
            "target_calories": slot_targets["calories"],
            "target_protein":  slot_targets["protein"],
            "recipes":         recipes,
        }
        logger.info("[recommender] %s: %d recipes selected.", meal_type, len(recipes))

    # Persist this recommendation to history (prevents same-day repeats)
    history_service.save_recommendations(user_id, meal_plan)

    return {
        "user_name":          profile.get("name", "User"),
        "goal":               profile.get("goal", "maintenance"),
        "nutrition_targets":  targets,
        "meal_plan":          meal_plan,
        "active_preferences": prefs,
    }
