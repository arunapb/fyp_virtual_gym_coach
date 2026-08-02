"""
goal_service.py
================
Goal Weight & Pace calculations for the nutrition module.

Single source of truth for the daily kcal deficit/surplus values used across
the app (pace options here AND the base calorie adjustment in
recommender_service.calculate_targets both read from PACE_OPTIONS below).
"""

from typing import Optional

KCAL_PER_KG = 7700  # ~7700 kcal stored per 1kg of body fat

PACE_OPTIONS = [
    {"key": "gentle",   "label": "Gentle",   "kcal_per_day": 200},
    {"key": "balanced", "label": "Balanced", "kcal_per_day": 250},
    {"key": "faster",   "label": "Faster",   "kcal_per_day": 300},
]
DEFAULT_PACE = "balanced"
PACE_KEYS = {opt["key"] for opt in PACE_OPTIONS}


def get_pace_kcal(pace_key: Optional[str]) -> int:
    """Return the daily kcal deficit/surplus for a pace key (falls back to the default pace)."""
    for opt in PACE_OPTIONS:
        if opt["key"] == pace_key:
            return opt["kcal_per_day"]
    return next(o["kcal_per_day"] for o in PACE_OPTIONS if o["key"] == DEFAULT_PACE)


def calculate_goal_plan(current_weight_kg: float, goal_weight_kg: float) -> dict:
    """
    Compute kg to change, direction, and the 3 pace options (weeks/months estimate)
    for moving from current_weight_kg to goal_weight_kg.
    """
    if goal_weight_kg is None or goal_weight_kg <= 0:
        raise ValueError("goal_weight_kg must be a positive number")

    kg_to_change = abs(goal_weight_kg - current_weight_kg)
    is_loss = goal_weight_kg < current_weight_kg
    already_at_goal = kg_to_change < 0.01

    pace_options = []
    for opt in PACE_OPTIONS:
        kcal = opt["kcal_per_day"]
        weekly_rate_kg = (kcal * 7) / KCAL_PER_KG
        estimated_weeks = 0 if already_at_goal else kg_to_change / weekly_rate_kg
        pace_options.append({
            "key":               opt["key"],
            "label":             opt["label"],
            "kcal_per_day":      kcal,
            "weekly_rate_kg":    round(weekly_rate_kg, 3),
            "estimated_weeks":   round(estimated_weeks),
            "estimated_months":  round(estimated_weeks / 4.345, 1),
            "is_default":        opt["key"] == DEFAULT_PACE,
        })

    return {
        "current_weight_kg": current_weight_kg,
        "goal_weight_kg":    goal_weight_kg,
        "kg_to_change":      round(kg_to_change, 2),
        "is_loss":           is_loss,
        "already_at_goal":   already_at_goal,
        "pace_options":      pace_options,
    }
