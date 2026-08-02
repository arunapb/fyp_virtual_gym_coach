"""
history_service.py
==================
Persists recommendation history to data/recommendation_history.json.

Purpose:
  - Avoid recommending the same recipe twice on the same day
  - Recipes are only excluded within the same calendar day
  - On the next day they can be suggested again
"""

import json
import logging
import os
from datetime import date, datetime
from typing import List, Set

logger = logging.getLogger(__name__)

_SERVICES_DIR   = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR    = os.path.dirname(_SERVICES_DIR)
_HISTORY_FILE   = os.path.join(_BACKEND_DIR, "data", "recommendation_history.json")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load() -> list:
    if not os.path.exists(_HISTORY_FILE):
        return []
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            data = json.loads(content) if content else []
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(history: list) -> None:
    os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
    with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _normalize(title: str) -> str:
    return title.strip().lower()


# ── Public API ────────────────────────────────────────────────────────────────

def get_today_excluded_titles(user_id: str) -> Set[str]:
    """
    Return a set of normalised recipe titles already recommended
    to this user today — used to prevent same-day duplicates.
    """
    today    = date.today().isoformat()
    history  = _load()
    excluded = set()

    for record in history:
        if record.get("user_id") == user_id and record.get("date") == today:
            title = record.get("normalized_title") or _normalize(record.get("recipe_title", ""))
            if title:
                excluded.add(title)

    return excluded


def save_recommendations(user_id: str, meal_plan: dict) -> None:
    """
    Append all recipes in the meal plan to recommendation_history.json.
    Called after get_daily_meal_plan() succeeds.

    meal_plan format:
        {
          "Breakfast": {"target_calories": int, "recipes": [{"title": ..., ...}]},
          ...
        }
    """
    history = _load()
    now     = datetime.now()

    for meal_type, slot in meal_plan.items():
        target_calories = slot.get("target_calories", 0)
        for recipe in slot.get("recipes", []):
            title = recipe.get("title", "")
            if not title:
                continue
            history.append({
                "date":             now.date().isoformat(),
                "time":             now.strftime("%H:%M"),
                "user_id":          user_id,
                "meal_type":        meal_type,
                "recipe_title":     title,
                "normalized_title": _normalize(title),
                "target_calories":  target_calories,
            })

    _save(history)
    logger.info("[history_service] Saved %d history entries for user '%s'.", 
                sum(len(s.get("recipes", [])) for s in meal_plan.values()), user_id)


def load_history() -> list:
    """Return the full recommendation history log."""
    return _load()


def clear_history() -> None:
    """Wipe the entire history file (useful for testing)."""
    _save([])
    logger.info("[history_service] History cleared.")
