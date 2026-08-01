"""
profile_service.py
==================
Read and update user_profile.json stored in backend/data/.
"""

import json
import os

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROFILE_PATH = os.path.join(_BACKEND_DIR, "data", "user_profile.json")

_DEFAULT_PROFILE = {
    "user_id": "user_001",
    "name": "User",
    "age": 25,
    "gender": "male",
    "weight_kg": 70,
    "height_cm": 175,
    "activity_level": "moderately_active",
    "goal": "maintenance",
    "meals_per_day": 3,
    "allergies": [],
    "disliked_ingredients": [],
    "goal_weight_kg": None,
    "selected_pace": "balanced",
}


def get_profile() -> dict:
    """Load and return the current user profile."""
    if not os.path.exists(_PROFILE_PATH):
        _save_profile(_DEFAULT_PROFILE)
        return dict(_DEFAULT_PROFILE)
    with open(_PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def update_profile(updates: dict) -> dict:
    """Merge updates into the existing profile and save."""
    profile = get_profile()
    # Only update fields that are explicitly provided (not None)
    for key, value in updates.items():
        if value is not None:
            profile[key] = value
    _save_profile(profile)
    return profile


def _save_profile(profile: dict) -> None:
    os.makedirs(os.path.dirname(_PROFILE_PATH), exist_ok=True)
    with open(_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
