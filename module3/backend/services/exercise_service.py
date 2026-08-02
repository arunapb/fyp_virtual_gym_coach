"""
exercise_service.py
===================
MET-based calorie calculation, exercise log persistence,
and 7-day rolling average — ported from Reccomendation/exercise_service.py.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGS_PATH = os.path.join(_BACKEND_DIR, "data", "exercise_logs.json")

MET_VALUES = {
    "squat":          5.0,
    "deadlift":       5.0,
    "bicep_curl":     3.5,
    "shoulder_press": 3.5,
}


def calculate_exercise_calories(exercise_name: str, duration_minutes: float, weight_kg: float) -> float:
    """Return kilocalories burned using the MET formula."""
    met = MET_VALUES.get(exercise_name.lower(), 3.5)
    calories = met * 3.5 * weight_kg / 200 * duration_minutes
    return round(calories, 2)


def load_exercise_logs() -> List[dict]:
    """Load all saved exercise log entries."""
    if not os.path.exists(_LOGS_PATH):
        return []
    with open(_LOGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_exercise_log(log_entry: dict) -> None:
    """Append a single log entry to the exercise log file."""
    logs = load_exercise_logs()
    logs.append(log_entry)
    os.makedirs(os.path.dirname(_LOGS_PATH), exist_ok=True)
    with open(_LOGS_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def get_previous_7_day_average(today_date=None) -> float:
    """Return average calories burned per day over the previous 7 days."""
    logs = load_exercise_logs()
    if today_date is None:
        from datetime import date
        today_date = date.today()

    if isinstance(today_date, str):
        today_date = datetime.strptime(today_date, "%Y-%m-%d").date()

    calories_by_day = defaultdict(float)
    for log in logs:
        log_date = datetime.strptime(log["date"], "%Y-%m-%d").date()
        days_ago = (today_date - log_date).days
        if 1 <= days_ago <= 7:
            calories_by_day[log_date] += log.get("calories_burned", 0)

    total = sum(calories_by_day.values())
    return round(total / 7, 2)
