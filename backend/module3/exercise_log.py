"""
backend/module3/exercise_log.py — the hand-off from a finished workout to
Module 3's nutrition maths.

What this replaces
------------------
`module1/app.py` (Module 1's standalone desktop demo) ended every run with:

        # Export exercise durations for Module 3
        exercise_durations_seconds = {ex: frames / 30.0 for ex, frames in ...}
        with open('data/session_summary.json', 'w') as f:
            json.dump(exercise_durations_seconds, f, indent=4)

— a `{exercise_name: seconds}` dump that a human then had to retype into
Module 3. Nothing consumed it automatically, and its shape matched nothing on
Module 3's side.

This module closes that loop. The same measurement now goes straight through
`exercise_service.calculate_exercise_calories()` and is appended to
`module3/backend/data/exercise_logs.json` in Module 3's OWN record shape:

        {"date", "exercise_name", "duration_minutes", "calories_burned"}

which is exactly `models.schemas.ExerciseLogEntry` — the same record
`POST /exercise` writes — so `get_previous_7_day_average()` picks it up and it
flows into `recommender_service.calculate_targets()`'s `exercise_avg_7` on the
next meal plan. The workout you just analysed changes what you get told to eat.

Field-by-field, against Module 3's parameters:

    date              today, ISO (`YYYY-MM-DD`) — the format
                      `get_previous_7_day_average()` parses with `strptime`
    exercise_name     Module 1's own label, unchanged. `config.TARGET_EXERCISES`
                      is ['squat', 'bicep_curl', 'deadlift', 'shoulder_press']
                      and `exercise_service.MET_VALUES` is keyed by exactly
                      those four strings — they already line up, so there is no
                      translation table here and none is needed. (Module 2's
                      CamelCase registry keys — 'Squat', 'BicepCurl' — would
                      have needed one, which is a second reason to source the
                      durations from Module 1.)
    duration_minutes  seconds / 60. Module 3 takes minutes; Module 1 measured
                      seconds.
    calories_burned   `calculate_exercise_calories()`, called not reimplemented
                      (MET x 3.5 x kg / 200 x minutes).
    weight_kg         read from Module 3's live profile, matching what
                      `POST /exercise` does when the caller omits it.

Duration accuracy: the seconds handed in are derived from FRAME COUNTS and the
video's fps, not from `GymCoachSession.exercise_durations`. That attribute
accumulates off `time.time()`, which is correct for a real-time webcam but
meaningless here — this pipeline pushes frames through as fast as the CPU
allows, so it would measure how long the SERVER worked, not how long the USER
exercised. frames / fps is the true in-video duration (and is what app.py's
`frames / 30.0` was approximating). See backend/merged_analysis.py.
"""

import logging
from datetime import date

from backend import user_store
from backend.module3 import bridge

log = logging.getLogger("module3.exercise_log")

# Below this, a "set" is a classifier flicker, not exercise. Module 1's
# confirmed-exercise lock needs 15-90 consecutive confident frames, so a
# genuine set clears this comfortably; a one-off blip on the way into or out of
# a movement does not. Without a floor, every run would append a handful of
# sub-second rows that add nothing but skew the 7-day average's day count.
MIN_LOGGED_SECONDS = 2.0


def build_entries(durations_seconds, *, weight_kg, day=None,
                  min_seconds=MIN_LOGGED_SECONDS):
    """
    Convert `{exercise_name: seconds}` into Module 3 exercise-log records.

    Pure: computes and returns, writes nothing. Separated from `save_session`
    so the conversion can be tested (and previewed in the API response)
    without touching the log file.
    """
    ex_svc = bridge.exercise()
    day_iso = (day or date.today()).isoformat()

    entries = []
    for name, seconds in sorted(durations_seconds.items()):
        if not name or seconds < min_seconds:
            continue
        minutes = round(seconds / 60.0, 2)
        if minutes <= 0:      # ExerciseLogRequest declares duration_minutes > 0
            continue
        entries.append({
            "date": day_iso,
            "exercise_name": name,
            "duration_minutes": minutes,
            "calories_burned": ex_svc.calculate_exercise_calories(
                name, minutes, weight_kg),
        })
    return entries


def save_session(durations_seconds, *, username=None, day=None,
                 min_seconds=MIN_LOGGED_SECONDS):
    """
    Log one finished workout to Module 3 and return the records written.

    Called at the end of every analysis — including one the user stopped early
    — from backend/merged_analysis.py.

    `username` is whoever uploaded the video (carried from the stream socket,
    see backend/main.py). The write is wrapped in `user_store.as_user()` so it
    lands in THAT person's exercise_logs.json, and so a browser request that
    switches the active user while the analysis runs cannot misfile it. Their
    body weight is read inside the same block, for the same reason. With no
    username the entry goes to Module 3's own shared log, as before.

    Never raises. A failure here (Module 3 absent, profile unreadable, log file
    locked) must not fail an analysis that has already completed successfully
    and whose video, CSVs and result document are all on disk; the workout
    results are the deliverable, the nutrition log is a bonus. Failures are
    logged and reported to the browser as an empty list.
    """
    if not durations_seconds:
        return []

    try:
        with user_store.as_user(username):
            weight_kg = bridge.profile().get_profile().get("weight_kg") or 70
            entries = build_entries(durations_seconds, weight_kg=weight_kg,
                                    day=day, min_seconds=min_seconds)
            ex_svc = bridge.exercise()
            for entry in entries:
                ex_svc.save_exercise_log(entry)
        if entries:
            log.info("[module3] logged %d exercise entr%s for %s: %s",
                     len(entries), "y" if len(entries) == 1 else "ies",
                     username or "the shared profile",
                     ", ".join(f"{e['exercise_name']} "
                               f"{e['duration_minutes']}min "
                               f"{e['calories_burned']}kcal" for e in entries))
        return entries
    except Exception as exc:                       # deliberately broad — see docstring
        log.warning("[module3] could not log this session's exercise: %s", exc)
        return []
