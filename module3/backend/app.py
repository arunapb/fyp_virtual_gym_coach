"""
app.py
======
Unified FastAPI entrypoint for the Meal Recommendation & Feedback System.
Reloaded with updated recipe_metadata containing full ingredients_raw and steps.

All ML models (BERT, BART, FAISS, SentenceTransformer, spaCy) load once on
startup. Every API request is served from memory.

Run with:
    uvicorn app:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

import logging
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ── configure logging before importing services (which log at import time) ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── import services (models load here at startup) ───────────────────────────
logger.info("Loading all services ...")
from services import exercise_service, feedback_service, goal_service, history_service, profile_service, recommender_service
from models.schemas import (
    AspectResult,
    ExerciseLogEntry,
    ExerciseLogRequest,
    ExerciseSummaryResponse,
    FeedbackRequest,
    FeedbackResponse,
    GoalOptionsRequest,
    GoalOptionsResponse,
    HealthResponse,
    MealSlot,
    NutritionTargets,
    PreferenceResponse,
    ProfileResponse,
    ProfileUpdateRequest,
    RecipeResult,
    RecommendationResponse,
    SelectPaceRequest,
    SelectPaceResponse,
)
logger.info("All services ready.")

# ── FastAPI application ──────────────────────────────────────────────────────
app = FastAPI(
    title="Meal Recommendation System API",
    description=(
        "Unified backend combining the NL-feedback pipeline and FAISS-based "
        "meal recommendation engine. Feedback instantly updates preferences "
        "which are reflected in the next /recommend call."
    ),
    version="1.0.0",
)

# Allow all origins so any frontend (React Native, web, Postman) can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════
# Health
# ════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health_check():
    """Returns OK if the server and all models are running."""
    return HealthResponse(status="ok", message="All systems operational.")


# ════════════════════════════════════════════════════════════════════════════
# Profile
# ════════════════════════════════════════════════════════════════════════════

def _with_goal_progress(profile: dict) -> dict:
    """
    Attach computed goal-progress fields (kg_to_change, is_loss, already_at_goal,
    estimated_weeks/months for the currently selected pace) to a profile dict.

    These are derived live from weight_kg/goal_weight_kg/selected_pace via
    goal_service -- never written back to user_profile.json, so they can never
    go stale relative to the stored values. No-op if goal_weight_kg isn't set.
    """
    out = dict(profile)
    goal_weight = profile.get("goal_weight_kg")
    if not goal_weight:
        return out
    try:
        plan = goal_service.calculate_goal_plan(profile["weight_kg"], goal_weight)
    except ValueError:
        return out

    pace_key = profile.get("selected_pace") or goal_service.DEFAULT_PACE
    pace_opt = next((o for o in plan["pace_options"] if o["key"] == pace_key), None)

    out["kg_to_change"]    = plan["kg_to_change"]
    out["is_loss"]         = plan["is_loss"]
    out["already_at_goal"] = plan["already_at_goal"]
    if pace_opt:
        out["estimated_weeks"]    = pace_opt["estimated_weeks"]
        out["estimated_months"]   = pace_opt["estimated_months"]
        out["pace_kcal_per_day"]  = pace_opt["kcal_per_day"]
    return out


@app.get(
    "/profile",
    response_model=ProfileResponse,
    summary="Get user profile",
    tags=["Profile"],
)
def get_profile():
    """Return the current user profile (loaded from data/user_profile.json)."""
    try:
        return _with_goal_progress(profile_service.get_profile())
    except Exception as e:
        logger.exception("Error loading profile")
        raise HTTPException(status_code=500, detail=str(e))


@app.put(
    "/profile",
    response_model=ProfileResponse,
    summary="Update user profile",
    tags=["Profile"],
)
def update_profile(body: ProfileUpdateRequest):
    """
    Update one or more user profile fields.
    Only fields explicitly provided in the request body are changed.
    """
    try:
        updated = profile_service.update_profile(body.model_dump(exclude_none=True))
        return _with_goal_progress(updated)
    except Exception as e:
        logger.exception("Error updating profile")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# Goal Weight & Pace
# ════════════════════════════════════════════════════════════════════════════

@app.post(
    "/goal/options",
    response_model=GoalOptionsResponse,
    summary="Calculate pace options for a goal weight",
    tags=["Goal"],
)
def get_goal_options(body: GoalOptionsRequest):
    """
    Given a goal weight, return the kg to change and 3 pace options (Gentle /
    Balanced / Faster) with their kcal/day adjustment and estimated timeframe.

    Also persists `goal_weight_kg` to the profile, and syncs the profile's
    `goal` field (weight_loss / weight_gain / maintenance) to match the
    direction implied by current vs. goal weight -- so the rest of the
    pipeline (calculate_targets) stays consistent with a single source of truth.
    """
    try:
        profile = profile_service.get_profile()
        current_weight = profile["weight_kg"]
        plan = goal_service.calculate_goal_plan(current_weight, body.goal_weight_kg)

        if plan["already_at_goal"]:
            new_goal = "maintenance"
        elif plan["is_loss"]:
            new_goal = "weight_loss"
        else:
            new_goal = "weight_gain"
        profile_service.update_profile({
            "goal_weight_kg": body.goal_weight_kg,
            "goal":           new_goal,
        })

        return GoalOptionsResponse(**plan)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error calculating goal options")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/goal/select-pace",
    response_model=SelectPaceResponse,
    summary="Select a pace option and recalculate nutrition targets",
    tags=["Goal"],
)
def select_pace(body: SelectPaceRequest):
    """
    Save the user's chosen pace (gentle/balanced/faster) and recalculate the
    daily calorie target, which flows straight into the existing
    Breakfast/Lunch/Dinner calorie-splitting logic.
    """
    if body.pace not in goal_service.PACE_KEYS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pace '{body.pace}'. Must be one of: {sorted(goal_service.PACE_KEYS)}",
        )
    try:
        profile_service.update_profile({"selected_pace": body.pace})
        profile = profile_service.get_profile()
        targets = recommender_service.calculate_targets(profile)

        return SelectPaceResponse(
            pace              = body.pace,
            kcal_per_day      = goal_service.get_pace_kcal(body.pace),
            new_daily_target  = targets["total_calories"],
            nutrition_targets = NutritionTargets(
                bmr            = targets["bmr"],
                tdee           = targets["tdee"],
                total_calories = targets["total_calories"],
                total_protein  = targets["total_protein"],
                exercise_avg_7 = targets["exercise_avg_7"],
                meal_targets   = targets["meal_targets"],
            ),
        )
    except Exception as e:
        logger.exception("Error selecting pace")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# Preferences
# ════════════════════════════════════════════════════════════════════════════

@app.get(
    "/preferences",
    response_model=PreferenceResponse,
    summary="Get ingredient preference vector",
    tags=["Preferences"],
)
def get_preferences():
    """
    Return the current ingredient preference scores.
    Values range from -1.0 (strongly disliked) to +1.0 (strongly liked).
    """
    prefs = feedback_service.load_preferences()
    return PreferenceResponse(preferences=prefs, count=len(prefs))


@app.delete(
    "/preferences",
    response_model=PreferenceResponse,
    summary="Reset all preferences",
    tags=["Preferences"],
)
def reset_preferences():
    """Clear the entire preference vector (fresh start)."""
    feedback_service.reset_preferences()
    return PreferenceResponse(preferences={}, count=0)


# ════════════════════════════════════════════════════════════════════════════
# Feedback
# ════════════════════════════════════════════════════════════════════════════

@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit meal feedback",
    tags=["Feedback"],
)
def submit_feedback(body: FeedbackRequest):
    """
    Process natural-language feedback for a meal.

    Pipeline:
    1. Causal detection (spaCy) — isolates the cause clause
    2. Intent classification (BART zero-shot) — REDUCE / INCREASE / KEEP
    3. BERT ABSA — extract aspect + sentiment pairs
    4. Intensity scoring (VADER) — LOW / MODERATE / HIGH
    5. Delta scoring — updates user_preference.json

    The preference update is immediate — the next `GET /recommend` call
    will reflect the new preferences.
    """
    try:
        result = feedback_service.analyze_feedback(
            feedback_text = body.feedback_text,
        )

        aspects_out = [
            AspectResult(ingredient=name, action=action, delta=round(delta, 4))
            for name, action, delta in result["aspects"]
        ]

        return FeedbackResponse(
            feedback_text        = body.feedback_text,
            causal               = result["causal"],
            cause_ingredient     = result["cause_ingredient"],
            mixed                = result["mixed"],
            aspects              = aspects_out,
            updated_preferences  = result["updated_preferences"],
        )
    except Exception as e:
        logger.exception("Error processing feedback")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# Recommendation
# ════════════════════════════════════════════════════════════════════════════

@app.get(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get daily meal plan",
    tags=["Recommendation"],
)
def get_recommendations(
    top_n: int = Query(default=3, ge=1, le=10, description="Recipes per meal slot")
):
    """
    Generate a full personalised daily meal plan.

    Uses:
    - Current profile  (data/user_profile.json)
    - Current preferences (data/user_preference.json — updated by /feedback)
    - Exercise 7-day average (data/exercise_logs.json)

    Pipeline per meal slot:
      BM25 lexical → FAISS semantic → Contrastive re-scoring →
      Adaptive RRF + Nutrition scoring → MMR diversity selection
    """
    try:
        plan = recommender_service.get_daily_meal_plan(top_n=top_n)

        # Build response
        meal_plan_out = {}
        for meal_type, slot in plan["meal_plan"].items():
            recipes_out = [
                RecipeResult(
                    title           = r.get("title", "Unknown"),
                    calories        = r.get("calories", 0),
                    protein         = r.get("protein", 0),
                    servings        = r.get("servings", 1),
                    ingredients     = r.get("ingredients", []),
                    ingredients_raw = r.get("ingredients_raw", []),
                    steps           = r.get("steps", []),
                    instructions    = r.get("instructions") or r.get("steps", []),
                    rating          = r.get("rating"),
                    rating_count    = r.get("rating_count"),
                    rrf_score       = round(r.get("rrf_score", 0), 6),
                    final_score     = round(r.get("final_score", 0), 6),
                    explanation     = r.get("explanation", []),
                )
                for r in slot["recipes"]
            ]
            meal_plan_out[meal_type] = MealSlot(
                target_calories = slot["target_calories"],
                target_protein  = slot["target_protein"],
                recipes         = recipes_out,
            )

        nt = plan["nutrition_targets"]
        return RecommendationResponse(
            user_name           = plan["user_name"],
            goal                = plan["goal"],
            nutrition_targets   = NutritionTargets(
                bmr               = nt["bmr"],
                tdee              = nt["tdee"],
                total_calories    = nt["total_calories"],
                total_protein     = nt["total_protein"],
                exercise_avg_7    = nt["exercise_avg_7"],
                meal_targets      = nt["meal_targets"],
            ),
            meal_plan           = meal_plan_out,
            active_preferences  = plan["active_preferences"],
        )
    except Exception as e:
        logger.exception("Error generating recommendations")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# Exercise
# ════════════════════════════════════════════════════════════════════════════

@app.post(
    "/exercise",
    response_model=ExerciseLogEntry,
    summary="Log an exercise session",
    tags=["Exercise"],
)
def log_exercise(body: ExerciseLogRequest):
    """
    Log an exercise session.  
    If `weight_kg` is not provided, the user's profile weight is used.
    """
    try:
        weight_kg = body.weight_kg
        if weight_kg is None:
            profile   = profile_service.get_profile()
            weight_kg = profile.get("weight_kg", 70)

        calories_burned = exercise_service.calculate_exercise_calories(
            body.exercise_name, body.duration_minutes, weight_kg
        )
        log_entry = {
            "date":             body.date,
            "exercise_name":    body.exercise_name,
            "duration_minutes": body.duration_minutes,
            "calories_burned":  calories_burned,
        }
        exercise_service.save_exercise_log(log_entry)
        return ExerciseLogEntry(**log_entry)
    except Exception as e:
        logger.exception("Error logging exercise")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/exercise",
    response_model=ExerciseSummaryResponse,
    summary="Get exercise logs and 7-day average",
    tags=["Exercise"],
)
def get_exercise():
    """Return all exercise log entries and the 7-day rolling calorie average."""
    try:
        logs    = exercise_service.load_exercise_logs()
        avg_7   = exercise_service.get_previous_7_day_average()
        return ExerciseSummaryResponse(
            logs=[ExerciseLogEntry(**e) for e in logs],
            seven_day_average_calories=avg_7,
        )
    except Exception as e:
        logger.exception("Error loading exercise logs")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# History
# ════════════════════════════════════════════════════════════════════════════

@app.get(
    "/history",
    summary="Get recommendation history",
    tags=["History"],
)
def get_history():
    """
    Return the full recommendation history log.
    Each entry records which recipe was recommended, for which meal slot, and on which date.
    Same-day entries are used to avoid recommending duplicate recipes within a day.
    """
    try:
        return history_service.load_history()
    except Exception as e:
        logger.exception("Error loading history")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/history",
    summary="Clear recommendation history",
    tags=["History"],
)
def clear_history():
    """Wipe the recommendation history (useful for testing / fresh start)."""
    try:
        history_service.clear_history()
        return {"status": "ok", "message": "History cleared."}
    except Exception as e:
        logger.exception("Error clearing history")
        raise HTTPException(status_code=500, detail=str(e))
