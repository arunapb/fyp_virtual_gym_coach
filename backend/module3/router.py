"""
backend/module3/router.py — Module 3's HTTP surface on the shared server.

`module3/backend/app.py` declared these as decorators on its own module-level
`FastAPI()` at `/health`, `/profile`, `/recommend`, `/feedback`, `/exercise`,
`/history`, `/preferences`, `/goal/*`. Two changes were needed to host them
alongside Modules 1, 2 and 4:

  * an `APIRouter` rather than a second `FastAPI` app, so the routes attach to
    the one server;
  * a `/api/meals` prefix — `/health` and `/profile` are far too generic for a
    server that now hosts four modules, and `/exercise` sat one letter away
    from Module 2's existing `/api/exercises` (which lists the *registry*, a
    completely different thing).

Request/response shapes, status codes and error text are otherwise Module 3's,
verbatim. Handler bodies are the same calls in the same order; the response
models are Module 3's own, re-exported (see schemas.py).

Everything runs in a worker thread
----------------------------------
Every original handler was `def` (which Starlette already threads) or `async
def` calling blocking code inline. Here they are all `async def` +
`run_in_threadpool`, explicitly, because on this server the event loop is also
pumping the Module 1 + Module 2 analysis WebSocket. `GET /recommend` is a
multi-second BM25 + FAISS + MMR pipeline and `POST /feedback` runs BERT and
BART; either one executed inline would freeze the live video stream for its
whole duration. The first call of each additionally pays the service import
(see bridge.py) — tens of seconds — which absolutely cannot happen on the loop.
"""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from backend import user_store
from backend.module3 import bridge
from backend.module3.schemas import (
    AspectResult, ExerciseLogEntry, ExerciseLogRequest, ExerciseSummaryResponse,
    FeedbackRequest, FeedbackResponse, GoalOptionsRequest, GoalOptionsResponse,
    HealthResponse, MealSlot, NutritionTargets, PreferenceResponse,
    ProfileResponse, ProfileUpdateRequest, RecipeResult, RecommendationResponse,
    SelectPaceRequest, SelectPaceResponse,
)

logger = logging.getLogger("module3")


def active_user(x_demo_user: str = Header(default="")) -> str:
    """
    Point Module 3 at the caller's own profile, preferences, exercise log and
    history before the handler runs.

    The browser sends `X-Demo-User` on every request (frontend/auth.js). This
    is NOT authentication — the header is self-asserted and trivially forged;
    it is how a demo with no accounts keeps several people's data apart. An
    absent header leaves Module 3 on its own shared files, which is what the
    interactive API docs and any curl get.

    A dependency rather than middleware so it applies to exactly these routes.
    """
    username = (x_demo_user or "").strip()
    user_store.set_active(username or None)
    return username


router = APIRouter(prefix="/api/meals", tags=["meals"],
                   dependencies=[Depends(active_user)])


# ════════════════════════════════════════════════════════════════════════════
# System
# ════════════════════════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    """
    Answers without importing anything heavy.

    Module 3's own `/health` could claim "All models are running" because its
    server refused to finish booting until they were. Here they load on first
    use, so this reports readiness honestly instead.
    """
    warm = [name for name in bridge.HEAVY_SERVICES if bridge.is_loaded(name)]
    if warm:
        message = f"Ready. Loaded: {', '.join(warm)}."
    else:
        message = ("Ready. Retrieval and feedback models load on first use — "
                   "the first meal plan or feedback request will be slow.")
    return HealthResponse(status="ok", message=message)


# ════════════════════════════════════════════════════════════════════════════
# Profile
# ════════════════════════════════════════════════════════════════════════════

def _with_goal_progress(profile: dict) -> dict:
    """
    Attach computed goal-progress fields (kg_to_change, is_loss,
    already_at_goal, estimated_weeks/months for the selected pace).

    Transcribed from `module3/backend/app.py::_with_goal_progress` — derived
    live from weight_kg/goal_weight_kg/selected_pace, never written back to
    user_profile.json, so it can never go stale. No-op without goal_weight_kg.
    """
    goal_svc = bridge.goal()
    out = dict(profile)
    goal_weight = profile.get("goal_weight_kg")
    if not goal_weight:
        return out
    try:
        plan = goal_svc.calculate_goal_plan(profile["weight_kg"], goal_weight)
    except ValueError:
        return out

    pace_key = profile.get("selected_pace") or goal_svc.DEFAULT_PACE
    pace_opt = next((o for o in plan["pace_options"] if o["key"] == pace_key), None)

    out["kg_to_change"] = plan["kg_to_change"]
    out["is_loss"] = plan["is_loss"]
    out["already_at_goal"] = plan["already_at_goal"]
    if pace_opt:
        out["estimated_weeks"] = pace_opt["estimated_weeks"]
        out["estimated_months"] = pace_opt["estimated_months"]
        out["pace_kcal_per_day"] = pace_opt["kcal_per_day"]
    return out


@router.get("/profile", response_model=ProfileResponse, summary="Get user profile")
async def get_profile():
    def work():
        return _with_goal_progress(bridge.profile().get_profile())
    try:
        return await run_in_threadpool(work)
    except Exception as exc:
        logger.exception("Error loading profile")
        raise HTTPException(500, str(exc))


@router.put("/profile", response_model=ProfileResponse, summary="Update user profile")
async def update_profile(body: ProfileUpdateRequest):
    def work():
        updated = bridge.profile().update_profile(body.model_dump(exclude_none=True))
        return _with_goal_progress(updated)
    try:
        return await run_in_threadpool(work)
    except Exception as exc:
        logger.exception("Error updating profile")
        raise HTTPException(500, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# Goal weight & pace
# ════════════════════════════════════════════════════════════════════════════

@router.post("/goal/options", response_model=GoalOptionsResponse,
             summary="Calculate pace options for a goal weight")
async def goal_options(body: GoalOptionsRequest):
    """
    Return kg-to-change plus the three pace options, and persist
    `goal_weight_kg` — also syncing the profile's `goal` direction so
    `calculate_targets()` stays consistent with a single source of truth.
    """
    def work():
        profile_svc, goal_svc = bridge.profile(), bridge.goal()
        profile = profile_svc.get_profile()
        plan = goal_svc.calculate_goal_plan(profile["weight_kg"], body.goal_weight_kg)

        if plan["already_at_goal"]:
            new_goal = "maintenance"
        elif plan["is_loss"]:
            new_goal = "weight_loss"
        else:
            new_goal = "weight_gain"
        profile_svc.update_profile({"goal_weight_kg": body.goal_weight_kg,
                                    "goal": new_goal})
        return plan

    try:
        return GoalOptionsResponse(**await run_in_threadpool(work))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Error calculating goal options")
        raise HTTPException(500, str(exc))


@router.post("/goal/select-pace", response_model=SelectPaceResponse,
             summary="Select a pace and recalculate nutrition targets")
async def select_pace(body: SelectPaceRequest):
    goal_svc = bridge.goal()
    if body.pace not in goal_svc.PACE_KEYS:
        raise HTTPException(
            400, f"Invalid pace '{body.pace}'. Must be one of: "
                 f"{sorted(goal_svc.PACE_KEYS)}")

    def work():
        profile_svc = bridge.profile()
        profile_svc.update_profile({"selected_pace": body.pace})
        return bridge.recommender().calculate_targets(profile_svc.get_profile())

    try:
        targets = await run_in_threadpool(work)
        return SelectPaceResponse(
            pace=body.pace,
            kcal_per_day=goal_svc.get_pace_kcal(body.pace),
            new_daily_target=targets["total_calories"],
            nutrition_targets=NutritionTargets(
                bmr=targets["bmr"],
                tdee=targets["tdee"],
                total_calories=targets["total_calories"],
                total_protein=targets["total_protein"],
                exercise_avg_7=targets["exercise_avg_7"],
                meal_targets=targets["meal_targets"],
            ),
        )
    except Exception as exc:
        logger.exception("Error selecting pace")
        raise HTTPException(500, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# Preferences
# ════════════════════════════════════════════════════════════════════════════

@router.get("/preferences", response_model=PreferenceResponse,
            summary="Get ingredient preference vector")
async def get_preferences():
    """Scores run -1.0 (strongly disliked) to +1.0 (strongly liked)."""
    prefs = await run_in_threadpool(lambda: bridge.feedback().load_preferences())
    return PreferenceResponse(preferences=prefs, count=len(prefs))


@router.delete("/preferences", response_model=PreferenceResponse,
               summary="Reset all preferences")
async def reset_preferences():
    await run_in_threadpool(lambda: bridge.feedback().reset_preferences())
    return PreferenceResponse(preferences={}, count=0)


# ════════════════════════════════════════════════════════════════════════════
# Feedback
# ════════════════════════════════════════════════════════════════════════════

@router.post("/feedback", response_model=FeedbackResponse,
             summary="Submit meal feedback")
async def submit_feedback(body: FeedbackRequest):
    """
    Causal detection (spaCy) -> intent (BART zero-shot) -> BERT ABSA ->
    intensity (VADER) -> delta scoring, updating user_preference.json.

    The update is immediate: the next `GET /api/meals/recommend` reflects it.
    """
    try:
        result = await run_in_threadpool(
            lambda: bridge.feedback().analyze_feedback(feedback_text=body.feedback_text))
        return FeedbackResponse(
            feedback_text=body.feedback_text,
            causal=result["causal"],
            cause_ingredient=result["cause_ingredient"],
            mixed=result["mixed"],
            aspects=[AspectResult(ingredient=name, action=action, delta=round(delta, 4))
                     for name, action, delta in result["aspects"]],
            updated_preferences=result["updated_preferences"],
        )
    except Exception as exc:
        logger.exception("Error processing feedback")
        raise HTTPException(500, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# Recommendation
# ════════════════════════════════════════════════════════════════════════════

@router.get("/recommend", response_model=RecommendationResponse,
            summary="Get daily meal plan")
async def recommend(top_n: int = Query(default=3, ge=1, le=10,
                                       description="Recipes per meal slot")):
    """
    Per slot: BM25 lexical -> FAISS semantic -> contrastive re-scoring ->
    adaptive RRF + nutrition scoring -> MMR diversity selection.

    Reads the live profile, the preference vector `/feedback` maintains, and
    the 7-day exercise average — which now includes workouts logged straight
    from the Module 1 + Module 2 analysis (backend/module3/exercise_log.py).
    """
    try:
        plan = await run_in_threadpool(
            lambda: bridge.recommender().get_daily_meal_plan(top_n=top_n))

        meal_plan_out = {}
        for meal_type, slot in plan["meal_plan"].items():
            meal_plan_out[meal_type] = MealSlot(
                target_calories=slot["target_calories"],
                target_protein=slot["target_protein"],
                recipes=[
                    RecipeResult(
                        title=r.get("title", "Unknown"),
                        calories=r.get("calories", 0),
                        protein=r.get("protein", 0),
                        servings=r.get("servings", 1),
                        ingredients=r.get("ingredients", []),
                        ingredients_raw=r.get("ingredients_raw", []),
                        steps=r.get("steps", []),
                        instructions=r.get("instructions") or r.get("steps", []),
                        rating=r.get("rating"),
                        rating_count=r.get("rating_count"),
                        rrf_score=round(r.get("rrf_score", 0), 6),
                        final_score=round(r.get("final_score", 0), 6),
                        explanation=r.get("explanation", []),
                    )
                    for r in slot["recipes"]
                ],
            )

        nt = plan["nutrition_targets"]
        return RecommendationResponse(
            user_name=plan["user_name"],
            goal=plan["goal"],
            nutrition_targets=NutritionTargets(
                bmr=nt["bmr"],
                tdee=nt["tdee"],
                total_calories=nt["total_calories"],
                total_protein=nt["total_protein"],
                exercise_avg_7=nt["exercise_avg_7"],
                meal_targets=nt["meal_targets"],
            ),
            meal_plan=meal_plan_out,
            active_preferences=plan["active_preferences"],
        )
    except Exception as exc:
        logger.exception("Error generating recommendations")
        raise HTTPException(500, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# Exercise log
# ════════════════════════════════════════════════════════════════════════════

@router.post("/exercise", response_model=ExerciseLogEntry,
             summary="Log an exercise session")
async def log_exercise(body: ExerciseLogRequest):
    """
    Manual entry. The Module 1 + Module 2 analysis writes the same record
    automatically when a workout finishes — see
    backend/module3/exercise_log.py — so this is for sessions the camera never
    saw. Falls back to the profile weight when `weight_kg` is omitted.
    """
    def work():
        weight_kg = body.weight_kg
        if weight_kg is None:
            weight_kg = bridge.profile().get_profile().get("weight_kg", 70)
        ex_svc = bridge.exercise()
        entry = {
            "date": body.date,
            "exercise_name": body.exercise_name,
            "duration_minutes": body.duration_minutes,
            "calories_burned": ex_svc.calculate_exercise_calories(
                body.exercise_name, body.duration_minutes, weight_kg),
        }
        ex_svc.save_exercise_log(entry)
        return entry

    try:
        return ExerciseLogEntry(**await run_in_threadpool(work))
    except Exception as exc:
        logger.exception("Error logging exercise")
        raise HTTPException(500, str(exc))


@router.get("/exercise", response_model=ExerciseSummaryResponse,
            summary="Get exercise logs and 7-day average")
async def get_exercise():
    def work():
        ex_svc = bridge.exercise()
        return ex_svc.load_exercise_logs(), ex_svc.get_previous_7_day_average()
    try:
        logs, avg_7 = await run_in_threadpool(work)
        return ExerciseSummaryResponse(
            logs=[ExerciseLogEntry(**e) for e in logs],
            seven_day_average_calories=avg_7,
        )
    except Exception as exc:
        logger.exception("Error loading exercise logs")
        raise HTTPException(500, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# History
# ════════════════════════════════════════════════════════════════════════════

@router.get("/history", summary="Get recommendation history")
async def get_history():
    """Same-day entries are what prevent duplicate recipes within a day."""
    try:
        return await run_in_threadpool(lambda: bridge.history().load_history())
    except Exception as exc:
        logger.exception("Error loading history")
        raise HTTPException(500, str(exc))


@router.delete("/history", summary="Clear recommendation history")
async def clear_history():
    try:
        await run_in_threadpool(lambda: bridge.history().clear_history())
        return {"status": "ok", "message": "History cleared."}
    except Exception as exc:
        logger.exception("Error clearing history")
        raise HTTPException(500, str(exc))
