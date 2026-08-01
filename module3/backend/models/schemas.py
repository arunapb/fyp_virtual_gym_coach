"""
schemas.py
==========
Pydantic models for all request and response bodies in the unified backend API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    message: str


# ── Profile ────────────────────────────────────────────────────────────────────

class ProfileResponse(BaseModel):
    user_id: str
    name: str
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    activity_level: str
    goal: str
    meals_per_day: int
    allergies: Optional[List[str]] = []
    disliked_ingredients: Optional[List[str]] = []
    goal_weight_kg: Optional[float] = None
    selected_pace: Optional[str] = None
    # Computed goal-progress fields (derived live from weight_kg/goal_weight_kg/
    # selected_pace via goal_service -- never persisted to user_profile.json,
    # always in sync with the stored values). Present only when goal_weight_kg is set.
    kg_to_change: Optional[float] = None
    is_loss: Optional[bool] = None
    already_at_goal: Optional[bool] = None
    estimated_weeks: Optional[int] = None
    estimated_months: Optional[float] = None
    pace_kcal_per_day: Optional[int] = None


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    activity_level: Optional[str] = Field(
        default=None,
        description="One of: sedentary, lightly_active, moderately_active, very_active"
    )
    goal: Optional[str] = Field(
        default=None,
        description="One of: weight_loss, weight_gain, maintenance"
    )
    meals_per_day: Optional[int] = None
    allergies: Optional[List[str]] = None
    disliked_ingredients: Optional[List[str]] = None
    goal_weight_kg: Optional[float] = Field(
        default=None,
        description="Target goal weight in kg, used by the Goal Weight & Pace Selector"
    )
    selected_pace: Optional[str] = Field(
        default=None,
        description="One of: gentle, balanced, faster"
    )


# ── Goal Weight & Pace ─────────────────────────────────────────────────────────

class GoalOptionsRequest(BaseModel):
    goal_weight_kg: float = Field(..., gt=0, description="Target goal weight in kg")


class PaceOption(BaseModel):
    key: str
    label: str
    kcal_per_day: int
    weekly_rate_kg: float
    estimated_weeks: int
    estimated_months: float
    is_default: bool


class GoalOptionsResponse(BaseModel):
    current_weight_kg: float
    goal_weight_kg: float
    kg_to_change: float
    is_loss: bool
    already_at_goal: bool
    pace_options: List[PaceOption]


class SelectPaceRequest(BaseModel):
    pace: str = Field(..., description="One of: gentle, balanced, faster")


class SelectPaceResponse(BaseModel):
    pace: str
    kcal_per_day: int
    new_daily_target: int
    nutrition_targets: "NutritionTargets"


# ── Feedback ───────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    feedback_text: str = Field(..., description="Natural language feedback from the user")


class AspectResult(BaseModel):
    ingredient: str
    action: str       # REDUCE | INCREASE | KEEP
    delta: float      # change applied to preference score


class FeedbackResponse(BaseModel):
    feedback_text: str
    causal: bool
    cause_ingredient: Optional[str]
    mixed: bool
    aspects: List[AspectResult]
    updated_preferences: Dict[str, float]


# ── Preferences ────────────────────────────────────────────────────────────────

class PreferenceResponse(BaseModel):
    preferences: Dict[str, float]
    count: int


# ── Exercise ───────────────────────────────────────────────────────────────────

class ExerciseLogRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    exercise_name: str = Field(..., description="One of: squat, deadlift, bicep_curl, shoulder_press")
    duration_minutes: float = Field(..., gt=0)
    weight_kg: Optional[float] = Field(None, description="User weight in kg (uses profile weight if omitted)")


class ExerciseLogEntry(BaseModel):
    date: str
    exercise_name: str
    duration_minutes: float
    calories_burned: float


class ExerciseSummaryResponse(BaseModel):
    logs: List[ExerciseLogEntry]
    seven_day_average_calories: float


# ── Recommendation ─────────────────────────────────────────────────────────────

class RecipeResult(BaseModel):
    title: str
    calories: float
    protein: float
    servings: Optional[int] = 1
    ingredients: List[str]          # cleaned ingredient names (used for matching)
    ingredients_raw: Optional[List[str]] = None  # full strings with quantities from dataset
    steps: Optional[List[str]] = None            # preparation steps
    instructions: Optional[List[str]] = None     # preparation instructions (alias for steps)
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    rrf_score: Optional[float] = None
    final_score: Optional[float] = None
    explanation: Optional[List[str]] = None


class MealSlot(BaseModel):
    target_calories: int
    target_protein: int
    recipes: List[RecipeResult]


class NutritionTargets(BaseModel):
    bmr: float
    tdee: float
    total_calories: int
    total_protein: int
    exercise_avg_7: float
    meal_targets: Dict[str, Dict[str, int]]


class RecommendationResponse(BaseModel):
    user_name: str
    goal: str
    nutrition_targets: NutritionTargets
    meal_plan: Dict[str, MealSlot]   # {"Breakfast": ..., "Lunch": ..., "Dinner": ...}
    active_preferences: Dict[str, float]


SelectPaceResponse.model_rebuild()
