"""
backend/module3/schemas.py — Module 3's Pydantic request/response models,
re-exported for the shared server's router.

`module3/backend/models/schemas.py` imports nothing but `typing` and
`pydantic`, so unlike the services (see bridge.py) it is cheap enough to
import eagerly at boot — which it must be, since FastAPI reads
`response_model=` while the routes are being declared, long before any
request arrives.

Re-exported rather than redefined so the two can never drift: a field added
to Module 3's schema shows up here automatically.
"""

from backend.module3.bridge import ensure_path

ensure_path()

from models.schemas import (  # noqa: E402 — Module 3's, unmodified
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

__all__ = [
    "AspectResult", "ExerciseLogEntry", "ExerciseLogRequest",
    "ExerciseSummaryResponse", "FeedbackRequest", "FeedbackResponse",
    "GoalOptionsRequest", "GoalOptionsResponse", "HealthResponse", "MealSlot",
    "NutritionTargets", "PreferenceResponse", "ProfileResponse",
    "ProfileUpdateRequest", "RecipeResult", "RecommendationResponse",
    "SelectPaceRequest", "SelectPaceResponse",
]
