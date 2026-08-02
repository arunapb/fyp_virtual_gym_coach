"""
backend/user_store.py — the demo's multi-user layer.

WHAT THIS IS NOT: accounts. There are no passwords, no sessions and no
authorisation — anyone can claim any username by typing it (see
frontend/auth.js). This exists so several people can use the demo without
inheriting each other's workout log, meal preferences and goal weight.

The registry
------------
`backend/data/users.json` is the "database": one record per username, with the
directory its data lives in. Each user gets

    backend/data/users/<slug>/
        user_profile.json           goal weight, pace, body stats
        user_preference.json        the ingredient scores /feedback learns
        exercise_logs.json          MET-based entries written by each workout
        recommendation_history.json same-day de-duplication

seeded empty on first sight, which is what makes a brand-new username start
with no logged workouts and no active preferences.

How Module 3 is pointed at the right one
----------------------------------------
Module 3's services each resolve their data file ONCE at import time into a
module-level constant (`profile_service._PROFILE_PATH`, and the three others in
_ROUTED below) and then read that constant on every call. Nothing is captured
at call time, so rebinding the constant is enough to redirect every read and
write — no edit to module3/ and no wrapper around its functions.

That is global state, so it is guarded: `set_active()` takes the lock, and
`as_user()` restores the previous user afterwards. The analysis thread uses
`as_user()` when it writes a finished workout, so an HTTP request switching
users mid-analysis cannot misfile it. This is adequate for a single-browser
demo and would need real per-request isolation to be anything more.

Users whose data has never been requested are never routed: with no active
user, Module 3 reads its own module3/backend/data/ files exactly as it always
has. Those files are left untouched.
"""

import json
import re
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
USERS_DIR = DATA_DIR / "users"
REGISTRY = DATA_DIR / "users.json"

# (Module 3 service, its path constant, the file it should point at.)
_ROUTED = (
    ("profile_service", "_PROFILE_PATH", "user_profile.json"),
    ("feedback_service", "_PREF_FILE", "user_preference.json"),
    ("exercise_service", "_LOGS_PATH", "exercise_logs.json"),
    ("history_service", "_HISTORY_FILE", "recommendation_history.json"),
)

# A new user starts genuinely empty. The profile is the one exception — Module 3
# reads several of these fields unconditionally, so it gets defaults rather than
# {} (they are all overwritten by the mandatory goal popup at sign-in anyway).
_EMPTY = {
    "user_preference.json": {},
    "exercise_logs.json": [],
    "recommendation_history.json": [],
}

_lock = threading.RLock()
_active = None


# ── Naming ──────────────────────────────────────────────────────────────────

def slugify(username: str) -> str:
    """
    A username is free text typed into a login box; this is what becomes a
    directory name. Anything outside [a-z0-9._-] is collapsed to '-', so path
    separators and traversal sequences cannot survive into a path.
    """
    slug = re.sub(r"[^a-z0-9._-]+", "-", (username or "").strip().lower()).strip("-.")
    return slug[:48] or "user"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Registry ────────────────────────────────────────────────────────────────

def _read_registry() -> dict:
    if not REGISTRY.exists():
        return {"users": []}
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8") or "{}")
    except (json.JSONDecodeError, OSError):
        return {"users": []}
    return data if isinstance(data, dict) and isinstance(data.get("users"), list) \
        else {"users": []}


def _write_registry(registry: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def list_users() -> list:
    return _read_registry()["users"]


def user_dir(username: str) -> Path:
    return USERS_DIR / slugify(username)


# ── Provisioning ────────────────────────────────────────────────────────────

def ensure(username: str) -> dict:
    """
    Return this user's registry record, creating and seeding them on first
    sight. `record["new"]` reports whether this call is what created them.
    """
    slug = slugify(username)
    with _lock:
        registry = _read_registry()
        record = next((u for u in registry["users"] if u["slug"] == slug), None)
        created = record is None

        if created:
            record = {
                "username": (username or "").strip() or slug,
                "slug": slug,
                # Module 3 keys its recommendation history by user_id; a distinct
                # one per user keeps that meaningful even though the per-user
                # directory already isolates the file.
                "user_id": f"user_{slug}",
                "created": _now(),
                "last_login": _now(),
            }
            registry["users"].append(record)
        else:
            record["last_login"] = _now()

        _write_registry(registry)
        _seed(record)
        return {**record, "new": created}


def _seed(record: dict) -> None:
    """Create the user's directory and any missing data file. Never overwrites."""
    directory = USERS_DIR / record["slug"]
    directory.mkdir(parents=True, exist_ok=True)

    for name, empty in _EMPTY.items():
        path = directory / name
        if not path.exists():
            path.write_text(json.dumps(empty, indent=2), encoding="utf-8")

    profile = directory / "user_profile.json"
    if not profile.exists():
        profile.write_text(json.dumps({
            "user_id": record["user_id"],
            "name": record["username"],
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
        }, indent=2), encoding="utf-8")


# ── Routing Module 3's file paths ───────────────────────────────────────────

def active() -> str | None:
    return _active


def set_active(username) -> None:
    """
    Make `username` the user Module 3 reads and writes. `None` restores Module
    3's own module3/backend/data/ files.
    """
    global _active
    with _lock:
        _active = (username or "").strip() or None if username else None
        if _active is not None:
            ensure(_active)
        _repoint_loaded()


def _repoint_loaded() -> None:
    """
    Apply the active user's paths to every routed service already imported.

    Only touches `sys.modules` — deliberately never imports anything, since
    `feedback_service` costs torch + transformers + spaCy and switching users
    must stay instant. Services imported later are caught by `apply_paths()`,
    which backend/module3/bridge.py calls right after each import.
    """
    for service, attr, filename in _ROUTED:
        module = sys.modules.get(f"services.{service}")
        if module is not None:
            setattr(module, attr, _path_for(filename))


def apply_paths(service: str, module) -> None:
    """Called by bridge.load() the moment a service is imported."""
    for name, attr, filename in _ROUTED:
        if name == service:
            setattr(module, attr, _path_for(filename))


def _path_for(filename: str) -> str:
    """The active user's copy of `filename`, or Module 3's own when unset."""
    if _active is None:
        from backend.module3.bridge import MODULE3_BACKEND_DIR
        return str(MODULE3_BACKEND_DIR / "data" / filename)
    return str(user_dir(_active) / filename)


@contextmanager
def as_user(username):
    """
    Run a block against one user's data, then restore whoever was active.

    Used by the analysis thread so a finished workout is filed against the user
    who uploaded it even if a browser request switched the active user while it
    ran. Holds the lock for the whole block: correctness over concurrency, and
    the demo already serialises analyses (MAX_CONCURRENT_JOBS = 1).
    """
    with _lock:
        previous = _active
        try:
            set_active(username)
            yield
        finally:
            set_active(previous)
