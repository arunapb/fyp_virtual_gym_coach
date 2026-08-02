"""
backend/module3 — Module 3 (meal recommendation & feedback) on the shared
FastAPI server.

Like backend/module4, this package is an ADAPTER, not a copy: Module 3's own
`module3/backend/services/*` and `module3/backend/models/schemas.py` are
imported unmodified from where they live and called as-is. Nothing in
`module3/` is edited.

Why adapt rather than transcribe (module4 was transcribed)
----------------------------------------------------------
Module 4's inference was one self-contained file with no state on disk, so
lifting it into `backend/module4/model.py` cost nothing. Module 3 is the
opposite: its six services are mutually dependent (`from . import
exercise_service, goal_service, ...`) and every one of them resolves its data
file relative to its OWN location —

    module3/backend/data/user_profile.json
    module3/backend/data/user_preference.json
    module3/backend/data/recommendation_history.json
    module3/backend/data/exercise_logs.json
    module3/vector_store/recipe_index.faiss  (+ metadata/texts/vectors)

Copying the code would silently repoint all of that at a second, empty set of
files: `exercise_logs.json` already holds real history, and the 7-day exercise
average that feeds `calculate_targets()` is computed from it. One source of
truth matters more here than physical separation, so the originals stay
authoritative and this package only wires them to the shared server.

See bridge.py for the import mechanics and router.py for the HTTP surface.
"""
