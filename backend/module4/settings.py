"""
backend/module4/settings.py — Module 4's runtime settings.

Ported from module4/app.py's env handling, with one deliberate change: the
`.env` is located relative to THIS package rather than via
`find_dotenv(usecwd=True)`. The original resolved it from the current working
directory, which worked when the server was launched from module4/ but would
silently find nothing now that the server starts at the repo root — the
settings would fall back to defaults with no warning.
"""

import os
from pathlib import Path

MODULE4_DIR = Path(__file__).resolve().parent
# The original module4/ folder is where .env / .env.example live.
LEGACY_MODULE4_DIR = MODULE4_DIR.parent.parent / "module4"

try:
    from dotenv import load_dotenv

    for _candidate in (MODULE4_DIR / ".env", LEGACY_MODULE4_DIR / ".env"):
        if _candidate.exists():
            load_dotenv(_candidate, override=False)
            print(f"[NUTRI] settings loaded from {_candidate}")
            break
    else:
        print("[NUTRI] no .env found - using environment variables / defaults")
except ImportError:  # pragma: no cover - depends on the install
    print("[NUTRI] python-dotenv not installed; using environment variables only")


def env_str(key, default):
    v = os.environ.get(key)
    return default if v is None or v.strip() == "" else v.strip()


def env_bool(key, default):
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def env_int(key, default):
    try:
        return int(env_str(key, str(default)))
    except ValueError:
        return default


HF_REPO = env_str("HF_REPO", "arunapb/nutriingredientnet-v6")
HF_TOKEN = env_str("HF_TOKEN", "") or None      # only needed for a private repo
USE_TTA = env_bool("USE_TTA", True)             # 4-rotation averaging
USE_DEPTH = env_bool("USE_DEPTH", True)         # pseudo-depth 4th channel
MAX_UPLOAD_MB = env_int("MAX_UPLOAD_MB", 15)
