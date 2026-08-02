"""
qdrant_config.py
================
Qdrant Cloud connection settings.
Values are loaded from the .env file at project root (or from environment
variables set directly on the host / container).
"""

import os
from pathlib import Path

# Load .env file if python-dotenv is available (local dev).
# In production (AWS), environment variables are injected directly.
try:
    from dotenv import load_dotenv
    _ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_ENV_FILE)
except ImportError:
    pass  # dotenv not installed — rely on real environment variables

QDRANT_URL      = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY  = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "recipes")
QDRANT_TIMEOUT  = int(os.environ.get("QDRANT_TIMEOUT", "15"))
