# ─────────────────────────────────────────────────────────────────────────────
# Meal Recommendation System — Hugging Face Spaces Docker image
#
# Build context: the root "combine/" folder
# Runs on: port 7860 (HF Spaces default)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.9-slim

# Install system build tools (needed for some pip packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Create working directory ───────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────────
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Download spaCy language model ──────────────────────────────────────────────
RUN python -m spacy download en_core_web_sm

# ── Copy project files ─────────────────────────────────────────────────────────
# Backend application code
COPY backend/ ./backend/

# Feedback models (BERT ABSA + BART zero-shot) are downloaded from Hugging
# Face Hub at runtime (see services/feedback_service.py) -- nothing to copy.

# Retrieval index files (FAISS + BM25 data)
COPY vector_store/ ./vector_store/

# Utility & maintenance scripts
COPY scripts/      ./scripts/

# ── Hugging Face Spaces requires port 7860 ────────────────────────────────────
EXPOSE 7860

# ── Start the FastAPI server ───────────────────────────────────────────────────
WORKDIR /app/backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
