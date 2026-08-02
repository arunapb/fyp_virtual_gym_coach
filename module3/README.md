# Meal Recommendation & Feedback System

FastAPI backend that recommends meals (Breakfast/Lunch/Dinner) using a
BM25 + FAISS + MMR retrieval pipeline, adapts to natural-language feedback
via a BERT ABSA + BART zero-shot NLP pipeline, and tracks nutrition/exercise
targets (BMR/TDEE, goal weight & pace, MET-based exercise calories).

## Prerequisites

- Python 3.9+ (the Docker image uses 3.9-slim)
- ~3-4 GB free disk for downloaded model weights (BERT ABSA, BART
  zero-shot, SentenceTransformer) — these are pulled automatically from
  Hugging Face Hub on first run, not stored in this repo
- No GPU required (everything runs on CPU)

## Setup

```bash
# 1. Install Python dependencies
cd backend
pip install -r requirements.txt

# 2. Download the spaCy language model (used for feedback parsing)
python -m spacy download en_core_web_sm

# 3. (Optional) Qdrant Cloud vector search
#    Without this, retrieval automatically falls back to the local FAISS
#    index in vector_store/ -- the app runs fine either way.
cp ../.env.example ../.env
# then fill in QDRANT_URL / QDRANT_API_KEY / COLLECTION_NAME in .env

# 4. Run the server
uvicorn app:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

## What you do NOT need to set up manually

- `vector_store/` (FAISS index + embeddings) is already included in this
  repo — the app reads it directly, no rebuild needed.
- The BERT ABSA (`nipun145/food_feedback`), BART zero-shot
  (`facebook/bart-large-mnli`), and SentenceTransformer
  (`all-MiniLM-L6-v2`) models are downloaded automatically from Hugging
  Face Hub the first time each is used and cached locally afterward.
- `backend/data/RAW_recipes.csv` and `ml_models/` are not required at
  runtime and are intentionally excluded from the repo (large files,
  unused by the running app).

## Running with Docker

```bash
docker build -t meal-recsys .
docker run -p 7860:7860 meal-recsys
```

The image installs the spaCy model automatically. Requires network access
on first run so the Hugging Face models above can download.

## Project layout

```
backend/            FastAPI app, services, data
  app.py             API entrypoint (run this with uvicorn)
  services/          Feedback NLP, retrieval, recommender, exercise, goal logic
  data/              Runtime state (preferences, profile, logs) + label_map.json
  hyperparameter_output/experiments/
                      Standalone scripts for tuning retrieval hyperparameters
                      (each independently runnable, e.g. `python experiment5_nutrition_blend.py`)
vector_store/        Prebuilt FAISS index + recipe embeddings/metadata
scripts/             Offline index-building / Qdrant upload utilities
evaluation/          Pipeline evaluation scripts + results
test_frontend/       Minimal HTML page for manually exercising the API
```

## Rebuilding the retrieval index

If you change the underlying recipe dataset, rebuild `vector_store/` with:

```bash
python scripts/build_recipe_index.py
```
