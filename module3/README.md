# NutriIngredientNet v6 — serving package

Input: ONE RGB food photo. Output: dish totals first, then ingredient-wise nutrition.
Model: arunapb/nutriingredientnet-v6 (73.5 kcal MAE on the official Nutrition5k RGB test split).

## Files
- app.py           FastAPI server (model class + full v6 pipeline)
- predict.py       command line: `python predict.py photo.jpg`
- demo.html        drag-and-drop web page (edit the API constant at the top)
- .env.example     all settings, documented — copy to `.env` and edit
- .gitignore       keeps your real `.env` (and its token) out of git
- Dockerfile       for Hugging Face Spaces / any container host
- requirements.txt CPU-only install

## Settings (.env)
    cp .env.example .env      # then edit if you want

| key | default | what it does |
|---|---|---|
| HF_REPO | arunapb/nutriingredientnet-v6 | which trained model to download |
| HF_TOKEN | (blank) | only needed if that repo is private |
| USE_TTA | true | 4-rotation averaging: accurate (~2 s) vs false (~0.5 s) |
| USE_DEPTH | true | predict depth from the photo; false = faster, less accurate |
| DEVICE | auto | auto / cpu / cuda |
| MAX_UPLOAD_MB | 15 | reject bigger uploads |
| CORS_ORIGINS | * | which websites may call the API |

Precedence: real environment variables > `.env` > defaults — so hosted platforms
(HF Spaces secrets, Docker `-e`, systemd) override the file without editing it.
`.env` is gitignored: never commit your token.

## A. Run locally
    pip install -r requirements.txt
    uvicorn app:app --port 7860
Then open demo.html, or http://localhost:7860/docs for the built-in test page.
First start downloads ~200 MB (model + depth model); afterwards it works offline.

## B. One-off, no server
    python predict.py my_lunch.jpg

## C. Deploy free on Hugging Face Spaces
1. huggingface.co -> New Space -> SDK **Docker** -> CPU basic (free)
2. Upload app.py, requirements.txt, Dockerfile
3. API becomes https://<user>-<space>.hf.space/predict
4. Put that URL in demo.html

## API
POST /predict   multipart form, field `file` = image
GET  /health    readiness ping (use it to wake a sleeping free Space)

Response:
    { "totals": {"calories":783.0,"fat_g":55.9,"carbs_g":40.2,
                 "protein_g":29.7,"mass_g":222.1},
      "ingredients":[{"name":"almonds","status":"certain","confidence":0.98,
                      "grams":96.7,"kcal":605.3,"fat":47.9,"carb":22.0,"protein":21.7}, ...],
      "atwater_check": {...}, "diagnostics": {...}, "accuracy": {...}, "warning": "..." }

## Speed
TTA on (default): ~2 s/photo CPU, ~0.4 s GPU. Set USE_TTA=0 for ~4x faster.
Set USE_DEPTH=0 to skip pseudo-depth (faster, slightly less accurate).

## Honest limits
- Trained on a FIXED overhead camera rig; other angles/distances are less accurate.
- 166-ingredient vocabulary; unknown foods are invisible to it.
- Average error ~73 kcal (29%); very large or calorie-dense plates are under-predicted.
- Research demo, not medical or dietary advice.


# 1. Create the conda env (one-time)
conda create -n nutriapi python=3.11 -y

# 2. Activate it
conda activate nutriapi

# 3. Install dependencies
cd /Users/aruna/Documents/GitHub/nutrition-api
pip install -r requirements.txt

# 4. Start the server
uvicorn app:app  --port 7860 --reload

