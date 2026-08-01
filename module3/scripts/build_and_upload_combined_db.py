"""
build_and_upload_combined_db.py
===============================
Merges recipes from `newdataset.json` and `backend/data/RAW_recipes.csv`,
builds SentenceTransformer embeddings + FAISS index, and uploads all points
to the NEW Qdrant Cloud database cluster.
"""

import sys
import io
import os
import ast
import json
import pickle
import re
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPTS_DIR)
VECTOR_DIR  = os.path.join(BASE_DIR, "vector_store")
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
DATA_DIR    = os.path.join(BACKEND_DIR, "data")

NEWDATASET_PATH = os.path.join(VECTOR_DIR, "newdataset.json")
RAW_CSV_PATH    = os.path.join(DATA_DIR, "RAW_recipes.csv")

INDEX_PATH = os.path.join(VECTOR_DIR, "recipe_index.faiss")
META_PATH  = os.path.join(VECTOR_DIR, "recipe_metadata.pkl")
TEXTS_PATH = os.path.join(VECTOR_DIR, "recipe_texts.pkl")
VECS_PATH  = os.path.join(VECTOR_DIR, "recipe_vectors.npy")

# ── Import Qdrant Config ──────────────────────────────────────────────────────
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from qdrant_config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, QDRANT_TIMEOUT

# ── Ingredient Cleaning Helpers ────────────────────────────────────────────────
_UNIT_WORDS = {
    "teaspoon", "teaspoons", "tsp",
    "tablespoon", "tablespoons", "tbsp",
    "cup", "cups", "pound", "pounds", "lb", "lbs",
    "ounce", "ounces", "oz", "gram", "grams", "g",
    "ml", "milliliter", "milliliters", "liter", "liters", "l",
    "small", "medium", "large", "extra", "clove", "cloves",
    "chopped", "minced", "diced", "sliced", "crushed", "shredded", "grated",
    "to", "taste", "or", "and"
}

_LEADING_NUM_RE = re.compile(r"^\s*[\d/\-]+\s*")

def _clean_ingredient(raw: str) -> str:
    text = _LEADING_NUM_RE.sub("", str(raw).strip())
    words = text.split()
    while words and words[0].lower().rstrip(".,") in _UNIT_WORDS:
        words = words[1:]
    while words and words[-1].lower().rstrip(".,") in _UNIT_WORDS:
        words = words[:-1]
    result = " ".join(words).strip(" ,.")
    return result if result else str(raw).strip()

def load_newdataset_recipes():
    print("[1/5] Processing newdataset.json ...")
    if not os.path.exists(NEWDATASET_PATH):
        print(f"   [WARNING] File not found: {NEWDATASET_PATH}")
        return []

    with open(NEWDATASET_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    parsed = []
    for item in raw:
        try:
            basic = item.get("basic_info", {})
            prep  = item.get("prep_data", {})
            nutr  = item.get("nutritions", {})

            title    = basic.get("title", "").strip()
            category = basic.get("category", "")
            if not title:
                continue

            # Check category filter
            cat_lower = (category or "").lower()
            if not any(kw in cat_lower for kw in ["main", "dish", "salad", "soup", "stew", "casserole"]):
                continue

            calories = float(str(nutr.get("calories", "0")).replace("g","").replace(",",""))
            if calories <= 0 or calories > 2000 or nutr == {}:
                continue

            protein = float(str(nutr.get("protein", "0g")).replace("g","").replace("mg",""))
            fat     = float(str(nutr.get("fat", "0g")).replace("g",""))

            try:
                rating = float(str(basic.get("rating", "0")).replace("\n","").strip())
            except Exception:
                rating = 0.0

            rc_str = str(basic.get("rating_count", "(0)")).replace("\n","").replace("(","").replace(")","").replace(",","").strip()
            try:
                rating_count = int(rc_str)
            except Exception:
                rating_count = 0

            try:
                servings = int(str(prep.get("servings:", "1")).strip().split()[0])
            except Exception:
                servings = 1

            raw_ings = [str(i) for i in item.get("ingridients", [])]
            clean_ings = [_clean_ingredient(i) for i in raw_ings if _clean_ingredient(i)]

            parsed.append({
                "title":           title,
                "category":        category or "Main Dishes",
                "ingredients":     clean_ings,
                "ingredients_raw": raw_ings,
                "calories":        calories,
                "protein":         protein,
                "fat":             fat,
                "rating":          rating,
                "rating_count":    rating_count,
                "servings":        servings,
                "state":           item.get("state", ""),
                "source":          "newdataset.json",
            })
        except Exception:
            continue

    print(f"   Successfully parsed {len(parsed)} recipes from newdataset.json")
    return parsed

def load_raw_csv_recipes(limit=5000):
    print(f"[2/5] Processing RAW_recipes.csv (up to {limit} main dishes) ...")
    if not os.path.exists(RAW_CSV_PATH):
        print(f"   [WARNING] File not found: {RAW_CSV_PATH}")
        return []

    df = pd.read_csv(RAW_CSV_PATH)
    print(f"   Total rows in CSV: {len(df)}")

    # Filter for main dishes or lunch/dinner/breakfast tags
    def is_main(tags_str):
        t = str(tags_str).lower()
        return any(k in t for k in ["main-dish", "dinner", "lunch", "breakfast", "main-ingredient", "side-dishes"])

    df_filtered = df[df["tags"].apply(is_main)].copy()

    parsed = []
    for idx, row in df_filtered.iterrows():
        if len(parsed) >= limit:
            break

        try:
            title = str(row["name"]).strip().title()
            if not title or title.lower() == "nan":
                continue

            # Parse nutrition
            nutr = ast.literal_eval(str(row["nutrition"]))
            calories = float(nutr[0])
            if calories < 50 or calories > 2000:
                continue

            fat     = round(float(nutr[1]) * 65.0 / 100.0, 1)   # % PDV fat -> grams
            protein = round(float(nutr[4]) * 50.0 / 100.0, 1)   # % PDV protein -> grams

            raw_ings = ast.literal_eval(str(row["ingredients"]))
            clean_ings = [_clean_ingredient(i) for i in raw_ings if _clean_ingredient(i)]

            try:
                steps = ast.literal_eval(str(row["steps"])) if pd.notna(row.get("steps")) else []
            except Exception:
                steps = []

            parsed.append({
                "title":           title,
                "category":        "Main Dishes",
                "ingredients":     clean_ings,
                "ingredients_raw": raw_ings,
                "steps":           steps,
                "calories":        calories,
                "protein":         protein,
                "fat":             fat,
                "rating":          4.5,
                "rating_count":    100,
                "servings":        4,
                "state":           "",
                "source":          "RAW_recipes.csv",
            })
        except Exception:
            continue

    print(f"   Successfully parsed {len(parsed)} recipes from RAW_recipes.csv")
    return parsed

def main():
    print("=" * 60)
    print("BUILDING AND UPLOADING COMBINED RECIPE DATABASE TO NEW QDRANT CLUSTER")
    print(f"Target Qdrant URL: {QDRANT_URL}")
    print("=" * 60)

    # 1. Load both dataset sources
    ds1 = load_newdataset_recipes()
    ds2 = load_raw_csv_recipes(limit=5000)

    # Combine & deduplicate by normalized title
    seen_titles = set()
    combined_recipes = []

    for r in ds1 + ds2:
        norm_t = r["title"].strip().lower()
        if norm_t not in seen_titles:
            seen_titles.add(norm_t)
            combined_recipes.append(r)

    total_recipes = len(combined_recipes)
    print(f"\n[3/5] Total unique combined recipes to index: {total_recipes}")

    # 2. Build recipe texts
    PREFIX = "Represent a recipe for health-aware meal recommendation retrieval: "
    recipe_texts = []
    full_texts   = []

    for r in combined_recipes:
        text = (r["title"] + " " + " ".join(r["ingredients"]) + " " + r["state"]).strip()
        recipe_texts.append(text)
        full_texts.append(PREFIX + text)

    # 3. Generate embeddings
    print("\n[4/5] Encoding vectors using SentenceTransformer (all-MiniLM-L6-v2) ...")
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer("all-MiniLM-L6-v2")
    BATCH_SIZE = 128
    all_vectors = []

    for start in range(0, len(full_texts), BATCH_SIZE):
        batch = full_texts[start : start + BATCH_SIZE]
        vecs  = model.encode(batch, normalize_embeddings=False, show_progress_bar=False)
        all_vectors.append(vecs)
        if start > 0 and (start // BATCH_SIZE) % 10 == 0:
            print(f"   Encoded {start} / {len(full_texts)} recipes ...")

    vectors = np.vstack(all_vectors).astype("float32")
    faiss.normalize_L2(vectors)
    print(f"   Encoding complete! Vector matrix shape: {vectors.shape}")

    # Save local indices
    print("   Saving updated local vector store files ...")
    DIM = vectors.shape[1]
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)
    np.save(VECS_PATH, vectors)

    metadata = []
    for i, r in enumerate(combined_recipes):
        metadata.append({
            "id":              i,
            "title":           r["title"],
            "category":        r["category"],
            "ingredients":     r["ingredients"],
            "ingredients_raw": r["ingredients_raw"],
            "calories":        r["calories"],
            "protein":         r["protein"],
            "fat":             r["fat"],
            "rating":          r["rating"],
            "rating_count":    r["rating_count"],
            "servings":        r["servings"],
            "steps":           r.get("steps", []),
            "state":           r["state"],
            "text":            recipe_texts[i],
        })

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(recipe_texts, f)

    print("   Local vector store updated successfully.")

    # 4. Upload to NEW Qdrant Cloud DB
    print(f"\n[5/5] Connecting and uploading to NEW Qdrant Cloud cluster ...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = QdrantClient(
        url                 = QDRANT_URL,
        api_key             = QDRANT_API_KEY,
        timeout             = QDRANT_TIMEOUT,
        check_compatibility = False,
    )

    # Create / Recreate collection
    existing_cols = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing_cols:
        print(f"   Recreating collection '{COLLECTION_NAME}' in new cluster ...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name = COLLECTION_NAME,
        vectors_config  = VectorParams(size=DIM, distance=Distance.COSINE),
    )
    print(f"   Collection '{COLLECTION_NAME}' created in new cluster (dim={DIM}, distance=COSINE)")

    # Upload in batches of 100
    UP_BATCH_SIZE = 100
    uploaded      = 0
    t_start       = time.time()

    for b_start in range(0, total_recipes, UP_BATCH_SIZE):
        b_end  = min(b_start + UP_BATCH_SIZE, total_recipes)
        points = []

        for i in range(b_start, b_end):
            meta = metadata[i]
            points.append(
                PointStruct(
                    id      = i,
                    vector  = vectors[i].tolist(),
                    payload = meta,
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        uploaded += (b_end - b_start)
        elapsed   = time.time() - t_start
        pct       = uploaded / total_recipes * 100
        print(f"   Uploaded {uploaded:>5}/{total_recipes} ({pct:5.1f}%) | Elapsed: {elapsed:.1f}s")

    # Verify upload count
    final_info  = client.get_collection(COLLECTION_NAME)
    cloud_count = final_info.points_count
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Migration to NEW Qdrant Cloud DB Complete!")
    print(f"Total Combined Recipes Uploaded: {cloud_count}")
    print(f"Collection Name: {COLLECTION_NAME}")
    print(f"Endpoint: {QDRANT_URL}")
    print("=" * 60)

if __name__ == "__main__":
    main()
