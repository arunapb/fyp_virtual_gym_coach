"""
build_recipe_index.py
=====================
Run ONCE to build FAISS index from newdataset.json.

Usage:
    python build_recipe_index.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import os
import pickle
import re

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPTS_DIR)
VECTOR_DIR  = os.path.join(BASE_DIR, "vector_store")

DATA_PATH  = os.path.join(VECTOR_DIR, "newdataset.json")
INDEX_PATH = os.path.join(VECTOR_DIR, "recipe_index.faiss")
META_PATH  = os.path.join(VECTOR_DIR, "recipe_metadata.pkl")
TEXTS_PATH = os.path.join(VECTOR_DIR, "recipe_texts.pkl")
VECS_PATH  = os.path.join(VECTOR_DIR, "recipe_vectors.npy")

# ── Ingredient cleaning ────────────────────────────────────────────────────────
_UNIT_WORDS = {
    "teaspoon", "teaspoons", "tsp",
    "tablespoon", "tablespoons", "tbsp",
    "cup", "cups",
    "pound", "pounds", "lb", "lbs",
    "ounce", "ounces", "oz",
    "gram", "grams", "g",
    "ml", "milliliter", "milliliters",
    "liter", "liters", "l",
    "small", "medium", "large", "extra",
    "clove", "cloves",
    "chopped", "minced", "diced", "sliced",
    "crushed", "shredded", "grated",
    "to", "taste", "or", "and",
}

_LEADING_NUM_RE = re.compile(r"^\s*[\d/\-]+\s*")


def _clean_ingredient(raw: str) -> str:
    """Strip leading quantities and common unit/modifier words from an ingredient."""
    text = _LEADING_NUM_RE.sub("", raw.strip())
    words = text.split()
    while words and words[0].lower().rstrip(".,") in _UNIT_WORDS:
        words = words[1:]
    while words and words[-1].lower().rstrip(".,") in _UNIT_WORDS:
        words = words[:-1]
    result = " ".join(words).strip(" ,.")
    return result if result else raw.strip()


# ============================================================
# STEP 1 -- Load JSON
# ============================================================
print("=" * 60)
print("STEP 1  Loading JSON ...")
print("=" * 60)

with open(DATA_PATH, encoding="utf-8") as f:
    raw = json.load(f)

total_in_json = len(raw)
print(f"Total recipes loaded: {total_in_json}")

# ============================================================
# STEP 2 -- Parse each recipe
# ============================================================
print("\nSTEP 2  Parsing recipes ...")

parsed = []
parse_errors = 0

for item in raw:
    try:
        basic = item.get("basic_info", {})
        prep  = item.get("prep_data",  {})
        nutr  = item.get("nutritions", {})

        title    = basic.get("title", "Unknown")
        category = basic.get("category", "")

        # Rating
        try:
            rating = float(basic.get("rating", "0").strip().replace("\n", ""))
        except (ValueError, AttributeError):
            rating = 0.0

        # Rating count
        try:
            rc_str = basic.get("rating_count", "(0)")
            rating_count = int(
                rc_str.strip()
                      .replace("\n", "")
                      .replace("(", "")
                      .replace(")", "")
                      .replace(",", "")
            )
        except (ValueError, AttributeError):
            rating_count = 0

        # Servings
        try:
            servings = int(prep.get("servings:", "1").strip().split()[0])
        except (ValueError, AttributeError, IndexError):
            servings = 1

        # Total time
        total_time = prep.get("total_time:", "")

        # Ingredients  (field name in JSON is "ingridients" -- NOT "ingredients")
        ingredients_raw_list = [str(ing) for ing in item.get("ingridients", [])]
        ingredients_clean    = []
        for ing in ingredients_raw_list:
            cleaned = _clean_ingredient(ing)
            if cleaned:
                ingredients_clean.append(cleaned)

        # Nutrition
        calories_str = nutr.get("calories", "0")
        try:
            calories = float(
                str(calories_str).strip()
                                 .replace("g",  "")
                                 .replace("mg", "")
                                 .replace(",",  "")
            )
        except ValueError:
            calories = -1.0  # will be filtered

        protein_str = nutr.get("protein", "0g")
        try:
            protein = float(str(protein_str).strip().replace("g", "").replace("mg", ""))
        except ValueError:
            protein = 0.0

        fat_str = nutr.get("fat", "0g")
        try:
            fat = float(str(fat_str).strip().replace("g", ""))
        except ValueError:
            fat = 0.0

        state = item.get("state", "")

        parsed.append({
            "title":              title,
            "category":           category,
            "ingredients":        ingredients_clean,
            "ingredients_raw":    ingredients_raw_list,
            "calories":           calories,
            "protein":            protein,
            "fat":                fat,
            "rating":             rating,
            "rating_count":       rating_count,
            "servings":           servings,
            "total_time":         total_time,
            "state":              state,
            "nutritions":         nutr,
        })
    except Exception:
        parse_errors += 1
        continue

print(f"  Parsed: {len(parsed)}  |  Parse errors skipped: {parse_errors}")

# ============================================================
# STEP 3 -- Filter recipes
# ============================================================
print("\nSTEP 3  Filtering ...")

PRIMARY_KEYWORDS   = ["main", "dish"]
SECONDARY_KEYWORDS = ["salad", "soup", "stew", "casserole"]


def _passes_primary(cat) -> bool:
    c = (cat or "").lower()
    return any(kw in c for kw in PRIMARY_KEYWORDS)


after_category    = [r for r in parsed if _passes_primary(r["category"])]
main_dishes_count = len(after_category)

print(f"  Total loaded:                    {total_in_json}")
print(f"  After category filter (Main):    {main_dishes_count}")

# If fewer than 500 main dishes, widen to include extra categories
if main_dishes_count < 500:
    print(f"  [!] Fewer than 500 Main Dishes found ({main_dishes_count}).")
    print("  Expanding to include: Salad, Soup, Stew, Casserole ...")
    after_category = [
        r for r in parsed
        if _passes_primary(r["category"])
        or any(kw in (r["category"] or "").lower() for kw in SECONDARY_KEYWORDS)
    ]
    print(f"  After expanded category filter:  {len(after_category)}")

# Nutrition filter
skipped_no_nutrition = 0
after_nutrition = []
for r in after_category:
    cal = r["calories"]
    if cal <= 0 or cal > 2000:
        skipped_no_nutrition += 1
        continue
    if r["nutritions"] == {}:
        skipped_no_nutrition += 1
        continue
    after_nutrition.append(r)

print(f"  After nutrition filter:          {len(after_nutrition)}")
print(f"  Skipped (bad/missing nutrition): {skipped_no_nutrition}")

final_recipes = after_nutrition
print(f"  Final recipes for indexing:      {len(final_recipes)}")

if len(final_recipes) == 0:
    print("\n[ERROR] No recipes passed the filters. Aborting.")
    sys.exit(1)

# ============================================================
# STEP 4 -- Build recipe texts
# ============================================================
print("\nSTEP 4  Building recipe texts ...")

PREFIX = "Represent a recipe for health-aware meal recommendation retrieval: "

recipe_texts = []   # plain texts without prefix  -- saved separately
full_texts   = []   # with prefix                 -- used for encoding

for r in final_recipes:
    recipe_text = (
        r["title"]
        + " " + " ".join(r["ingredients"])
        + " " + r["state"]
    ).strip()
    recipe_texts.append(recipe_text)
    full_texts.append(PREFIX + recipe_text)

print(f"  Built {len(recipe_texts)} recipe texts")

# STEP 5 -- Encode with SentenceTransformer
print("\nSTEP 5  Encoding with SentenceTransformer (all-MiniLM-L6-v2) ...")

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Install with:  pip install faiss-cpu sentence-transformers")
    sys.exit(1)

model = SentenceTransformer("all-MiniLM-L6-v2")

BATCH_SIZE  = 64
all_vectors = []

for start in range(0, len(full_texts), BATCH_SIZE):
    batch = full_texts[start : start + BATCH_SIZE]
    vecs  = model.encode(batch, normalize_embeddings=False, show_progress_bar=False)
    all_vectors.append(vecs)
    if start > 0 and (start // BATCH_SIZE) % (1000 // BATCH_SIZE) == 0:
        print(f"  Encoded {start} / {len(full_texts)} ...")

vectors = np.vstack(all_vectors).astype("float32")
faiss.normalize_L2(vectors)
print(f"  Encoding complete. Vector shape: {vectors.shape}")

# STEP 6 -- Build & save FAISS index
print("\nSTEP 6  Building FAISS index ...")

DIM   = vectors.shape[1]   # 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatIP(DIM)
index.add(vectors)

faiss.write_index(index, INDEX_PATH)
np.save(VECS_PATH, vectors)

print(f"  Index saved   -> {INDEX_PATH}")
print(f"  Vectors saved -> {VECS_PATH}")

# ============================================================
# STEP 7 -- Save metadata and texts
# ============================================================
print("\nSTEP 7  Saving metadata ...")

metadata = []
for i, r in enumerate(final_recipes):
    metadata.append({
        "id":           i,
        "title":        r["title"],
        "category":     r["category"],
        "ingredients":  r["ingredients"],
        "calories":     r["calories"],
        "protein":      r["protein"],
        "fat":          r["fat"],
        "rating":       r["rating"],
        "rating_count": r["rating_count"],
        "servings":     r["servings"],
        "state":        r["state"],
        "text":         recipe_texts[i],
    })

with open(META_PATH,  "wb") as f:
    pickle.dump(metadata, f)

with open(TEXTS_PATH, "wb") as f:
    pickle.dump(recipe_texts, f)

print(f"  Metadata saved -> {META_PATH}")
print(f"  Texts saved    -> {TEXTS_PATH}")

# ============================================================
# FINAL STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

all_cal     = [r["calories"]     for r in final_recipes]
all_prot    = [r["protein"]      for r in final_recipes]
all_ratings = [r["rating"]       for r in final_recipes if r["rating"] > 0]
all_rc      = [r["rating_count"] for r in final_recipes if r["rating_count"] > 0]
all_states  = set(r["state"] for r in final_recipes if r["state"])

avg_cal    = sum(all_cal)     / len(all_cal)     if all_cal     else 0
avg_prot   = sum(all_prot)    / len(all_prot)    if all_prot    else 0
avg_rating = sum(all_ratings) / len(all_ratings) if all_ratings else 0
avg_rc     = sum(all_rc)      / len(all_rc)      if all_rc      else 0

print(f"Total recipes in JSON:    {total_in_json}")
print(f"Main Dishes found:        {main_dishes_count}")
print(f"With nutrition data:      {len(after_nutrition)}")
print(f"Final indexed:            {len(final_recipes)}")
print(f"Average calories:         {avg_cal:.0f}")
print(f"Average protein:          {avg_prot:.1f}g")
print(f"States covered:           {len(all_states)}")
print(f"Average rating:           {avg_rating:.1f}")
print(f"Average rating count:     {avg_rc:.0f}")
print("=" * 60)
print("Index and metadata ready. Run run_retrieval_test.py next.")
