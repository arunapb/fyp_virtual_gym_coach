"""
qdrant_uploader.py
==================
Run ONCE to upload your local recipe data into Qdrant Cloud.

What it does:
  1. Reads  recipe_metadata.pkl  (title, ingredients, nutrition, ...)
  2. Reads  recipe_texts.pkl     (text strings used for BM25)
  3. Reads  recipe_vectors.npy   (384-dim L2-normalised embeddings)
  4. Creates (or recreates) a "recipes" collection in Qdrant Cloud
  5. Uploads every recipe as a Point:
       id      = integer index  (same as local array position)
       vector  = 384-dim float  (same as recipe_vectors.npy row)
       payload = all metadata fields + text

Usage:
    cd Reccomendation
    python qdrant_uploader.py

After this script completes, retrieval_pipeline.py will automatically
use Qdrant Cloud for semantic search.  Local files are NOT deleted.
"""

import os
import pickle
import sys
import time

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPTS_DIR)
VECTOR_DIR  = os.path.join(BASE_DIR, "vector_store")

META_PATH   = os.path.join(VECTOR_DIR, "recipe_metadata.pkl")
TEXTS_PATH  = os.path.join(VECTOR_DIR, "recipe_texts.pkl")
VECS_PATH   = os.path.join(VECTOR_DIR, "recipe_vectors.npy")

# ── Qdrant config ──────────────────────────────────────────────────────────────
try:
    from qdrant_config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, QDRANT_TIMEOUT
except ImportError:
    print("[ERROR] qdrant_config.py not found. Make sure it exists in the same directory.")
    sys.exit(1)

# ── Check local files ──────────────────────────────────────────────────────────
for path in [META_PATH, TEXTS_PATH, VECS_PATH]:
    if not os.path.exists(path):
        print(f"[ERROR] Required file not found: {path}")
        print("Run build_recipe_index.py first.")
        sys.exit(1)

# ── Install qdrant-client if needed ───────────────────────────────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
except ImportError:
    print("[ERROR] qdrant-client not installed.")
    print("Run:  pip install qdrant-client")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load local files
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1  Loading local files ...")
print("=" * 60)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

vectors = np.load(VECS_PATH)   # shape: [N, 384], float32, L2-normalised

total = len(metadata)
assert len(texts)   == total, "texts count mismatch"
assert vectors.shape[0] == total, "vectors count mismatch"

DIM = vectors.shape[1]
print(f"  Recipes:  {total}")
print(f"  Dims:     {DIM}")
print(f"  Vectors:  {vectors.shape}  dtype={vectors.dtype}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Connect to Qdrant Cloud
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2  Connecting to Qdrant Cloud ...")
print(f"  URL: {QDRANT_URL}")

client = QdrantClient(
    url                 = QDRANT_URL,
    api_key             = QDRANT_API_KEY,
    timeout             = QDRANT_TIMEOUT,
    check_compatibility = False,
)

try:
    info = client.get_collections()
    existing = [c.name for c in info.collections]
    print(f"  Connected.  Existing collections: {existing}")
except Exception as e:
    print(f"\n[ERROR] Cannot connect to Qdrant: {e}")
    print("Check your URL and API key in qdrant_config.py")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Create / Recreate collection
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 3  Setting up collection '{COLLECTION_NAME}' ...")

if COLLECTION_NAME in existing:
    existing_info = client.get_collection(COLLECTION_NAME)
    existing_count = existing_info.points_count
    print(f"  Collection exists with {existing_count} points.")
    answer = input(f"  Recreate it? This will DELETE existing data. [y/N]: ").strip().lower()
    if answer != "y":
        print("  Aborted. Collection unchanged.")
        sys.exit(0)
    client.delete_collection(COLLECTION_NAME)
    print("  Old collection deleted.")

client.create_collection(
    collection_name = COLLECTION_NAME,
    vectors_config  = VectorParams(
        size     = DIM,
        distance = Distance.COSINE,   # cosine = dot on L2-normalised vectors
    ),
)
print(f"  Collection '{COLLECTION_NAME}' created  (dim={DIM}, distance=COSINE)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Upload in batches
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSTEP 4  Uploading {total} recipes to Qdrant ...")

BATCH_SIZE = 100
uploaded   = 0
t_start    = time.time()

for batch_start in range(0, total, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total)
    points    = []

    for i in range(batch_start, batch_end):
        meta = metadata[i]

        # Build payload — all metadata fields + text
        payload = {
            "local_idx":    i,                              # index into recipe_vectors.npy (for MMR)
            "title":        meta.get("title", ""),
            "category":     meta.get("category", ""),
            "ingredients":  meta.get("ingredients", []),    # cleaned list
            "calories":     float(meta.get("calories", 0)),
            "protein":      float(meta.get("protein", 0)),
            "fat":          float(meta.get("fat", 0)),
            "rating":       float(meta.get("rating", 0)),
            "rating_count": int(meta.get("rating_count", 0)),
            "servings":     int(meta.get("servings", 1)),
            "state":        meta.get("state", ""),
            "text":         texts[i],                       # for BM25 rebuild if needed
        }

        # id = local array index so recipe_vectors[id] still works for MMR
        points.append(
            PointStruct(
                id      = i,
                vector  = vectors[i].tolist(),
                payload = payload,
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    uploaded += (batch_end - batch_start)

    elapsed = time.time() - t_start
    pct     = uploaded / total * 100
    rate    = uploaded / elapsed if elapsed > 0 else 0
    eta     = (total - uploaded) / rate if rate > 0 else 0
    print(f"  Uploaded {uploaded:>5}/{total}  ({pct:5.1f}%)  "
          f"elapsed={elapsed:.1f}s  ETA={eta:.0f}s")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Verify
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5  Verifying upload ...")

final_info = client.get_collection(COLLECTION_NAME)
cloud_count = final_info.points_count

print(f"  Local recipes:  {total}")
print(f"  Qdrant points:  {cloud_count}")

if cloud_count == total:
    print("\n[OK] Upload COMPLETE -- all recipes are in Qdrant Cloud.")
    print(f"     Collection : {COLLECTION_NAME}")
    print(f"     URL        : {QDRANT_URL}")
    print(f"     Total time : {time.time() - t_start:.1f}s")
    print("\nLocal files are UNCHANGED. retrieval_pipeline.py will now")
    print("automatically use Qdrant for semantic search (with FAISS fallback).")
else:
    print(f"\n[WARNING] Count mismatch: local={total}, cloud={cloud_count}")
    print("Some points may have failed. Try running the script again.")
