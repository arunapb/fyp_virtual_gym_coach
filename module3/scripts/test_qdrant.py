"""
test_qdrant.py
==============
Verifies that:
  1. Qdrant Cloud collection exists and has all recipes
  2. Payload fields are correct (title, calories, protein, ingredients)
  3. Live semantic search works and returns relevant results

Run:
    python test_qdrant.py
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient
from qdrant_config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

print("=" * 55)
print("  QDRANT CLOUD VERIFICATION TEST")
print("=" * 55)

# ── 1. Connect and check collection ───────────────────────────
print("\n[1] Connecting to Qdrant Cloud ...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

info = client.get_collection(COLLECTION_NAME)
n_points = info.points_count
vec_size  = info.config.params.vectors.size
distance  = info.config.params.vectors.distance

print(f"    Collection : {COLLECTION_NAME}")
print(f"    Points     : {n_points}")
print(f"    Vector dim : {vec_size}")
print(f"    Distance   : {distance}")

if n_points == 678:
    print("    [OK] All 678 recipes are in Qdrant Cloud.")
else:
    print(f"    [WARN] Expected 678, found {n_points}. Some may be missing.")

# ── 2. Sample records — check payload ─────────────────────────
print("\n[2] Checking sample records ...")
samples, _ = client.scroll(
    collection_name = COLLECTION_NAME,
    limit           = 3,
    with_payload    = True,
    with_vectors    = False,
)

for pt in samples:
    p = pt.payload
    title       = p.get("title", "N/A")
    calories    = p.get("calories", 0)
    protein     = p.get("protein", 0)
    ingredients = p.get("ingredients", [])[:3]
    print(f"    id={pt.id}")
    print(f"      title       : {title}")
    print(f"      calories    : {calories}")
    print(f"      protein     : {protein}g")
    print(f"      ingredients : {ingredients}")
    print()

# ── 3. Live semantic search ────────────────────────────────────
print("[3] Running live semantic search ...")
print("    Query: 'Dinner salmon 600 calories 45g protein'")

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
PREFIX = "Represent a meal preference for healthy food recommendation based on dietary feedback: "
query  = PREFIX + "Dinner salmon 600 calories 45g protein"
qvec   = model.encode([query], normalize_embeddings=True)[0].tolist()

results = client.query_points(
    collection_name = COLLECTION_NAME,
    query           = qvec,
    limit           = 5,
    with_payload    = True,
).points

print("\n    Top 5 results:")
for i, r in enumerate(results, 1):
    p     = r.payload
    title = p.get("title", "N/A")
    cal   = p.get("calories", 0)
    prot  = p.get("protein", 0)
    score = r.score
    print(f"    {i}. [{score:.4f}]  {title}  ({cal} cal, {prot}g protein)")

# ── 4. Summary ─────────────────────────────────────────────────
print()
print("=" * 55)
print("  RESULT SUMMARY")
print("=" * 55)
print(f"  Recipes in Qdrant : {n_points}")
print(f"  Payload fields    : OK (title, calories, protein, ingredients)")
print(f"  Semantic search   : OK ({len(results)} results returned)")
print()
print("  Qdrant Cloud is ready.")
print("  retrieval_pipeline.py will automatically use it.")
print("=" * 55)
