"""
retrieval_service.py
====================
FAISS + BM25 + MMR retrieval pipeline — ported from Reccomendation/retrieval_pipeline.py.

All index files, models, and metadata (including steps & ingredients_raw) are loaded
ONCE at module import time so that every API request is served from memory (fast).

Exposes:
    get_top_recipes(meal_type, targets, prefs, n_feedback, top_n, exclude_titles)
"""

import logging
import os
import pickle
import sys

import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_SERVICES_DIR    = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR     = os.path.dirname(_SERVICES_DIR)
_COMBINE_DIR     = os.path.dirname(_BACKEND_DIR)
_VECTOR_DIR      = os.path.join(_COMBINE_DIR, "vector_store")
_SCRIPTS_DIR     = os.path.join(_COMBINE_DIR, "scripts")

INDEX_PATH   = os.path.join(_VECTOR_DIR, "recipe_index.faiss")
META_PATH    = os.path.join(_VECTOR_DIR, "recipe_metadata.pkl")
TEXTS_PATH   = os.path.join(_VECTOR_DIR, "recipe_texts.pkl")
VECTORS_PATH = os.path.join(_VECTOR_DIR, "recipe_vectors.npy")

QUERY_PREFIX = (
    "Represent a meal preference for healthy food "
    "recommendation based on dietary feedback: "
)

# ── Qdrant config (optional -- falls back to FAISS if unavailable) ─────────────
# qdrant_config.py lives in scripts/ -- add it to path temporarily
try:
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    from qdrant_config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, QDRANT_TIMEOUT
    _QDRANT_CONFIG_LOADED = True
except ImportError:
    _QDRANT_CONFIG_LOADED = False
    QDRANT_URL = QDRANT_API_KEY = COLLECTION_NAME = None
    QDRANT_TIMEOUT = 15

# ── Startup loading ────────────────────────────────────────────────────────────
logger.info("[retrieval_service] Loading index and models ...")

try:
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logger.error("Missing dependency: %s", e)
    raise

for _path in [INDEX_PATH, META_PATH, TEXTS_PATH, VECTORS_PATH]:
    if not os.path.exists(_path):
        raise FileNotFoundError(
            f"[retrieval_service] Required file not found: {_path}\n"
            f"Run scripts/build_recipe_index.py first."
        )

logger.info("[retrieval_service]   Loading FAISS index (always loaded -- used as fallback + MMR) ...")
index = faiss.read_index(INDEX_PATH)
logger.info("[retrieval_service]   FAISS index loaded: %d vectors", index.ntotal)

# ── Qdrant Cloud connection (tried after FAISS confirms loaded) ────────────────
USE_QDRANT    = False
qdrant_client = None

if _QDRANT_CONFIG_LOADED:
    try:
        from qdrant_client import QdrantClient as _QdrantClient
        logger.info("[retrieval_service]   Connecting to Qdrant Cloud: %s", QDRANT_URL)
        _qc   = _QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY,
                               timeout=QDRANT_TIMEOUT, check_compatibility=False)
        _cols = [c.name for c in _qc.get_collections().collections]
        if COLLECTION_NAME in _cols:
            _info    = _qc.get_collection(COLLECTION_NAME)
            _n_pts   = _info.points_count
            if _n_pts and _n_pts > 0:
                qdrant_client = _qc
                USE_QDRANT    = True
                logger.info(
                    "[retrieval_service]   Qdrant Cloud CONNECTED -- collection '%s' has %d recipes.",
                    COLLECTION_NAME, _n_pts,
                )
                logger.info("[retrieval_service]   Stage 3 will use Qdrant Cloud (FAISS kept as fallback).")
            else:
                logger.warning(
                    "[retrieval_service]   Qdrant collection '%s' is EMPTY. "
                    "Run qdrant_uploader.py first. Falling back to FAISS.",
                    COLLECTION_NAME,
                )
        else:
            logger.warning(
                "[retrieval_service]   Qdrant collection '%s' NOT FOUND. "
                "Run qdrant_uploader.py first. Falling back to FAISS.",
                COLLECTION_NAME,
            )
    except Exception as _qe:
        logger.warning(
            "[retrieval_service]   Qdrant connection FAILED: %s. Falling back to FAISS.", _qe
        )
else:
    logger.info("[retrieval_service]   qdrant_config.py not found -- using local FAISS only.")

logger.info("[retrieval_service]   Loading recipe metadata ...")
with open(META_PATH, "rb") as _f:
    recipe_metadata = pickle.load(_f)

def _reload_metadata_if_needed():
    global recipe_metadata
    if recipe_metadata and "steps" not in recipe_metadata[0]:
        try:
            with open(META_PATH, "rb") as _f:
                recipe_metadata = pickle.load(_f)
            logger.info("[retrieval_service] Reloaded metadata with %d items.", len(recipe_metadata))
        except Exception as _e:
            logger.warning("[retrieval_service] Could not reload metadata: %s", _e)

logger.info("[retrieval_service]   Loading recipe texts ...")
with open(TEXTS_PATH, "rb") as _f:
    recipe_texts = pickle.load(_f)

logger.info("[retrieval_service]   Loading recipe vectors ...")
recipe_vectors = np.load(VECTORS_PATH)   # shape [N, 384], float32, L2-normalised

logger.info("[retrieval_service]   Building BM25 index ...")
_tokenized_corpus = [t.lower().split() for t in recipe_texts]
bm25 = BM25Okapi(_tokenized_corpus)

logger.info("[retrieval_service]   Loading SentenceTransformer ...")
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Build a title → raw ingredients lookup from the original dataset.
# This lets us return the full ingredient strings (with quantities) in the
# API response without changing the FAISS index or metadata.
logger.info("[retrieval_service]   Building raw-ingredient lookup from newdataset.json ...")
_raw_ingredients_by_title: dict = {}
try:
    import json as _json
    _DATASET_PATH = os.path.join(_VECTOR_DIR, "newdataset.json")
    with open(_DATASET_PATH, encoding="utf-8") as _df:
        _raw_data = _json.load(_df)
    for _item in _raw_data:
        _title = (_item.get("basic_info") or {}).get("title", "").strip()
        _ings  = [str(i) for i in _item.get("ingridients", [])]
        if _title:
            _raw_ingredients_by_title[_title] = _ings
    logger.info("[retrieval_service]   Raw-ingredient lookup: %d titles.", len(_raw_ingredients_by_title))
except Exception as _e:
    logger.warning("[retrieval_service]   Could not build raw-ingredient lookup: %s", _e)

# ── Ingredient embedding cache for semantic matching (NOT yet wired into the live
#    pipeline -- get_top_recipes() Stage 5 hard filter and pref_match calculation
#    still use _item_has_term()/FLAVOR_SYNONYMS below. This cache + _semantic_match()
#    exist purely so the embedding-based approach can be diagnosed/compared against
#    the keyword-list approach before deciding whether to switch over. ────────────
logger.info("[retrieval_service]   Precomputing ingredient embeddings for semantic matching ...")
_ingredient_embedding_cache: dict = {}
try:
    _unique_ingredients = sorted({
        ing.strip().lower()
        for _recipe in recipe_metadata
        for ing in _recipe.get("ingredients", [])
        if ing and ing.strip()
    })
    if _unique_ingredients:
        _ingredient_embeddings = _embed_model.encode(
            _unique_ingredients, normalize_embeddings=True, show_progress_bar=False
        )
        _ingredient_embedding_cache = {
            name: vec.astype("float32") for name, vec in zip(_unique_ingredients, _ingredient_embeddings)
        }
    logger.info(
        "[retrieval_service]   Ingredient embedding cache built: %d unique ingredients.",
        len(_ingredient_embedding_cache),
    )
except Exception as _e:
    logger.warning("[retrieval_service]   Could not build ingredient embedding cache: %s", _e)


def _get_text_embedding(text: str) -> np.ndarray:
    """Return a cached embedding for `text` if it's a known ingredient name; otherwise compute + cache it on the fly."""
    key = text.strip().lower()
    vec = _ingredient_embedding_cache.get(key)
    if vec is None:
        vec = _embed_model.encode([key], normalize_embeddings=True, show_progress_bar=False)[0].astype("float32")
        _ingredient_embedding_cache[key] = vec
    return vec


def _semantic_match(preference_term: str, ingredient_text: str, threshold: float = 0.5) -> bool:
    """
    Embedding-based alternative to the FLAVOR_SYNONYMS keyword-list approach used by
    _item_has_term(). Encodes `preference_term` and `ingredient_text` with the same
    SentenceTransformer model used elsewhere in this pipeline, and returns True if
    their cosine similarity is >= threshold.

    NOT currently called from get_top_recipes() -- exists for side-by-side diagnostic
    comparison against the keyword-list approach before switching the live pipeline over.
    """
    pref_vec = _get_text_embedding(preference_term)
    ing_vec  = _get_text_embedding(ingredient_text)
    similarity = float(np.dot(pref_vec, ing_vec))
    return similarity >= threshold


logger.info("[retrieval_service] Ready.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _explain(recipe: dict, prefs: dict, targets: dict, meal_type: str):
    reasons     = []
    ingredients = [ing.lower() for ing in recipe.get("ingredients", [])]

    liked = {k: v for k, v in prefs.items() if v > 0.30}
    for pref_word, score in sorted(liked.items(), key=lambda x: x[1], reverse=True):
        matched = [ing for ing in ingredients if pref_word in ing]
        if matched:
            strength = "strongly" if score > 0.60 else "moderately"
            reasons.append(f"Contains {matched[0]} (you {strength} like {pref_word})")

    disliked = {k: v for k, v in prefs.items() if v <= -0.20}
    for pref_word in sorted(disliked, key=lambda k: disliked[k]):
        if not any(pref_word in ing for ing in ingredients):
            reasons.append(f"No {pref_word} (you dislike {pref_word})")

    rec_cal    = recipe.get("calories", 0)
    target_cal = targets.get("calories", 1)
    cal_diff   = abs(rec_cal - target_cal)
    if cal_diff <= 50:
        reasons.append(f"Calories {rec_cal:.0f} kcal — very close to {meal_type} target {target_cal} kcal")
    elif cal_diff <= 150:
        reasons.append(f"Calories {rec_cal:.0f} kcal — reasonably close to {meal_type} target {target_cal} kcal")
    else:
        reasons.append(f"Calories {rec_cal:.0f} kcal (target {target_cal} kcal)")

    rec_prot    = recipe.get("protein", 0)
    target_prot = targets.get("protein", 1)
    if abs(rec_prot - target_prot) <= 5:
        reasons.append(f"Protein {rec_prot:.0f}g — matches target {target_prot}g")
    else:
        reasons.append(f"Protein {rec_prot:.0f}g (target {target_prot}g)")

    rating = recipe.get("rating", 0)
    rc     = recipe.get("rating_count", 0)
    if rating >= 4.5 and rc >= 100:
        reasons.append(f"Highly rated {rating:.1f}/5 by {rc} users")
    elif rating >= 4.0:
        reasons.append(f"Well rated {rating:.1f}/5")

    return reasons


def _contains_preferred(recipe: dict, positive_words: list) -> bool:
    title = recipe.get("title", "")
    ings  = recipe.get("ingredients", [])
    text  = (title + " " + " ".join(ings)).lower()
    return any(w.lower() in text for w in positive_words)


def _close_to_target(recipe: dict, target_cal: float, tolerance: float = 0.20) -> bool:
    rec_cal = recipe.get("calories", 0)
    if not rec_cal or not target_cal:
        return False
    return abs(rec_cal - target_cal) / target_cal <= tolerance


# ── Public API ─────────────────────────────────────────────────────────────────

def get_top_recipes(
    meal_type: str,
    targets: dict,
    prefs: dict,
    n_feedback: int,
    top_n: int = 3,
    exclude_titles: set = None,
    threshold1: float = 0.30,
    threshold2: float = 0.30,
    mmr_lambda: float = 0.5,
    daily_ingredient_counts: dict = None,
    max_per_preferred_ingredient: int = 2,
) -> list:
    """
    Retrieve top-N diverse, preference-aware recipes via BM25 + FAISS + MMR.

    Parameters
    ----------
    meal_type      : "Breakfast" | "Lunch" | "Dinner"
    targets        : {"calories": float, "protein": float}
    prefs          : {ingredient: score}  (positive=like, negative=dislike)
    n_feedback     : number of feedback interactions so far (drives RRF weight)
    top_n          : recipes to return
    exclude_titles : meal titles already chosen (prevents cross-meal duplicates)
    threshold1     : query building threshold (default 0.30)
    threshold2     : hard filter threshold for disliked ingredients (default 0.30)
    mmr_lambda     : MMR relevance/diversity trade-off (default 0.5 -- equal weight;
                     higher favors relevance, lower favors diversity)
    daily_ingredient_counts : optional dict the CALLER owns and passes into every
                     get_top_recipes() call for the same day (Breakfast/Lunch/Dinner),
                     e.g. recommender_service.get_daily_meal_plan(). Mutated in place
                     to accumulate how many recipes matching each preferred ingredient
                     have been selected so far *today*. When None (the default), no
                     preferred-ingredient cap is applied at all -- a standalone call
                     (e.g. just "Lunch" on its own) is unaffected and never self-caps.
    max_per_preferred_ingredient : cap enforced only when daily_ingredient_counts is
                     provided (default 2)

    Returns
    -------
    list of recipe dicts enriched with rrf_score, final_score, explanation
    """
    exclude_titles = exclude_titles or set()
    _reload_metadata_if_needed()

    # Stage 1 — Query construction
    target_cal  = float(targets["calories"])
    target_prot = float(targets["protein"])
    positive_terms = [ing for ing, s in prefs.items() if s >= threshold1]

    # BM25 query only includes positive preference terms to avoid matching negated keywords
    query = (
        meal_type
        + (" " + " ".join(positive_terms) if positive_terms else "")
        + f" {int(target_cal)} calories"
        + f" {int(target_prot)}g protein"
    ).strip()
    logger.debug("[retrieval] Query: %s", query)

    # Stage 2 — BM25 lexical retrieval (top-300)
    bm25_scores  = bm25.get_scores(query.lower().split())
    bm25_top_idx = np.argsort(bm25_scores)[-300:][::-1].tolist()
    bm25_ranks   = {idx: rank for rank, idx in enumerate(bm25_top_idx)}

    # Stage 3 -- Semantic retrieval: Qdrant Cloud OR local FAISS (top-300)
    query_vec = np.array(
        _embed_model.encode([QUERY_PREFIX + query], normalize_embeddings=True),
        dtype="float32",
    )

    if USE_QDRANT:
        try:
            qdrant_results = qdrant_client.query_points(
                collection_name = COLLECTION_NAME,
                query           = query_vec[0].tolist(),
                limit           = 300,
                with_payload    = False,
            ).points
            faiss_top_idx = [int(r.id) for r in qdrant_results]
            faiss_scores  = {int(r.id): float(r.score) for r in qdrant_results}
            faiss_ranks   = {idx: rank for rank, idx in enumerate(faiss_top_idx)}
            logger.info("[retrieval] Stage 3: Qdrant Cloud returned %d candidates", len(faiss_top_idx))
        except Exception as _qe:
            logger.warning("[retrieval] Stage 3: Qdrant search failed (%s) -- falling back to FAISS", _qe)
            distances, indices = index.search(query_vec, 300)
            faiss_top_idx = indices[0].tolist()
            faiss_scores  = {idx: float(distances[0][i]) for i, idx in enumerate(faiss_top_idx)}
            faiss_ranks   = {idx: rank for rank, idx in enumerate(faiss_top_idx)}
    else:
        distances, indices = index.search(query_vec, 300)
        faiss_top_idx = indices[0].tolist()
        faiss_scores  = {idx: float(distances[0][i]) for i, idx in enumerate(faiss_top_idx)}
        faiss_ranks   = {idx: rank for rank, idx in enumerate(faiss_top_idx)}
        logger.info("[retrieval] Stage 3: FAISS local search returned 300 candidates")

    liked_words    = [k for k, v in prefs.items() if v >= threshold1]
    disliked_words = [k for k, v in prefs.items() if v <= -threshold1]

    if liked_words or disliked_words:
        pos_vec = (
            np.array(_embed_model.encode([" ".join(liked_words)], normalize_embeddings=True), dtype="float32")
            if liked_words else None
        )
        neg_vec = (
            np.array(_embed_model.encode([" ".join(disliked_words)], normalize_embeddings=True), dtype="float32")
            if disliked_words else None
        )
        contrastive_map = {
            idx: (float(np.dot(recipe_vectors[idx], pos_vec[0])) if pos_vec is not None else 0.0)
                 - (float(np.dot(recipe_vectors[idx], neg_vec[0])) if neg_vec is not None else 0.0)
            for idx in faiss_top_idx
        }
        faiss_top_idx = sorted(faiss_top_idx, key=lambda i: contrastive_map[i], reverse=True)
        faiss_scores  = contrastive_map
        faiss_ranks   = {idx: rank for rank, idx in enumerate(faiss_top_idx)}

    # Stage 4 — Adaptive RRF + Nutrition scoring
    n = n_feedback
    g = min(0.50, n / 100.0)
    a = (1.0 - g) * 0.30
    b = (1.0 - g) * 0.70

    # Scale of a single a/(60+rank) term at rank 0 -- used to bring pref_match onto
    # the same order of magnitude as the RRF rank terms instead of dominating them.
    MAX_RRF_TERM = 1.0 / 60

    candidates    = set(bm25_top_idx) | set(faiss_top_idx)
    candidate_rrf = {}

    FLAVOR_SYNONYMS = {
        "spicy":   {"spicy", "chili", "chilli", "jalapeno", "hot sauce", "cayenne", "tabasco", "habanero", "sriracha", "chili powder", "pepper flakes"},
        "chili":   {"spicy", "chili", "chilli", "jalapeno", "hot sauce", "cayenne", "tabasco", "habanero", "sriracha", "chili powder"},
        "seafood": {"seafood", "salmon", "shrimp", "crab", "lobster", "tuna", "fish", "tilapia", "halibut", "cod", "scallop", "mussel", "clam", "prawn"},
        "dairy":   {"dairy", "milk", "cheese", "cream", "butter", "yogurt", "yoghurt", "sour cream", "buttermilk"},
        "poultry": {"poultry", "chicken", "turkey", "duck", "hen"},
    }

    def _item_has_term(pref_word: str, recipe_dict: dict) -> bool:
        pw = pref_word.lower()
        title_ing_text = (recipe_dict.get("title", "") + " " + " ".join(recipe_dict.get("ingredients", []))).lower()
        syns = FLAVOR_SYNONYMS.get(pw, {pw})
        return any(syn in title_ing_text for syn in syns)

    for idx in candidates:
        bm25_r  = bm25_ranks.get(idx, 301)
        faiss_r = faiss_ranks.get(idx, 301)
        recipe  = recipe_metadata[idx]
        pref_match = sum(
            pref_score
            for pref_word, pref_score in prefs.items()
            if _item_has_term(pref_word, recipe)
        )
        pref_match = max(-1.0, min(1.0, pref_match)) * MAX_RRF_TERM
        candidate_rrf[idx] = (
            a * 1.0 / (60 + bm25_r)
            + b * 1.0 / (60 + faiss_r)
            + g * pref_match
        )

    # Min-max normalize rrf_score across this candidate pool so it occupies a proper
    # 0-1 range before blending -- its raw scale (~0-0.017) would otherwise be negligible
    # next to nut_score (~0-1) despite the stated 15% weight below.
    rrf_values = list(candidate_rrf.values())
    rrf_min    = min(rrf_values) if rrf_values else 0.0
    rrf_max    = max(rrf_values) if rrf_values else 0.0
    rrf_range  = rrf_max - rrf_min

    nutrition_final_scores = {}
    for idx, rrf_score in candidate_rrf.items():
        recipe   = recipe_metadata[idx]
        cal_score  = max(0.0, 1.0 - abs(recipe["calories"] - target_cal) / max(target_cal, 1.0))
        prot_score = max(0.0, 1.0 - abs(recipe["protein"]  - target_prot) / max(target_prot, 1.0))
        nut_score  = 0.6 * cal_score + 0.4 * prot_score
        rrf_norm   = (rrf_score - rrf_min) / rrf_range if rrf_range > 0 else 0.0
        final      = nut_score * 0.85 + rrf_norm * 0.15

        nutrition_final_scores[idx] = (rrf_score, final)

    sorted_candidates = sorted(
        nutrition_final_scores, key=lambda i: nutrition_final_scores[i][1], reverse=True
    )[:150]

    # Hard filter for strongly disliked ingredients (checking title and ingredients using flavor synonyms)
    filtered = [
        idx for idx in sorted_candidates
        if not any(
            _item_has_term(dw, recipe_metadata[idx])
            for dw in disliked_words
        )
    ]
    if not filtered:
        filtered = sorted_candidates

    # Exclude already-used meal titles
    exclude_norm = {t.strip().lower() for t in exclude_titles}
    filtered = [idx for idx in filtered if recipe_metadata[idx]["title"].strip().lower() not in exclude_norm]
    if not filtered:
        filtered = sorted_candidates

    candidate_scores = {idx: nutrition_final_scores[idx][1] for idx in filtered}

    # Stage 5 — MMR diversity selection
    # Diversity must never override a poor calorie/protein match: restrict the MMR
    # candidate pool to recipes already close to the nutrition target (same "close"
    # definition used elsewhere in the pipeline, see _close_to_target) before MMR
    # runs, so diversity only decides *which* good-match recipes get picked -- it
    # never pulls in a bad-calorie-match recipe just because it adds variety.
    mmr_pool = [idx for idx in filtered if _close_to_target(recipe_metadata[idx], target_cal)]
    if not mmr_pool:
        mmr_pool = filtered

    selected = []
    while len(selected) < top_n:
        best_idx, best_mmr = None, -9999.0
        for idx in mmr_pool:
            if idx in selected:
                continue
            relevance = candidate_scores[idx]
            if selected:
                max_sim = max(float(np.dot(recipe_vectors[idx], recipe_vectors[s])) for s in selected)
            else:
                max_sim = 0.0
            mmr = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            if mmr > best_mmr:
                best_mmr, best_idx = mmr, idx
        if best_idx is None:
            break
        selected.append(best_idx)

    # Soft preferred-ingredient guarantee
    positive_words = [ing for ing, s in prefs.items() if s >= threshold1]
    if positive_words and selected:
        has_preferred = any(_contains_preferred(recipe_metadata[i], positive_words) for i in selected)
        if not has_preferred:
            pref_pool = [
                idx for idx in filtered
                if idx not in selected
                and _contains_preferred(recipe_metadata[idx], positive_words)
                and _close_to_target(recipe_metadata[idx], target_cal)
            ]
            if pref_pool:
                best_pref    = max(pref_pool, key=lambda i: nutrition_final_scores[i][1])
                weakest      = min(selected, key=lambda i: nutrition_final_scores[i][1])
                selected.remove(weakest)
                selected.append(best_pref)

    # Stage 5b — Cross-slot (daily) preferred-ingredient cap. Isolated post-processing
    # step; does not touch MMR/RRF/nutrition scoring. OPT-IN: only runs when the
    # caller supplies `daily_ingredient_counts`, a dict IT owns and shares across the
    # Breakfast/Lunch/Dinner calls for one day's plan (see
    # recommender_service.get_daily_meal_plan()). A standalone call that doesn't pass
    # this dict is completely unaffected -- no cap is ever applied within a single
    # slot's own request on its own; the constraint only exists across the full day.
    if daily_ingredient_counts is not None and positive_words and selected:
        # Prefer replacement candidates that still respect this slot's own
        # calorie/protein target (mmr_pool); only widen to the full nutrition-ranked
        # `filtered` pool as a last resort when ignoring the cap entirely.
        ranked_pool = mmr_pool if mmr_pool else filtered

        def _matched_preferred(idx):
            recipe = recipe_metadata[idx]
            return [w for w in positive_words if _item_has_term(w, recipe)]

        capped_selected = []
        for idx in selected:
            matches = _matched_preferred(idx)
            if any(daily_ingredient_counts.get(w, 0) >= max_per_preferred_ingredient for w in matches):
                logger.debug(
                    "[retrieval] %s: skipping '%s' -- would push daily total for %s past %d",
                    meal_type, recipe_metadata[idx].get("title", "?"), matches, max_per_preferred_ingredient,
                )
                continue
            capped_selected.append(idx)
            for w in matches:
                daily_ingredient_counts[w] = daily_ingredient_counts.get(w, 0) + 1

        if len(capped_selected) < top_n:
            already_tried = set(selected) | set(capped_selected)
            for idx in ranked_pool:
                if len(capped_selected) >= top_n:
                    break
                if idx in already_tried:
                    continue
                already_tried.add(idx)
                matches = _matched_preferred(idx)
                if any(daily_ingredient_counts.get(w, 0) >= max_per_preferred_ingredient for w in matches):
                    continue
                capped_selected.append(idx)
                for w in matches:
                    daily_ingredient_counts[w] = daily_ingredient_counts.get(w, 0) + 1

        if len(capped_selected) < top_n:
            logger.warning(
                "[retrieval] %s: daily preferred-ingredient cap (max %d) left only %d/%d recipes "
                "-- backfilling remaining slots for THIS SLOT ONLY, ignoring the cap.",
                meal_type, max_per_preferred_ingredient, len(capped_selected), top_n,
            )
            for idx in filtered:
                if len(capped_selected) >= top_n:
                    break
                if idx in capped_selected:
                    continue
                capped_selected.append(idx)
                # Still record what actually got served today, even though the cap
                # was overridden for this slot, so later slots see the true total.
                for w in _matched_preferred(idx):
                    daily_ingredient_counts[w] = daily_ingredient_counts.get(w, 0) + 1

        selected = capped_selected

    # Assemble results
    results = []
    for idx in selected:
        rec  = dict(recipe_metadata[idx])
        rrf_raw, final = nutrition_final_scores[idx]
        rec["rrf_score"]      = rrf_raw
        rec["final_score"]    = final
        rec["explanation"]    = _explain(rec, prefs, targets, meal_type)
        rec["ingredients_raw"] = rec.get("ingredients_raw") or _raw_ingredients_by_title.get(rec.get("title", "").strip(), [])
        rec["steps"]           = rec.get("steps", [])
        rec["instructions"]    = rec.get("instructions") or rec.get("steps", [])
        results.append(rec)

    return results
