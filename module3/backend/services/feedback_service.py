"""
feedback_service.py
===================
Full NL-feedback pipeline — ported and unified from feedBack/feedback_pipeline.py.

Pipeline:
  1. Causal detection     (spaCy dependency grammar)
  2. Intent detection     (BART zero-shot: REDUCE / INCREASE / KEEP)
  3. BERT ABSA            (aspect + sentiment extraction)
  4. spaCy adjective fallback
  5. Intensity scoring    (VADER)
  6. Delta computation    (simple explainable rule)
  7. Preference update    (load → clip → save user_preference.json)

All models are loaded once at module import time.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import spacy
import torch
from transformers import BertForTokenClassification, BertTokenizerFast, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR  = os.path.dirname(_SERVICES_DIR)
_PREF_FILE    = os.path.join(_BACKEND_DIR, "data", "user_preference.json")

# ── HF Hub model identifiers ───────────────────────────────────────────────────
_ABSA_HF_REPO   = "nipun145/food_feedback"    # your uploaded BERT ABSA model
_ZERO_SHOT_REPO = "facebook/bart-large-mnli"  # already public on HF, no upload needed

# ── Load BERT ABSA model from HF Hub ──────────────────────────────────────────
logger.info("[feedback_service] Loading BERT ABSA model from HF Hub (%s) ...", _ABSA_HF_REPO)
_label_map_path = os.path.join(_BACKEND_DIR, "data", "label_map.json")
with open(_label_map_path) as _f:
    _label_data = json.load(_f)
_id2label = {int(k): v for k, v in _label_data["id2label"].items()}

_tokenizer  = None
_bert_model = None
_zero_shot  = None

# ── Load spaCy model ───────────────────────────────────────────────────────────
_nlp = None
try:
    _nlp = spacy.load("en_core_web_sm")
    logger.info("[feedback_service] spaCy model (en_core_web_sm) loaded successfully.")
except Exception as _e:
    logger.warning("[feedback_service] Could not load spaCy model: %s", _e)

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        logger.info("[feedback_service] Loading BERT ABSA tokenizer from HF Hub (%s) ...", _ABSA_HF_REPO)
        _tokenizer = BertTokenizerFast.from_pretrained(_ABSA_HF_REPO)
    return _tokenizer

def _get_bert_model():
    global _bert_model
    if _bert_model is None:
        logger.info("[feedback_service] Loading BERT ABSA model from HF Hub (%s) ...", _ABSA_HF_REPO)
        _bert_model = BertForTokenClassification.from_pretrained(_ABSA_HF_REPO)
        _bert_model.eval()
    return _bert_model

def _get_zero_shot():
    global _zero_shot
    if _zero_shot is None:
        logger.info("[feedback_service] Loading BART zero-shot from HF Hub (%s) ...", _ZERO_SHOT_REPO)
        _zero_shot = pipeline("zero-shot-classification", model=_ZERO_SHOT_REPO)
    return _zero_shot

# ── Load VADER intensity analyser ──────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()

# ── Zero-shot labels ───────────────────────────────────────────────────────────
LABEL_REDUCE   = "want less of this ingredient or remove it"
LABEL_INCREASE = "want more of this ingredient or add more of it"
LABEL_KEEP     = "satisfied with the current amount of this ingredient"

_MIXED_LABELS = [
    "this comment has both positive and negative parts",
    "this comment is entirely expressing dislike or complaint",
    "this comment is entirely expressing preference or satisfaction",
]
_ACTION_LABELS = [LABEL_REDUCE, LABEL_INCREASE, LABEL_KEEP]
_CAUSAL_DEPS   = {"advcl"}


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_bert(text: str) -> List[dict]:
    """Run BERT ABSA on text, return list of {aspect, sentiment, confidence}."""
    tok = _get_tokenizer()
    model = _get_bert_model()
    encoding = tok(text, return_tensors="pt", truncation=True, max_length=128)
    word_ids = encoding.word_ids()
    tokens   = tok.convert_ids_to_tokens(encoding["input_ids"][0])

    with torch.no_grad():
        outputs = model(**encoding)

    probs    = torch.softmax(outputs.logits[0], dim=-1)
    pred_ids = torch.argmax(probs, dim=-1).tolist()

    word_info: Dict[int, dict] = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in word_info:
            word_info[wid] = {
                "label":     _id2label[pred_ids[i]],
                "conf":      probs[i][pred_ids[i]].item(),
                "subtokens": [tokens[i]],
            }
        else:
            word_info[wid]["subtokens"].append(tokens[i])

    def _reconstruct(subtokens):
        out = ""
        for t in subtokens:
            out += t[2:] if t.startswith("##") else t
        return out

    aspects, cur_words, cur_sent, cur_confs = [], [], None, []

    for wid in sorted(word_info):
        info  = word_info[wid]
        label = info["label"]
        conf  = info["conf"]
        word  = _reconstruct(info["subtokens"])

        if label.startswith("B-"):
            if cur_words:
                aspects.append({"aspect": " ".join(cur_words),
                                 "sentiment": cur_sent,
                                 "confidence": sum(cur_confs) / len(cur_confs)})
            cur_words, cur_sent, cur_confs = [word], label[2:], [conf]
        elif label.startswith("I-") and cur_words:
            cur_words.append(word)
            cur_confs.append(conf)
        else:
            if cur_words:
                aspects.append({"aspect": " ".join(cur_words),
                                 "sentiment": cur_sent,
                                 "confidence": sum(cur_confs) / len(cur_confs)})
            cur_words, cur_sent, cur_confs = [], None, []

    if cur_words:
        aspects.append({"aspect": " ".join(cur_words),
                         "sentiment": cur_sent,
                         "confidence": sum(cur_confs) / len(cur_confs)})
    return aspects


def _spacy_split(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Split on coordinating / subordinating conjunction using spaCy grammar."""
    if _nlp is None:
        return None, None
    doc    = _nlp(text)
    tokens = list(doc)
    split_idx = None
    for token in tokens:
        if token.dep_ in ("cc", "mark"):
            split_idx = token.i
            break
    if split_idx is None or split_idx == 0:
        return None, None
    part1 = text[:tokens[split_idx].idx].strip()
    part2 = text[tokens[split_idx].idx:].strip()
    if len(part1.split()) < 2 or len(part2.split()) < 2:
        return None, None
    return part1, part2


def _split_into_clauses(text: str) -> List[str]:
    """Split text into individual sentences and clauses using spaCy."""
    if _nlp is None:
        import re
        parts = [p.strip() for p in re.split(r'[.!?]+', text) if p.strip()]
        return parts if parts else [text]

    doc = _nlp(text)
    clauses = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        p1, p2 = _spacy_split(sent_text)
        if p1 and p2:
            clauses.extend([p1, p2])
        else:
            clauses.append(sent_text)

    return clauses if clauses else [text]


def _adjective_aspects(text: str) -> List[dict]:
    """Fallback: extract adjectives from spaCy parse tree."""
    if _nlp is None:
        return []
    doc     = _nlp(text)
    aspects = []
    _NEG_ADV = {"too", "extremely"}

    for token in doc:
        if token.pos_ == "ADJ":
            child_intensifiers = [
                c.text.lower() for c in token.children
                if c.text.lower() in ("too", "very", "much")
            ]
            nearby_neg = False
            for offset in (1, 2):
                prev_i = token.i - offset
                if prev_i >= 0:
                    prev = doc[prev_i]
                    if prev.pos_ == "ADV" and prev.text.lower() in _NEG_ADV:
                        nearby_neg = True
                        break

            if nearby_neg or any(w in child_intensifiers for w in ("too", "much")):
                sentiment = "NEG"
            else:
                sentiment = "NEU"

            aspects.append({"aspect": token.lemma_, "sentiment": sentiment, "confidence": 0.6})
    return aspects


def _noun_fallback(text: str, action: str) -> List[dict]:
    """Last-resort: pull nouns as aspects."""
    if _nlp is None:
        return []
    doc     = _nlp(text)
    aspects = []
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            sentiment = "NEG" if action in ("REDUCE", "INCREASE") else "NEU"
            aspects.append({"aspect": token.lemma_, "sentiment": sentiment, "confidence": 0.5})
    return aspects[:3]


def _detect_causal(feedback_text: str) -> dict:
    """Detect causal relationships using spaCy dependency grammar."""
    if _nlp is None:
        return {"is_causal": False, "cause_ingredient": None, "working_text": feedback_text}

    doc = _nlp(feedback_text)
    for token in doc:
        if token.dep_ in _CAUSAL_DEPS:
            subtree_tokens = list(token.subtree)
            start = subtree_tokens[0].idx
            end   = subtree_tokens[-1].idx + len(subtree_tokens[-1].text)
            cause_text = doc.text[start:end]
            cause_ingredient = next(
                (t.text for t in token.subtree if t.pos_ in ("NOUN", "PROPN")), None
            )
            return {"is_causal": True, "cause_ingredient": cause_ingredient, "working_text": cause_text}

    return {"is_causal": False, "cause_ingredient": None, "working_text": feedback_text}


def _detect_intent(text: str) -> dict:
    """BART zero-shot: returns action REDUCE | INCREASE | KEEP."""
    lower_text = text.lower()
    
    # Direct rule pre-check for clear positive request phrases ("more", "add", "extra", "love to have more")
    if any(phrase in lower_text for phrase in ["love to have more", "want more", "add more", "more spicy", "extra spicy", "need more"]):
        return {"is_mixed": False, "action": "INCREASE", "confidence": 0.95}

    zs = _get_zero_shot()
    r1 = zs(text, candidate_labels=_MIXED_LABELS)
    top_label            = r1["labels"][0]
    is_mixed             = top_label == "this comment has both positive and negative parts"
    is_entirely_positive = top_label == "this comment is entirely expressing preference or satisfaction"

    r2      = zs(text, candidate_labels=_ACTION_LABELS)
    winning = r2["labels"][0]

    if winning == LABEL_REDUCE:
        action = "REDUCE"
    elif winning == LABEL_INCREASE:
        action = "INCREASE"
    else:
        action = "KEEP"

    confidence = r2["scores"][0]
    return {"is_mixed": is_mixed, "action": action, "confidence": confidence}


def _detect_intensity(text: str) -> dict:
    """VADER compound score → LOW / MODERATE / HIGH level."""
    compound = abs(_vader.polarity_scores(text)["compound"])
    if compound > 0.6:
        level = "HIGH"
    elif compound > 0.3:
        level = "MODERATE"
    else:
        level = "LOW"
    return {"level": level, "compound": compound}


def _delta(sentiment: str, action: str, intensity_level: str) -> float:
    """Simple explainable preference delta: LOW=0.2, MODERATE=0.4, HIGH=0.6."""
    strength = {"LOW": 0.2, "MODERATE": 0.4, "HIGH": 0.6}.get(intensity_level, 0.2)
    if action == "REDUCE":
        return -strength
    if action == "INCREASE":
        return +strength
    if sentiment == "POS":
        return +strength
    return 0.0


def _clip(score: float) -> float:
    return max(-1.0, min(1.0, score))


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def load_preferences() -> Dict[str, float]:
    """Return the current preference vector from disk."""
    if not os.path.exists(_PREF_FILE):
        return {}
    with open(_PREF_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_preferences(prefs: Dict[str, float]) -> None:
    """Persist preference vector to disk."""
    os.makedirs(os.path.dirname(_PREF_FILE), exist_ok=True)
    with open(_PREF_FILE, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)


EMA_ALPHA = 0.5

def reset_preferences() -> None:
    """Clear all preferences."""
    save_preferences({})


def analyze_feedback(
    recipe_name: str = "",
    feedback_text: str = "",
    star_rating: int = 3,
    consumed_percent: int = 50,
) -> dict:
    """
    Run the full NL-feedback pipeline and update user_preference.json.

    Score is determined ENTIRELY by the text — star_rating and consumed_percent
    are provided for compatibility.

    Returns:
        {
            "causal":               bool,
            "cause_ingredient":     str | None,
            "mixed":                bool,
            "aspects":              [(name, action, delta), ...],
            "aspect_details":      {ingredient: {"sentiment": str, "action": str}},
            "updated_preferences":  {ingredient: score},
        }
    """
    if not feedback_text and recipe_name:
        feedback_text = recipe_name

    # Step 1 — Causal detection
    causal = _detect_causal(feedback_text)
    working_text = causal["working_text"]

    # Step 2 & 3 — spaCy sentence/clause splitting before BERT ABSA and BART intent classification
    clauses = _split_into_clauses(working_text)

    results: List[Tuple[str, str, float]] = []
    aspect_details: Dict[str, dict] = {}
    is_mixed = False

    for clause in clauses:
        clause_causal = _detect_causal(clause)
        clause_intent = _detect_intent(clause)
        clause_intensity = _detect_intensity(clause)

        if clause_intent.get("is_mixed"):
            is_mixed = True

        raw_aspects = _run_bert(clause) or _adjective_aspects(clause)

        if not raw_aspects:
            if clause_causal["is_causal"] and clause_causal["cause_ingredient"]:
                raw_aspects = [{"aspect": clause_causal["cause_ingredient"], "sentiment": "NEG", "confidence": 0.5}]
            elif causal["is_causal"] and causal["cause_ingredient"] and causal["cause_ingredient"].lower() in clause.lower():
                raw_aspects = [{"aspect": causal["cause_ingredient"], "sentiment": "NEG", "confidence": 0.5}]
            else:
                raw_aspects = _noun_fallback(clause, clause_intent["action"])

        for aspect_info in raw_aspects:
            name = aspect_info["aspect"].lower()
            bert_sent = aspect_info["sentiment"]

            if clause_intent["action"] == "REDUCE":
                action = "REDUCE"
                sentiment = "NEGATIVE"
            elif clause_intent["action"] == "INCREASE":
                action = "INCREASE"
                sentiment = "POSITIVE"
            else:
                if bert_sent == "POS":
                    action = "KEEP"
                    sentiment = "POSITIVE"
                elif bert_sent == "NEG":
                    action = "REDUCE"
                    sentiment = "NEGATIVE"
                else:
                    action = "KEEP"
                    sentiment = "NEUTRAL"

            sent_code = "POS" if sentiment == "POSITIVE" else ("NEG" if sentiment == "NEGATIVE" else "NEU")
            final_delta = _delta(sent_code, action, clause_intensity["level"])

            results.append((name, action, final_delta))
            aspect_details[name] = {
                "sentiment": sentiment,
                "action": action,
            }

    # Step 6 — Update and save preferences.
    # - First mention of an ingredient: preference = delta directly (snap to that
    #   tier's full value, e.g. 0.20 for LOW).
    # - Repeat mention, SAME direction: grow by REPEAT_MENTION_STEP (0.10) from the
    #   current value each time (0.20 -> 0.30 -> 0.40 -> ...) -- unless the new
    #   mention's own tier value is a stronger (larger-magnitude) explicit statement
    #   than the grown value, in which case the explicit tier value wins outright
    #   (growth never overrides a genuinely stronger statement).
    # - Repeat mention that reverses direction: standard EMA blending. There's no
    #   separate stored "mention count" -- growth is always computed relative to the
    #   current score, so a reversal's EMA-blended result naturally becomes the new
    #   growth baseline (i.e. the "count" resets itself for free).
    REPEAT_MENTION_STEP = 0.10
    prefs = load_preferences()
    for name, action, delta in results:
        if name in prefs:
            old = prefs[name]
            same_direction = (old == 0.0) or (delta == 0.0) or ((delta > 0) == (old > 0))
            if same_direction:
                grown = old + (REPEAT_MENTION_STEP if delta >= 0 else -REPEAT_MENTION_STEP)
                new_val = grown if abs(grown) >= abs(delta) else delta
            else:
                new_val = (EMA_ALPHA * delta) + ((1.0 - EMA_ALPHA) * old)
        else:
            new_val = delta
        prefs[name] = _clip(new_val)
    save_preferences(prefs)

    logger.info(
        "[feedback_service] Processed feedback: %d aspects updated.",
        len(results)
    )

    return {
        "causal":              causal["is_causal"],
        "cause_ingredient":    causal["cause_ingredient"],
        "mixed":               is_mixed or len(clauses) > 1,
        "aspects":             results,       # List[(name, action, delta)]
        "aspect_details":      aspect_details,
        "updated_preferences": prefs,
    }
