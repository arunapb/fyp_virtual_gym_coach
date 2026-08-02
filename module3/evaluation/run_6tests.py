"""
6-test runner for semantic-split feedback_pipeline.
Deletes user_preference.json before starting.
Prints rich formatted output matching the requested layout.
"""
import json
import os
import sys

_EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
_COMBINE_DIR = os.path.dirname(_EVAL_DIR)
_BACKEND_DIR = os.path.join(_COMBINE_DIR, "backend")
_ML_DIR      = os.path.join(_COMBINE_DIR, "ml_models")

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_PREF_FILE = os.path.join(_BACKEND_DIR, "data", "user_preference.json")

# delete old preference file
if os.path.exists(_PREF_FILE):
    os.remove(_PREF_FILE)
    print("Deleted user_preference.json\n")

# ── imports that trigger model loading ─────────────────────────────────
import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from services.causal_detector import detect_causal
from services.intent_detector import detect_intent
from services.intensity_detector import detect_intensity

# ── load BERT ──────────────────────────────────────────────────────────
_MODEL_DIR = os.path.join(_ML_DIR, "my_absa_model")
with open(os.path.join(_MODEL_DIR, "label_map.json")) as _f:
    _label_data = json.load(_f)
_id2label = {int(k): v for k, v in _label_data["id2label"].items()}
tokenizer  = BertTokenizerFast.from_pretrained(_MODEL_DIR)
bert_model = BertForTokenClassification.from_pretrained(_MODEL_DIR)
bert_model.eval()

# ── pre-load sentence transformer once ────────────────────────────────
_st_model = SentenceTransformer("all-MiniLM-L6-v2")

SEP = "=" * 54


def run_bert(text):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    word_ids = encoding.word_ids()
    tokens   = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    with torch.no_grad():
        outputs = bert_model(**encoding)
    probs    = torch.softmax(outputs.logits[0], dim=-1)
    pred_ids = torch.argmax(probs, dim=-1).tolist()

    word_info = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in word_info:
            word_info[wid] = {
                "label": _id2label[pred_ids[i]],
                "conf":  probs[i][pred_ids[i]].item(),
                "subtokens": [tokens[i]],
            }
        else:
            word_info[wid]["subtokens"].append(tokens[i])

    def _reconstruct(subtokens):
        out = ""
        for t in subtokens:
            out += t[2:] if t.startswith("##") else t
        return out

    aspects = []
    cur_words, cur_sent, cur_confs = [], None, []
    for wid in sorted(word_info):
        info  = word_info[wid]
        label, conf = info["label"], info["conf"]
        word  = _reconstruct(info["subtokens"])
        if label.startswith("B-"):
            if cur_words:
                aspects.append({"aspect": " ".join(cur_words),
                                 "sentiment": cur_sent,
                                 "confidence": sum(cur_confs)/len(cur_confs)})
            cur_words, cur_sent, cur_confs = [word], label[2:], [conf]
        elif label.startswith("I-") and cur_words:
            cur_words.append(word)
            cur_confs.append(conf)
        else:
            if cur_words:
                aspects.append({"aspect": " ".join(cur_words),
                                 "sentiment": cur_sent,
                                 "confidence": sum(cur_confs)/len(cur_confs)})
            cur_words, cur_sent, cur_confs = [], None, []
    if cur_words:
        aspects.append({"aspect": " ".join(cur_words),
                         "sentiment": cur_sent,
                         "confidence": sum(cur_confs)/len(cur_confs)})
    return aspects


def _find_semantic_split(text):
    words = text.split()
    if len(words) < 4:
        return None, None, None

    best_split = None
    lowest_sim = 1.0
    split_sims = {}

    for i in range(2, len(words) - 1):
        part1 = " ".join(words[:i])
        part2 = " ".join(words[i:])
        v1 = _st_model.encode([part1])
        v2 = _st_model.encode([part2])
        sim = float(cosine_similarity(v1, v2)[0][0])
        split_sims[i] = sim
        if sim < lowest_sim:
            lowest_sim = sim
            best_split = i

    if lowest_sim < 0.5 and best_split is not None:
        p1 = " ".join(words[:best_split])
        p2 = " ".join(words[best_split:])
        return p1, p2, lowest_sim

    return None, None, lowest_sim


def _load_prefs():
    if os.path.exists(_PREF_FILE):
        with open(_PREF_FILE) as f:
            return json.load(f)
    return {}


def _save_prefs(prefs):
    with open(_PREF_FILE, "w") as f:
        json.dump(prefs, f, indent=2)


TESTS = [
    ("Garlic Fish",    "fish was great but too much garlic",     3,  70),
    ("Salmon Dish",    "loved the salmon the garlic ruined it",  2,  40),
    ("Chicken Dish",   "chicken was perfect sauce was terrible", 3,  65),
    ("Garlic Chicken", "too much garlic",                        2,  40),
    ("Spiced Dish",    "spices were absolutely amazing",         5, 100),
    ("Chicken Sauce",  "chicken was dry because of less sauce",  2,  50),
]

for t_num, (recipe, feedback_text, star_rating, consumed_percent) in enumerate(TESTS, 1):
    print(f"\n{'='*54}")
    print(f"TEST {t_num}")
    print(f"{'='*54}")
    print(f"Feedback:  {feedback_text}")
    print(f"Rating:    {star_rating}/5   Consumed: {consumed_percent}%")
    print()

    causal       = detect_causal(feedback_text)
    working_text = causal["working_text"]
    causal_str   = f"YES [{causal['cause_ingredient']}]" if causal["is_causal"] else "NO"
    print(f"Causal:    {causal_str}")

    intent = detect_intent(working_text)
    print(f"Mixed:     {'YES' if intent['is_mixed'] else 'NO'}")
    print()

    # semantic split (always attempted, no keyword gate)
    p1, p2, lowest_sim = _find_semantic_split(working_text)

    print("Semantic split:")
    if p1 is not None:
        print(f"  Split found: YES")
        print(f"  Part 1: {p1} (weight 0.3)")
        print(f"  Part 2: {p2} (weight 0.7)")
        print(f"  Similarity at split: {lowest_sim:.2f}")
    else:
        print(f"  Split found: NO")
        if lowest_sim is not None:
            print(f"  Similarity at best candidate split: {lowest_sim:.2f} (threshold 0.50)")
        else:
            print(f"  Text too short (<4 words) to attempt split")
    print()

    if p1 is not None:
        p1_aspects = run_bert(p1) if p1 else []
        p2_aspects = run_bert(p2)
        aspects_with_weight = [(a, 0.3) for a in p1_aspects] + [(a, 0.7) for a in p2_aspects]
    else:
        raw = run_bert(working_text)
        aspects_with_weight = [(a, 1.0) for a in raw]

    all_aspects = [a for a, _ in aspects_with_weight]

    intensity = detect_intensity(working_text)

    sent_map = {"POS": "POSITIVE", "NEG": "NEGATIVE", "NEU": "NEUTRAL"}
    print("BERT found:")
    if all_aspects:
        for a in all_aspects:
            print(f"  {a['aspect']} -> {sent_map.get(a['sentiment'], a['sentiment'])} {a['confidence']*100:.0f}%")
    else:
        print("  (no aspects detected)")
    print()
    print("Zero Shot:")
    print(f"  Action: {intent['action']}")
    print()
    print(f"Intensity: {intensity['level']} [{intensity['compound']:.2f}]")
    print()

    overall_score = (star_rating / 5 * 0.6) + (consumed_percent / 100 * 0.4)
    prefs = _load_prefs()

    results = []
    for aspect_info, mix_w in aspects_with_weight:
        sentiment = aspect_info["sentiment"]
        bert_conf = aspect_info["confidence"]
        name      = aspect_info["aspect"].lower()

        if sentiment == "POS":
            action, base_delta = "KEEP", +1.0
        elif sentiment == "NEG":
            action = intent["action"]
            base_delta = (-1.0 if action == "REDUCE" else (+0.5 if action == "INCREASE" else 0.0))
        else:
            action, base_delta = "KEEP", 0.0

        final_delta = base_delta * bert_conf * overall_score * intensity["multiplier"] * mix_w
        results.append((name, action, final_delta))

    print("Final:")
    pref_updates = []
    for name, action, delta in results:
        old = prefs.get(name, 0.0)
        new = (0.3 * delta) + (0.7 * old)
        prefs[name] = new
        print(f"  {name} | {action} | delta: {delta:+.4f}")
        pref_updates.append((name, old, new))

    print()
    print("Preference updated:")
    for name, old, new in pref_updates:
        print(f"  {name}: {old:.4f} -> {new:.4f}")

    _save_prefs(prefs)

# ── final preference vector ────────────────────────────────────────────
final_prefs = _load_prefs()
print(f"\n{'='*54}")
print("FINAL PREFERENCE VECTOR")
print(f"{'='*54}")
for k, v in sorted(final_prefs.items()):
    bar = ("+" if v >= 0 else "-") * max(1, int(abs(v) * 20))
    print(f"  {k:<20s}: {v:+.4f}  {bar}")
print(f"{'='*54}")

print("\n-- Approach comparison ------------------------------------------")
print("Old approach used keywords: but, however, although, though, yet, while")
print("New approach uses semantic similarity")
print("No keyword list anywhere in code")
