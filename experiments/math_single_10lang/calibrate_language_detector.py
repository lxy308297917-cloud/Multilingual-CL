"""Screen candidate language detectors on published translated questions."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from common import CONFIG, dump_json
from eval_math import language_status


GLOTLID_LABELS = {"sw": "__label__swh_Latn", "te": "__label__tel_Telu", "bn": "__label__ben_Beng", "zh": "__label__cmn_Hani", "ja": "__label__jpn_Jpan", "th": "__label__tha_Thai", "fr": "__label__fra_Latn", "de": "__label__deu_Latn", "es": "__label__spa_Latn", "ru": "__label__rus_Cyrl"}


def strip_math(text):
    text = re.sub(r"\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^$]*\$", " ", text)
    text = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def glotlid_status(text, language, model):
    prose = strip_math(text)
    if len(prose) < 20:
        return {"status": "unknown", "reason": "too_short"}
    labels, probabilities = model.predict(prose, k=1)
    label, probability = labels[0], float(probabilities[0])
    return {"status": "candidate_pass" if label == GLOTLID_LABELS[language] and probability >= .8 else "candidate_fail", "best": label, "prob": probability, "calibrated": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-language", type=int, default=100)
    p.add_argument("--backend", choices=("langdetect", "glotlid"), default="langdetect")
    p.add_argument("--glotlid-model", type=Path)
    p.add_argument("--splits", type=Path, help="Limit the weak-positive screen to the frozen train/dev pool")
    a = p.parse_args()
    rows = [json.loads(x) for x in a.inventory.read_text().splitlines() if x]
    if a.splits:
        split = json.loads(a.splits.read_text())
        ids = set(split["train"] + split["dev_tune"] + split["dev_select"])
        rows = [row for row in rows if row["problem_id"] in ids]
    rng = random.Random(42)
    chosen = rng.sample(rows, min(len(rows), a.per_language))
    model = None
    if a.backend == "glotlid":
        if not a.glotlid_model:
            raise ValueError("--glotlid-model is required for GlotLID screening")
        import fasttext
        model = fasttext.load_model(str(a.glotlid_model))
    report = {}
    for lang in CONFIG["languages"]:
        if model:
            results = [glotlid_status(row["languages"][lang]["question_target"], lang, model) for row in chosen]
        else:
            results = [language_status(row["languages"][lang]["question_target"], lang) for row in chosen]
        report[lang] = {"count": len(results), "statuses": dict(Counter(x["status"] for x in results)), "predicted_best": dict(Counter(x.get("best", "unknown") for x in results)), "interpretation": "translated-question screening only; math outputs still require separate calibration"}
    dump_json(a.output, {"backend": a.backend, "model": str(a.glotlid_model) if model else None, "model_sha256": hashlib.sha256(a.glotlid_model.read_bytes()).hexdigest() if model else None, "sample_seed": 42, "source_pool_count": len(rows), "per_language_count": len(chosen), "target_language_positive_examples_only": True, "interpretation": "Not a calibrated math-output language pass; code-mixed or unnatural prose can still score candidate_pass", "languages": report})
    print(json.dumps({lang: r["statuses"] for lang,r in report.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
