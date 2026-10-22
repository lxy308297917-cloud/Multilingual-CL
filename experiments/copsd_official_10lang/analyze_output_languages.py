"""CPU GlotLID supplement for saved outputs; language ID does not measure fluency."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prose(text):
    # Same math stripping convention as the existing project detector screen.
    text = re.sub(r"\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^$]*\$", " ", text)
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    text = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=Path, nargs="+", required=True)
    p.add_argument("--model", type=Path, default=Path("/data0/users/lixiaoyang/models/glotlid/model_v3.bin"))
    p.add_argument("--target-label", default="__label__swh_Latn")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    import fasttext
    model = fasttext.load_model(str(a.model))
    if a.target_label not in model.get_labels():
        raise ValueError("Target label is not present in this GlotLID model: " + a.target_label)

    def identify(text):
        text = prose(text)
        if sum(c.isalpha() for c in text) < 20:
            return {"status": "too_short", "prose_chars": len(text)}
        labels, probs = model.predict(text, k=1)
        return {"status": "predicted", "label": labels[0], "prob": float(probs[0]),
                "prose_chars": len(text)}

    result = {"scope": "project language-ID supplement, not calibrated fluency or reasoning quality",
              "model": str(a.model), "model_sha256": digest(a.model),
              "target_label": a.target_label, "runs": {}}
    for path in a.items:
        with path.open(encoding='utf-8') as stream:
            rows = [json.loads(s) for s in stream if s.strip()]
        predictions = []
        for row in rows:
            sample = row["samples"][0]
            text = sample["raw_output"]
            thinking, separator, final = text.partition("</think>")
            predictions.append({"id": row["id"], "correct": sample["correct"],
                "thinking": identify(thinking),
                "final": identify(final) if separator else {"status": "no_final_section"}})
        summary = {}
        for part in ["thinking", "final"]:
            subset = [r[part] for r in predictions]
            summary[part] = {"total_items": len(rows),
                "statuses": dict(Counter(r["status"] for r in subset)),
                "top1_label_counts": dict(Counter(r["label"] for r in subset if r["status"] == "predicted")),
                "target_top1_probability_ge_0_8": sum(
                    r.get("label") == a.target_label and r.get("prob", 0) >= 0.8 for r in subset),
                "correct_items_label_counts": dict(Counter(
                    r[part].get("label", r[part]["status"]) for r in predictions if r["correct"]))}
        result["runs"][path.parent.name] = {
            "items": str(path), "items_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "summary": summary, "predictions": predictions}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v["summary"] for k, v in result["runs"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
