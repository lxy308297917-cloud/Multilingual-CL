"""Freeze the official GSM8K test questions for the project English retention metric."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


SOURCE_COMMIT = "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc"
SOURCE_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--mgsm-en", type=Path, required=True)
    a = p.parse_args()
    raw = a.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official GSM8K source identity mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 1319:
        raise ValueError("Expected all 1319 GSM8K test questions")
    items = []
    for i, row in enumerate(rows):
        if not row["question"].strip() or "####" not in row["answer"]:
            raise ValueError("Empty question or missing gold delimiter")
        answer = row["answer"].rsplit("####", 1)[1].strip()
        if not re.fullmatch(r"-?[\d,]+(?:\.\d+)?", answer):
            raise ValueError("Non-numeric gold")
        items.append({"id": f"gsm8k-test-{i:04d}", "question": row["question"], "answer": answer})
    test = {norm(row["question"]) for row in items}
    overlaps = {}
    for name, path in [("train_2900", a.train), ("dev_100", a.dev)]:
        overlaps[name] = [r["problem_id"] for r in json.loads(path.read_text()) if norm(r["problem"]) in test]
    mgsm = [line.split("\t", 1)[0] for line in a.mgsm_en.read_text().splitlines()]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    output = a.output_dir / "gsm8k_en_test_1319.json"
    output.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "benchmark": "GSM8K main test", "count": len(items), "source_git_revision": SOURCE_COMMIT,
        "canonical_source_sha256": SOURCE_SHA256,
        "canonical_source_url": f"https://raw.githubusercontent.com/openai/grade-school-math/{SOURCE_COMMIT}/grade_school_math/data/test.jsonl",
        "output": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "normalized_exact_train_dev_overlap": overlaps,
        "normalization": "NFKC, whitespace collapse, casefold; not fuzzy or semantic deduplication",
        "mgsm_rev2_en_questions_also_in_gsm8k": sum(norm(q) in test for q in mgsm),
        "mgsm_rev2_en_count": len(mgsm),
        "project_protocol": {"few_shot": 0, "max_new_tokens": 2048,
            "prompt": "COPSD EN math instruction and thinking prefix; no gold/reference in model input",
            "scoring": "single sampled answer, existing boxed extraction and math_verify",
            "identity": "project English retention; distinct from previous SW D2 5-shot and official COPSD reported benchmarks"},
    }
    (a.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if any(overlaps.values()):
        raise ValueError("Train/dev overlap found; retain evidence and resolve before formal evaluation")
    print(json.dumps({"items": len(items), "train_dev_overlap": overlaps,
                      "mgsm_en_overlap": manifest["mgsm_rev2_en_questions_also_in_gsm8k"]}))


if __name__ == "__main__":
    main()
