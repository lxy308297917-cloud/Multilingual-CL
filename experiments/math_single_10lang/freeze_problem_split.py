"""Conservative fixed-ID split; removes every flagged near-match before sampling."""
import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer
from common import CONFIG, dump_json, messages


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--near-duplicates", required=True)
    p.add_argument("--problem-exclusions", type=Path, help="Additional shared gold-semantic quarantine IDs")
    p.add_argument("--internal-near-duplicates", type=Path, help="Likely same-source paraphrase IDs to quarantine across train/dev")
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    audit = json.loads(Path(a.audit).read_text())
    if not audit["benchmark_audit_complete"] or audit["inconsistent_reference_count"] or any(x["issue_count"] for x in audit["languages"].values()):
        raise ValueError("Required schema/alignment/exact contamination audit has unresolved issues")
    inventory = lines(a.inventory)
    near = lines(a.near_duplicates)
    near_ids = {x["problem_id"] for x in near}
    known_ids = {x["problem_id"] for x in inventory}
    if near_ids - known_ids:
        raise ValueError("Near-match exclusion list contains unknown problem IDs")
    extra_ids = set(json.loads(a.problem_exclusions.read_text())) if a.problem_exclusions else set()
    if extra_ids - known_ids:
        raise ValueError("Gold-semantic exclusion list contains unknown problem IDs")
    internal_ids = set(json.loads(a.internal_near_duplicates.read_text())) if a.internal_near_duplicates else set()
    if internal_ids - known_ids:
        raise ValueError("Internal near-duplicate list contains unknown problem IDs")
    exclude = near_ids | extra_ids | internal_ids
    clean = [x for x in inventory if x["problem_id"] not in exclude]
    tokenizer = AutoTokenizer.from_pretrained(str(a.tokenizer), local_files_only=True)
    length_excluded = []
    length_clean = []
    for row in clean:
        counts = {lang: len(tokenizer.apply_chat_template(messages(row["languages"][lang]["question_target"], lang), tokenize=True, add_generation_prompt=True, enable_thinking=True)) for lang in CONFIG["languages"]}
        if max(counts.values()) > a.max_prompt_tokens:
            length_excluded.append({"problem_id": row["problem_id"], "prompt_tokens": counts})
        else:
            length_clean.append(row)
    clean = length_clean
    rng = random.Random(CONFIG["seed"])
    # Only a length proxy is available. It is not a verified difficulty label.
    clean.sort(key=lambda x: len(x["question_en"]))
    bins = [clean[i::5] for i in range(5)]
    for b in bins:
        rng.shuffle(b)
    shuffled = [b[i] for i in range(max(map(len, bins))) for b in bins if i < len(b)]
    seen = set()
    ordered = []
    for x in shuffled:
        if x["problem_id"] not in seen:
            ordered.append(x)
            seen.add(x["problem_id"])
    sizes = [("train", CONFIG["train_size"]), ("dev_tune", CONFIG["dev_tune_size"]), ("dev_select", CONFIG["dev_select_size"])]
    if len(ordered) < sum(n for _, n in sizes):
        raise ValueError(f"Only {len(ordered)} clean IDs; adjust all-language shared sizes explicitly")
    splits = {}
    cursor = 0
    for name, n in sizes:
        splits[name] = [x["problem_id"] for x in ordered[cursor:cursor + n]]
        cursor += n
    splits["reserve"] = [x["problem_id"] for x in ordered[cursor:]]
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(a.output_dir / "problem_ids.json", splits)
    dump_json(a.output_dir / "split_manifest.json", {"seed": CONFIG["seed"], "source_revision": CONFIG["train_dataset_revision"], "benchmark_sources": {"MGSM-Rev2": "5f091edeae66ebc605ac4f98dca07a6c345f0060", "MGSM-original": "452a21ad3dae5668c06ceeac21ff073e1e40f9be", "PolyMath": "71e0db902e0ece05e208dfd8b1695bd3d95cf130", "GSM8K-test": "740312add88f781978c0658806c59bc2815b9866"}, "near_match_policy": "exclude all benchmark char-TFIDF 3-5 gram cosine >= 0.30 candidate IDs; exclude all within-pool char-TFIDF cosine >= 0.8 same-gold candidate pairs", "excluded_near_match_problem_ids": sorted(near_ids), "excluded_gold_semantic_problem_ids": sorted(extra_ids), "excluded_internal_near_duplicate_problem_ids": sorted(internal_ids), "combined_excluded_problem_ids": sorted(exclude), "max_student_prompt_tokens": a.max_prompt_tokens, "length_excluded": length_excluded, "counts": {k: len(v) for k, v in splits.items()}, "audit_limit": "Candidate retrieval is lexical; semantic paraphrases below threshold may remain. Gold-semantic quarantine is conservative, not proof that remaining labels are correct. Target answer full-sequence lengths are unknown until labels are generated."})
    print(json.dumps({"near_match_excluded": len(near_ids), "gold_semantic_excluded": len(extra_ids), "internal_near_excluded": len(internal_ids), "combined_excluded": len(exclude), "length_excluded": len(length_excluded), "counts": {k: len(v) for k, v in splits.items()}}))


if __name__ == "__main__":
    main()
