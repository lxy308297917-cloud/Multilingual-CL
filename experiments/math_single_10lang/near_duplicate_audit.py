"""Surface likely paraphrases for human review; never auto-clear contamination."""
import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from common import normalize_question


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    train, test = read_jsonl(a.inventory), read_jsonl(a.benchmark)
    v = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True, dtype="float32")
    x = v.fit_transform([normalize_question(r["question_en"]).lower() for r in train + test])
    sim = x[:len(train)] @ x[len(train):].T
    candidates = []
    for i in range(len(train)):
        row = sim.getrow(i)
        ranked = sorted(zip(row.indices, row.data), key=lambda pair: -pair[1])[:3]
        for j, score in ranked:
            if score >= 0.30:
                candidates.append({"problem_id": train[i]["problem_id"], "dataset": test[j]["dataset"], "test_id": test[j]["id"], "cosine_char_3_5": round(float(score), 4), "train_question": train[i]["question_en"], "test_question": test[j]["question_en"]})
    candidates.sort(key=lambda r: -r["cosine_char_3_5"])
    Path(a.output).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in candidates))
    print(json.dumps({"candidate_pairs_ge_0.30": len(candidates), "pairs_ge_0.70": sum(r["cosine_char_3_5"] >= .70 for r in candidates), "max_similarity": candidates[0]["cosine_char_3_5"] if candidates else 0}))


if __name__ == "__main__":
    main()
