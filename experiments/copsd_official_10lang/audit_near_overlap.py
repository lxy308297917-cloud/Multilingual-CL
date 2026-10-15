"""Screen official English training questions against locked benchmark questions.

Character n-gram cosine is a triage heuristic; every flagged pair requires
manual interpretation before any exclusion or no-contamination claim.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-json", type=Path, required=True)
    p.add_argument("--benchmark-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    train = json.loads(a.training_json.read_text())
    test = [json.loads(line) for line in a.benchmark_jsonl.read_text().splitlines()]
    questions = [r["problem"] for r in train] + [r["question_en"] for r in test]
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), dtype=np.float32).fit_transform(questions)
    similarities = (tfidf[:len(train)] @ tfidf[len(train):].T).tocsr()
    flagged = []
    maxima = []
    for i, row in enumerate(train):
        sims = similarities.getrow(i)
        if sims.nnz:
            j = int(sims.indices[int(sims.data.argmax())])
            score = float(sims.data.max())
        else:
            j, score = -1, 0.0
        maxima.append(score)
        if score >= 0.7:
            b = test[j]
            flagged.append({"problem_id": row["problem_id"], "dataset": b["dataset"],
                            "test_id": b["id"], "cosine": round(score, 5),
                            "train_question": row["problem"], "test_question": b["question_en"]})
    result = {"method": "English character TF-IDF cosine, 3–5 grams; heuristic triage",
              "train_count": len(train), "benchmark_count": len(test),
              "max_score": max(maxima), "pairs_ge_0_7": len(flagged),
              "count_ge_0_8": sum(x >= .8 for x in maxima),
              "count_ge_0_9": sum(x >= .9 for x in maxima),
              "flagged": sorted(flagged, key=lambda x: -x["cosine"])}
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "flagged"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
