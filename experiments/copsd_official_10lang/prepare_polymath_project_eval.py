"""Freeze the local PolyMath four-split test set for the ten-language project metric."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/data0/users/lixiaoyang")
SOURCE = ROOT / "data/math_single_10lang_20261015/polymath"
OUTPUT = ROOT / "data/copsd_official_20261015/polymath_project_eval_v1"
LANGUAGES = "sw te th bn zh ja ru de fr es".split()
SPLITS = "low medium high top".split()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    manifest = {"source": str(SOURCE), "readme_sha256": sha256(SOURCE / "README.md"),
                "splits": SPLITS, "languages": LANGUAGES, "files": {}}
    english = {split: pq.read_table(SOURCE / "en" / f"{split}.parquet").to_pylist()
               for split in SPLITS}
    for lang in LANGUAGES:
        rows = []
        sources = {}
        for split in SPLITS:
            source = SOURCE / lang / f"{split}.parquet"
            data = pq.read_table(source).to_pylist()
            if len(data) != 125 or len(english[split]) != 125:
                raise ValueError(f"Expected 125 aligned rows: {lang}/{split}")
            for i, (item, en) in enumerate(zip(data, english[split])):
                if item["id"] != f"{split}-{lang}-{i}" or en["id"] != f"{split}-en-{i}":
                    raise ValueError(f"Unexpected ID: {lang}/{split}/{i}")
                if str(item["answer"]) != str(en["answer"]):
                    raise ValueError(f"Answer mismatch: {lang}/{split}/{i}")
                if not item["question"].strip():
                    raise ValueError(f"Empty question: {lang}/{split}/{i}")
                rows.append({"id": item["id"], "split": split,
                             "question": item["question"], "answer": item["answer"]})
            sources[split] = {"path": str(source), "sha256": sha256(source), "count": len(data)}
        target = OUTPUT / f"polymath_all500_{lang}.json"
        target.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        manifest["files"][lang] = {"path": str(target), "sha256": sha256(target),
                                   "count": len(rows), "sources": sources}
    (OUTPUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(OUTPUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
