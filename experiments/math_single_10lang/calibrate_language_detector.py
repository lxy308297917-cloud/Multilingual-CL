"""Screen langdetect on source translated questions; this is not final human calibration."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import CONFIG, dump_json
from eval_math import language_status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-language", type=int, default=100)
    a = p.parse_args()
    rows = [json.loads(x) for x in a.inventory.read_text().splitlines() if x]
    rng = random.Random(42)
    chosen = rng.sample(rows, min(len(rows), a.per_language))
    report = {}
    for lang in CONFIG["languages"]:
        results = [language_status(row["languages"][lang]["question_target"], lang) for row in chosen]
        report[lang] = {"count": len(results), "statuses": dict(Counter(x["status"] for x in results)), "predicted_best": dict(Counter(x.get("best", "unknown") for x in results)), "interpretation": "translated-question screening only; math outputs still require separate calibration"}
    dump_json(a.output, report)
    print(json.dumps({lang: r["statuses"] for lang,r in report.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
