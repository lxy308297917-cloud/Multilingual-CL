"""Prepare disjoint original-question pools without selecting languages/training.

Project extension, not an official COPSD split. Exact IDs and file fingerprints
are written only to the requested local output. Public report contains counts.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

LANGUAGES = ("sw", "te", "th", "bn", "zh", "ja", "ru", "de", "fr", "es")
ENGLISH_FIELDS = ("source", "problem", "solution", "Answer")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def quartiles(rows, field):
    ordered = sorted(rows, key=lambda row: (len(row[field]), row["problem_id"]))
    return {row["problem_id"]: min(3, i * 4 // len(ordered)) for i, row in enumerate(ordered)}


def build(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference = None
    input_files = {}
    dev_ids = None
    for language in LANGUAGES:
        path = args.input_dir / f"project_train_2900_{language}.json"
        rows = read(path)
        ids = [row["problem_id"] for row in rows]
        assert len(ids) == 2900 and len(set(ids)) == 2900
        assert all(row.get(f"problem_{language}") and row.get(f"problem_{language}_ok") is True for row in rows)
        current_dev = {row["problem_id"] for row in read(args.input_dir / f"project_dev_100_{language}.json")}
        assert len(current_dev) == 100 and not set(ids) & current_dev
        if reference is None:
            reference = rows
            dev_ids = current_dev
        else:
            assert ids == [row["problem_id"] for row in reference], language
            assert current_dev == dev_ids, language
            assert all(all(row[key] == base[key] for key in ENGLISH_FIELDS) for row, base in zip(rows, reference)), language
        input_files[language] = {"path": str(path.resolve()), "sha256": sha(path)}

    qbin, sbin = quartiles(reference, "problem"), quartiles(reference, "solution")
    groups = defaultdict(list)
    for row in reference:
        identifier = row["problem_id"]
        groups[(row["source"], qbin[identifier], sbin[identifier])].append(identifier)
    rng = random.Random(args.seed)
    pools = [[] for _ in range(3)]
    stratum_counts = []
    for key in sorted(groups):
        group = groups[key].copy()
        rng.shuffle(group)
        local = [0, 0, 0]
        for identifier in group:
            pool = min(range(3), key=lambda p: (local[p], len(pools[p]), p))
            pools[pool].append(identifier)
            local[pool] += 1
        assert max(local) - min(local) <= 1
        stratum_counts.append(local)
    order = {row["problem_id"]: i for i, row in enumerate(reference)}
    pools = [sorted(pool, key=order.__getitem__) for pool in pools]
    all_ids = set().union(*(set(pool) for pool in pools))
    assert len(all_ids) == 2900 and sum(map(len, pools)) == 2900
    assert not all_ids & dev_ids
    assert max(map(len, pools)) - min(map(len, pools)) <= 1
    for i in range(3):
        for j in range(i):
            assert not set(pools[i]) & set(pools[j])

    report = {
        "protocol": "Project sequential-pool extension; no training or language selection",
        "languages_checked": list(LANGUAGES), "seed": args.seed,
        "pool_sizes": list(map(len, pools)), "union_size": len(all_ids),
        "pairwise_overlap": 0, "development_overlap": 0,
        "english_fields_equal_across_languages": True,
        "source_counts": dict(Counter(row["source"] for row in reference)),
        "stratification": "source x English question character-length quartile x English reference character-length quartile",
        "difficulty_limit": "Released inputs have no difficulty labels. Length strata are proxies, not verified math difficulty or token budgets.",
        "stratum_pool_count_max_difference": max(max(v) - min(v) for v in stratum_counts),
        "binding_rule": "After independent-dev language selection, bind L1 to pool0, L2 to pool1, L3 to pool2. Reverse order reverses language/pool pairs together; it does not reassign pools by stage number.",
        "matched_single_rule": "For each chosen language use the identical bound pool and stage exposure budget from Base. Existing2900-question singles are not the matched small-pool control.",
        "exposure_note": "Pools have966/967 unique questions. A nominal3200-exposure stage repeats questions; actual exposure counts depend on epoch-end partial accumulation and must be logged.",
        "training_started": False, "languages_selected": False,
    }
    save(args.output_dir / "POOLS.PRIVATE.json", {
        "seed": args.seed, "inputs": input_files,
        "pool_ids": pools, "development_ids": sorted(dev_ids), "report": report,
    })
    save(args.output_dir / "REPORT.PUBLIC.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def materialize(args):
    manifest = read(args.manifest)
    item = manifest["inputs"][args.language]
    assert sha(item["path"]) == item["sha256"], "Original input changed"
    ids = manifest["pool_ids"][args.pool]
    lookup = {row["problem_id"]: row for row in read(item["path"])}
    assert len(set(ids)) == len(ids) and not set(ids) & set(manifest["development_ids"])
    rows = [lookup[identifier] for identifier in ids]
    assert all(row.get(f"problem_{args.language}") for row in rows)
    save(args.output, rows)
    print(json.dumps({"language": args.language, "pool": args.pool, "records": len(rows),
                      "fields_preserved": True, "training_started": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--input-dir", type=Path, required=True)
    builder.add_argument("--output-dir", type=Path, required=True)
    builder.add_argument("--seed", type=int, default=20261019)
    exporter = commands.add_parser("materialize")
    exporter.add_argument("--manifest", type=Path, required=True)
    exporter.add_argument("--language", choices=LANGUAGES, required=True)
    exporter.add_argument("--pool", type=int, choices=range(3), required=True)
    exporter.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()
