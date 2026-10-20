"""Create a separate, evidence-bound JA dataset; never edit the official input.

Only the already audited runaway prefix is removed. No translation or answer is
generated. All source IDs, hashes and text remain in the private local manifest.
"""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--official-parquet', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    args = p.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    assert not args.output.exists(), 'Never overwrite a prepared dataset'
    rows = json.loads(args.input.read_text())
    evidence = json.loads(args.evidence.read_text())['ja_source_evidence']
    assert len(rows) == 2900 and len(evidence) == 1
    e = evidence[0]
    i = e['row_index']
    before = rows[i]['problem_ja']
    assert rows[i]['problem_id'] == e['problem_id']
    assert digest(before) == e['raw_sha256']
    matches = [r for r in pq.read_table(args.official_parquet).to_pylist()
               if r['problem'] == rows[i]['problem'] and r['solution'] == rows[i]['solution']]
    assert len(matches) == 1 and matches[0]['problem_translated'] == before
    match = re.match(r'^(?:ーハン)+ー?', before)
    assert match and match.end() == e['removed_prefix_chars'] > 60000
    after = before[match.end():]
    assert digest(after) == e['retained_sha256'] and len(after) == 154
    fixed = copy.deepcopy(rows)
    fixed[i]['problem_ja'] = after
    assert [r['problem_id'] for r in rows] == [r['problem_id'] for r in fixed]
    assert [n for n, (a, b) in enumerate(zip(rows, fixed)) if a != b] == [i]
    assert {k: v for k, v in fixed[i].items() if k != 'problem_ja'} == {
        k: v for k, v in rows[i].items() if k != 'problem_ja'}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    lengths = [len(x) for x in tokenizer([r['problem_ja'] for r in fixed],
                                       add_special_tokens=False)['input_ids']]
    assert max(lengths) < 17000
    manifest = {'protocol': 'project_ja_literal_prefix_repair_v1', 'rows': len(rows),
                'modified_rows': 1, 'modified_field_only': 'problem_ja',
                'ids_order_english_reference_and_answers_unchanged': True,
                'original_dataset_sha256': digest(args.input.read_text()),
                'evidence': e, 'max_repaired_raw_question_tokens': max(lengths),
                'scope': 'Project adaptation; preserve original official data and failed runs. '
                         'Literal prefix removal only; no generated target-language labels.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(fixed, ensure_ascii=False, indent=2) + '\n'
    manifest['repaired_dataset_sha256'] = digest(text)
    args.output.write_text(text)
    args.output.with_suffix('.PRIVATE.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['protocol', 'rows', 'modified_rows',
          'ids_order_english_reference_and_answers_unchanged', 'max_repaired_raw_question_tokens']}))


if __name__ == '__main__':
    main()
