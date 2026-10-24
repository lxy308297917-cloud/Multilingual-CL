"""CPU-only percentage-unit diagnostic from unchanged formal generations.

Official scores are retained. English questions determine the unit mask before
rescoring; the same mask applies to every language/model. This is a project
benchmark diagnostic, not official COPSD scoring or a new model evaluation run.
"""
import argparse
import ast
import csv
import datetime
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

PROTOCOL = 'project_mgsm_percentage_units_diagnostic_v1'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
LANGUAGES = 'en sw te th bn zh ja ru de fr es'.split()
TARGET_PATTERNS = [r'\bwhat percentage\b', r'expressed as a percentage',
                   r'\bpercentage chance\b']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def data_rows(path):
    with Path(path).open(newline='') as stream:
        rows = [row for row in csv.reader(stream, delimiter='\t') if row]
    assert len(rows) == 250 and all(len(row) == 2 for row in rows)
    return [(q.strip(), a.strip()) for q, a in rows]


def numeric(answer):
    return Fraction(str(answer).replace(',', '').strip())


def official_grade(root):
    from math_verify import parse, verify
    path = root / 'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == 'grade_answer']
    assert len(nodes) == 1
    namespace = {'parse': parse, 'verify': verify}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['grade_answer'], path


def percentage_grade(predicted, gold, percentage_target, grade):
    # A bare prediction uses the benchmark's percentage-value convention.
    # An explicit percent expression uses its symbolic value, so put the
    # numeric gold in the same units. Non-percentage questions stay unchanged.
    if percentage_target and predicted is not None and '%' in predicted:
        return bool(grade(predicted, str(gold).replace(',', '').strip() + r'\%'))
    return bool(grade(predicted, gold))


def self_check(root):
    grade, _ = official_grade(root)
    controls = [('25', '25', True, True), (r'25\%', '25', True, True),
                ('0.25', '25', True, False), (r'0.25\%', '25', True, False),
                (r'2500\%', '25', True, False),
                (r'25\%', '0.25', True, False),
                (r'0.25\%', '0.25', True, True),
                (r'25\%', '25', False, False), (None, '25', True, False)]
    results = []
    for pred, gold, unit, expected in controls:
        actual = percentage_grade(pred, gold, unit, grade)
        assert actual is expected, (pred, gold, unit, actual)
        results.append(dict(predicted=pred, gold=gold, percentage_target=unit,
                            original=bool(grade(pred, gold)), diagnostic=actual))
    assert grade(r'2500\%', '25') is True  # Also test removal of a unit false-positive.
    assert grade(r'25\%', '25') is False
    return results


def prepare(root, output):
    folder = root / 'data/math_single_10lang_20261015/mgsm_rev2/MGSM-Rev2'
    english = data_rows(folder / 'mgsm_en.tsv')
    targets = {str(i) for i, (q, _) in enumerate(english)
               if any(re.search(pattern, q, re.I) for pattern in TARGET_PATTERNS)}
    candidates = {str(i) for i, (q, _) in enumerate(english) if re.search(r'percent|%', q, re.I)}
    assert len(targets) == 4 and targets <= candidates
    inputs = {}
    for language in LANGUAGES:
        path = folder / ('mgsm_' + language + '.tsv')
        rows = data_rows(path)
        assert all(numeric(a) == numeric(b) for (_, a), (_, b) in zip(rows, english))
        inputs[language] = dict(relative=str(path.relative_to(root)), sha256=digest(path),
            percentage_rows={str(i): dict(question_sha256=hashlib.sha256(q.encode()).hexdigest(),
                                         gold=a) for i, (q, a) in enumerate(rows) if str(i) in targets})
    _, source = official_grade(root)
    result = dict(protocol=PROTOCOL, prepared_at=datetime.datetime.now().astimezone().isoformat(),
                  source_sha256=digest(Path(__file__)), official_source_sha256=digest(source),
                  math_verify_version=importlib.metadata.version('math-verify'),
                  annotation_patterns=TARGET_PATTERNS, percentage_target_ids=sorted(targets, key=int),
                  percent_mention_candidates=len(candidates), percentage_answer_targets=len(targets),
                  percent_input_only_excluded=len(candidates - targets), inputs=inputs,
                  rule='Only explicit percentage-answer questions are tagged from canonical English. '
                       'Explicit percent predictions compare against gold in percent units; bare predictions '
                       'retain numeric percentage-value interpretation. Other questions retain official grading.',
                  scope='All250English questions screened independently of model outputs; four targets manually '
                        'reviewed. Numeric gold correspondence verified for English plus ten target languages. '
                        'This covers percentage-symbol conventions only, not every unit/format ambiguity.',
                  GPU_started=False, new_generations=0)
    save(output, result)
    return result


def collect(root, manifest_path, output):
    assert not output.exists()
    manifest = read(manifest_path)
    assert manifest['protocol'] == PROTOCOL and manifest['source_sha256'] == digest(Path(__file__))
    for value in manifest['inputs'].values():
        assert digest(root / value['relative']) == value['sha256']
    grade, official = official_grade(root)
    assert digest(official) == manifest['official_source_sha256']
    assert importlib.metadata.version('math-verify') == manifest['math_verify_version']
    controls = self_check(root)
    logs = root / 'logs/copsd_official_10lang'
    registry = {}

    def register(case, model_id, language, receipt):
        folder = root / case['output_relative']
        hashes = case.get('files_sha256', case.get('file_sha256'))
        assert hashes and set(hashes) >= {'CONFIG.json', 'DONE.json', 'ITEMS.jsonl'}
        key = str(folder.resolve())
        value = dict(folder=folder, hashes=hashes, model_id=model_id, language=language,
                     receipt=receipt, receipt_sha256=digest(receipt))
        if key in registry:
            assert registry[key]['hashes'] == hashes and registry[key]['model_id'] == model_id
        else:
            registry[key] = value

    paths = set(logs.glob('selected4b_*MGSM_pair_*.PRIVATE.json'))
    paths.update(logs.glob('primary4b_*_mgsm_rev2_*_PAIR_*.PRIVATE.json'))
    for path in sorted(paths):
        pair = read(path)
        if not pair.get('case', '').startswith('mgsm_rev2_'):
            continue
        assert pair['proves_complete_seven_case_stage'] is False
        for condition in ['Base', 'model']:
            case = pair[condition]
            register(case, 'Base' if condition == 'Base' else pair['model_id'],
                     case['config']['language'], path)
    legacy = logs / 'primary4b_SW20_cache_audit_20261024.PRIVATE.json'
    if legacy.exists():
        audit = read(legacy)
        assert audit['primary_SW20_only'] and audit['generation_and_scoring_suffix_AST_identical']
        assert audit['all_prompts_inputs_seeds_token_decodes_and_accounting_verified']
        register(audit['cases']['mgsm_rev2_sw'], 'Primary_SW20', 'sw', legacy)
    assert registry
    reports = []
    for value in registry.values():
        folder, language = value['folder'], value['language']
        assert language in LANGUAGES and language != 'en'
        for name, expected in value['hashes'].items():
            assert digest(folder / name) == expected
        config, done = read(folder / 'CONFIG.json'), read(folder / 'DONE.json')
        assert config['model_revision'] == REVISION and config['benchmark'] == 'mgsm_rev2'
        assert config['samples'] == done['n'] == 1 and config['max_new_tokens'] == 8192
        assert config['max_num_seqs'] == done['actual_max_num_seqs'] == 16
        assert done['formal_full_test'] and done['items'] == 250
        rows = [json.loads(line) for line in (folder / 'ITEMS.jsonl').read_text().splitlines()]
        assert len(rows) == 250 and [r['id'] for r in rows] == list(map(str, range(250)))
        tagged = manifest['inputs'][language]['percentage_rows']
        original_count = diagnostic_count = 0
        changed, tagged_results = [], []
        for row in rows:
            sample = row['samples'][0]
            assert len(row['samples']) == 1 and type(sample['correct']) is bool
            original = diagnostic = sample['correct']
            if row['id'] in tagged:
                tag = tagged[row['id']]
                assert hashlib.sha256(row['question'].encode()).hexdigest() == tag['question_sha256']
                assert numeric(row['answer']) == numeric(tag['gold'])
                assert bool(grade(sample['boxed_answer'], row['answer'])) is original
                diagnostic = percentage_grade(sample['boxed_answer'], row['answer'], True, grade)
                item = dict(id=row['id'], predicted=sample['boxed_answer'], gold=row['answer'],
                            original=original, diagnostic=diagnostic)
                tagged_results.append(item)
                if original != diagnostic:
                    changed.append(item)
            original_count += original
            diagnostic_count += diagnostic
        assert len(tagged_results) == 4 and original_count == done['single_correct']
        reports.append(dict(model_id=value['model_id'], language=language, items=250,
            generation_protocol=config['protocol'], output_relative=str(folder.relative_to(root)),
            source_files_sha256=value['hashes'], source_receipt_relative=str(value['receipt'].relative_to(root)),
            source_receipt_sha256=value['receipt_sha256'], original_correct=original_count,
            diagnostic_correct=diagnostic_count, delta_count=diagnostic_count - original_count,
            tagged_results=tagged_results, changed_items=changed))
    result = dict(protocol=PROTOCOL, observed_at=datetime.datetime.now().astimezone().isoformat(),
                  manifest_sha256=digest(manifest_path), source_sha256=digest(Path(__file__)),
                  completed_conditions=len(reports), cases=reports, controls=controls,
                  GPU_started=False, new_generations=0, original_scores_modified=False,
                  scope='Snapshot of accepted completed4B MGSM pairs plus audited original primarySW cache. '
                        'Cohorts/models remain separate; no cross-cohort ranking or aggregate effect claim. '
                        'All four percentage-answer IDs use the same question-derived rule for all models/languages. '
                        'Other246official labels are copied unchanged. Original generations/scoring remain official '
                        'records. This diagnostic does not establish general reasoning quality or complete the study.')
    save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('self-check')
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('collect')
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'self-check':
        print(json.dumps({'controls_passed': len(self_check(root)), 'GPU_started': False}))
        return
    assert args.output.resolve().is_relative_to((root / 'logs').resolve())
    if args.action == 'prepare':
        result = prepare(root, args.output)
        print(json.dumps({k: result[k] for k in ['percentage_answer_targets', 'percent_mention_candidates',
                                                'percent_input_only_excluded', 'GPU_started']}))
    else:
        result = collect(root, args.manifest, args.output)
        print(json.dumps({'completed_conditions': result['completed_conditions'],
                          'changed_conditions': sum(bool(x['changed_items']) for x in result['cases']),
                          'new_generations': 0, 'original_scores_modified': False}))


if __name__ == '__main__':
    main()
