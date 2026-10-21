"""Audit complete followup200 parents and apply the prespecified paired confirmation rule."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from evaluate_vllm import question_seed


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_case(root, code, frozen, freeze_sha, case):
    spec = frozen['cases'][case]
    queue = 'copsd-followup200-' + case.replace('_', '-') + '-v1'
    state = read(root / 'gpu-queue/queues' / queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0, queue
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert job['config']['job']['args'] == [case]
    assert job['script_sha256'] == frozen['runtime_code_sha256']['run_followup200.sh'] == sha(Path(job['script']))
    adapter = root / spec['adapter_relative'] if spec['adapter_relative'] else None
    if adapter:
        assert sha(adapter / 'adapter_model.safetensors') == spec['adapter_sha256']
        assert read(adapter / 'trainer_state.json')['global_step'] == 20
    evaluations = {}
    for lang in spec['evaluation_languages']:
        name = f'followup200_{case}_{lang}_20261021_v1'
        folder = root / 'logs/copsd_official_10lang/followup_development200_v1' / name
        cfg, metrics = read(folder / 'CONFIG.json'), read(folder / 'DONE.json')
        assert cfg['protocol'] == metrics['protocol'] == frozen['protocol']
        assert cfg['case'] == case and cfg['language'] == lang
        for field in ['items', 'samples', 'max_new_tokens', 'max_num_seqs', 'temperature', 'top_p', 'top_k', 'seed']:
            expected = frozen['items_each'] if field == 'items' else frozen[field]
            assert cfg[field] == expected, (case, lang, field)
        assert cfg['data_freeze_sha256'] == freeze_sha
        assert cfg['input_sha256'] == frozen['inputs'][lang]['sha256'] == sha(Path(cfg['input']))
        assert cfg['official_source_sha256'] == frozen['official_evaluation_source_sha256'] == sha(Path(cfg['source']))
        assert cfg['evaluator_sha256'] == frozen['runtime_code_sha256']['evaluate_followup200.py']
        assert Path(cfg['model']).resolve() == (root / 'models/Qwen3-1.7B').resolve()
        if adapter:
            assert Path(cfg['adapter']).resolve() == adapter.resolve() and cfg['checkpoint_step'] == 20
            assert cfg['adapter_sha256'] == spec['adapter_sha256']
        else:
            assert cfg['adapter'] is cfg['checkpoint_step'] is cfg['adapter_sha256'] is None
        rows = [json.loads(s) for s in (folder / 'ITEMS.jsonl').read_text().splitlines()]
        assert [r['id'] for r in rows] == frozen['selected_IDs']
        assert len(rows) == len({r['id'] for r in rows}) == metrics['items'] == metrics['generation_count'] == 200
        assert not metrics['formal_test'] and metrics['n'] == 1 and metrics['actual_max_num_seqs'] == 64
        inputs = read(Path(cfg['input']))
        for row, original in zip(rows, inputs):
            assert row['id'] == str(original['problem_id']) and row['answer'] == original['Answer']
            assert row['question'] == original['problem' if lang == 'en' else 'problem_' + lang]
            assert len(row['samples']) == 1
            sample = row['samples'][0]
            assert type(sample['correct']) is bool
            assert sample['seed'] == question_seed(42, lang, row['id'])
            assert 0 < len(sample['token_ids']) == sample['tokens'] <= 8192
            assert sample['finish_reason'] in ['stop', 'length']
            if sample['correct']: assert sample['boxed_answer'] is not None
        samples = [r['samples'][0] for r in rows]
        for field, values in {
            'single_correct': [s['correct'] for s in samples],
            'generated_tokens': [len(s['token_ids']) for s in samples],
            'truncated': [s['finish_reason'] == 'length' for s in samples],
            'normal_terminations': [s['finish_reason'] == 'stop' for s in samples],
            'answer_extraction_failures': [s['boxed_answer'] is None for s in samples],
        }.items(): assert sum(values) == metrics[field], (case, lang, field)
        assert metrics['single_accuracy'] == metrics['single_correct'] / 200
        evaluations[lang] = {'config': cfg, 'metrics': metrics, 'rows': rows,
            'source': {'run_id': name, 'items_sha256': sha(folder / 'ITEMS.jsonl'), 'config_sha256': sha(folder / 'CONFIG.json')}}
    hours = (datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds() / 3600
    return evaluations, {'queue': queue, 'parent_GPU_hours_counted_once': hours,
                         'evaluation_cases': len(evaluations), 'training_updates': 0}


def paired(base, model):
    for key in ['protocol', 'language', 'model', 'items', 'samples', 'max_new_tokens',
                'max_num_seqs', 'temperature', 'top_p', 'top_k', 'seed', 'input_sha256',
                'official_source_sha256', 'data_freeze_sha256', 'evaluator_sha256', 'packages']:
        assert base['config'][key] == model['config'][key], key
    for x, y in zip(base['rows'], model['rows']):
        assert all(x[k] == y[k] for k in ['id', 'question', 'answer', 'prompt'])
        assert x['samples'][0]['seed'] == y['samples'][0]['seed']
    b, m = [[r['samples'][0]['correct'] for r in run['rows']] for run in [base, model]]
    return {'Base_correct': sum(b), 'checkpoint_correct': sum(m),
            'gained': sum(not x and y for x, y in zip(b, m)),
            'lost': sum(x and not y for x, y in zip(b, m)), 'delta_correct': sum(m) - sum(b)}


def bootstrap_mean(base, first, second):
    flags = lambda x: np.array([r['samples'][0]['correct'] for r in x['rows']], dtype=float)
    delta = (flags(first) + flags(second)) / 2 - flags(base)
    assert len(delta) == 200
    rng = np.random.default_rng(20261021)
    distribution = delta[rng.integers(0, 200, size=(10000, 200))].mean(axis=1)
    lo, hi = np.quantile(distribution, [.025, .975])
    return {'mean_delta_correct_out_of200': float(delta.sum()),
            'mean_delta_percentage_points': float(delta.mean() * 100),
            'paired_question_bootstrap95_CI_percentage_points': [float(lo * 100), float(hi * 100)],
            'bootstrap_replicates': 10000, 'bootstrap_seed': 20261021,
            'CI_lower_strictly_positive': bool(lo > 0),
            'uncertainty_scope': 'Questions only for these two fixed model instances; not a training-seed population CI, new independent test sets or multiplicity-adjusted inference.'}


def collect(root, code, primary_only=False):
    frozen_path = code / 'followup200_frozen_20261021.PUBLIC.json'
    frozen, freeze_sha = read(frozen_path), sha(frozen_path)
    assert frozen['protocol'] == 'project_followup_dev200_single8192_c64_v1'
    assert '10000' in frozen['selection_policy'] and 'seed20261021' in frozen['selection_policy']
    for name, expected in frozen['runtime_code_sha256'].items(): assert sha(code / name) == expected, name
    data, costs = {}, {}
    requested_cases = [case for case in frozen['cases'] if not primary_only or not case.startswith('bn_unclip')]
    for case in requested_cases:
        data[case], costs[case] = audit_case(root, code, frozen, freeze_sha, case)
    comparisons, supported = {}, []
    for lang, prefix in [('sw', 'sw'), ('ja', 'ja'), ('zh', 'zh'), ('bn', 'bn_unclip')]:
        if primary_only and lang == 'bn': continue
        first, second = data[prefix + '42'], data[prefix + '43']
        target_pairs = [paired(data['base'][lang], x[lang]) for x in [first, second]]
        english_pairs = [paired(data['base']['en'], x['en']) for x in [first, second]]
        target_boot = bootstrap_mean(data['base'][lang], first[lang], second[lang])
        english_boot = bootstrap_mean(data['base']['en'], first['en'], second['en'])
        positive = all(p['delta_correct'] > 0 for p in target_pairs)
        passes = positive and target_boot['CI_lower_strictly_positive']
        if lang in frozen['primary_default_candidates'] and passes: supported.append(lang)
        comparisons[lang] = {'group': 'separate_unclipped_intervention' if lang == 'bn' else 'default_component_clip0.05',
            'target_Base_vs_instances42_43': target_pairs, 'English_Base_vs_instances42_43': english_pairs,
            'target_fixed_two_model_mean': target_boot, 'English_fixed_two_model_mean': english_boot,
            'both_target_deltas_positive': positive, 'prespecified_math_confirmation_passes': passes}
    metrics = [x['metrics'] for v in data.values() for x in v.values()]
    expected_cases = 17 if primary_only else 21
    assert len(metrics) == expected_cases and sum(m['items'] for m in metrics) == 200 * expected_cases
    return {'observed_at': datetime.datetime.now().isoformat(), 'protocol': frozen['protocol'],
        'data_freeze_sha256': freeze_sha, 'all_requested_parents_completed_exit0': True,
        'confirmation_stage': 'primary_default_only' if primary_only else 'all_cases',
        'all_nine_parents_completion_verified': not primary_only,
        'optional_BN_cases_not_loaded': ['bn_unclip42', 'bn_unclip43'] if primary_only else [],
        'paired_confirmation': comparisons, 'math_supported_default_candidates': supported,
        'final_three_selected': False, 'language_termination_and_English_review_still_required': True,
        'metrics': {case: {lang: x['metrics'] for lang, x in rr.items()} for case, rr in data.items()},
        'source_records': {case: {lang: x['source'] for lang, x in rr.items()} for case, rr in data.items()},
        'parent_costs': costs, 'cost': {'responses_in_this_stage': 200 * expected_cases,
            'generated_tokens': sum(m['generated_tokens'] for m in metrics),
            'generation_seconds': sum(m['generation_seconds'] for m in metrics),
            'allocated_GPU_hours_unique_parents_once': sum(c['parent_GPU_hours_counted_once'] for c in costs.values()),
            'scope': 'Unique requested parent allocation clocks include startup/CPU/inference and all their languages once (Base includes all5). Not GPU kernel time; exclude queue/hold time. Primary-only and full reports overlap and must not be summed; prior development100/formal outputs are not reused.'},
        'analysis_source_sha256': sha(Path(__file__)), 'numpy_version': importlib.metadata.version('numpy'),
        'selection_policy': frozen['selection_policy'], 'scope': frozen['scope']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--primary-only', action='store_true', help='Audit all7 default-candidate parents first; separate BN diagnostics continue independently.')
    a = p.parse_args()
    assert not a.output.exists()
    out = collect(a.root, Path(__file__).parent, a.primary_only)
    with a.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'math_supported_default_candidates': out['math_supported_default_candidates'], 'paired_confirmation': out['paired_confirmation']}, ensure_ascii=False))


if __name__ == '__main__': main()
