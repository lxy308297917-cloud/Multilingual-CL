"""Validate an explicit 4B development selection and language-bound stage data.

No ranking, language selection, queue submission or model execution occurs here.
The existing 1.7B selection/controller is preserved separately.
"""
import argparse
import json
from pathlib import Path
import re

from formal4b_contract import REVISION, LANGUAGES, digest, read


def local(root, relative):
    path = Path(relative)
    assert not path.is_absolute() and '..' not in path.parts
    return root / path


def development_snapshot(path):
    snapshot = read(path)
    assert snapshot['model'] == 'Qwen/Qwen3-4B' and snapshot['model_revision'] == REVISION
    assert snapshot['training_updates_each'] == 20
    assert snapshot['training_scheduler_horizon'] == 100 and snapshot['effective_batch'] == 32
    assert snapshot['training_max_new_tokens'] == 2048
    evaluation = snapshot['evaluation']
    assert evaluation['split'] == 'reserved development' and evaluation['items_each'] == 200
    assert evaluation['n'] == 1 and evaluation['max_new_tokens'] == 8192
    assert evaluation['max_num_seqs'] == 16 and evaluation['question_seed'] == 42
    method = snapshot['training_method']
    assert method['rank'] == 64 and method['alpha'] == 128
    assert method['beta'] == 0 and method['vocabulary_component_clip'] == .05
    models = {}
    for row in snapshot['models']:
        key = row['language'], row['training_seed']
        assert key not in models and key[0] in LANGUAGES and type(key[1]) is int
        for field in ['target_Base_correct_of200', 'target_OPD_correct_of200',
                      'English_Base_correct_of200', 'English_OPD_correct_of200']:
            assert type(row[field]) is int and 0 <= row[field] <= 200
        models[key] = row
    observed = {language for language, seed in models if seed == 42}
    assert observed == LANGUAGES and snapshot['completed_paired_languages'] == 10, (
        'Ten accepted4B language pairs are required before final selection', sorted(observed))
    assert len(models) == snapshot['completed_model_pairs']
    return snapshot, models


def checked_metric(metric):
    assert metric['items'] == 200 and metric['n'] == 1
    assert metric['actual_max_num_seqs'] == 16 and not metric['formal_test']
    assert '4b' in metric['protocol'].lower() and '8192' in metric['protocol']
    return metric['single_correct']


def evidence_counts(report, language, seed):
    if 'training_seed' in report:
        assert report['training_seed'] == report['data_seed'] == seed
    target = report['languages'][language]
    english = report['languages']['en']
    if 'conditions' in target:
        assert report['TensorBoard_four_accuracies_two_costs_read_back']
        case, = [name for name in target['conditions'] if name != 'Base']
        counts = [checked_metric(target['conditions'][name]['metrics']) for name in ['Base', case]]
        counts += [checked_metric(english['conditions'][name]['metrics']) for name in ['Base', case]]
        for pair, b, m in [(report['languages'][language]['paired'], *counts[:2]),
                            (report['languages']['en']['paired'], *counts[2:])]:
            assert pair['Base_correct'] == b and pair['checkpoint_correct'] == m
        return counts
    # The earlier SW seed43 collector stores both seeds in a compact schema.
    assert language == 'sw' and seed == 43
    assert report['TensorBoard_six_accuracies_one_cost_read_back']
    assert checked_metric(target['seed43_metrics']) == target['seed43_correct']
    assert checked_metric(english['seed43_metrics']) == english['seed43_correct']
    return [target['Base_correct'], target['seed43_correct'],
            english['Base_correct'], english['seed43_correct']]


def selection(root, path, expected_sha):
    assert digest(path) == expected_sha
    s = read(path)
    assert s['source_protocol'] == 'independent_development_only'
    assert s['model_revision'] == s['teacher_revision'] == REVISION
    chosen = s['selected_languages']
    assert len(chosen) == len(set(chosen)) == 3 and set(chosen) <= LANGUAGES
    assert s['pool_by_language'] == {lang: i for i, lang in enumerate(chosen)}
    assert re.fullmatch(r'[A-Za-z0-9_-]+', s['experiment_id'])
    assert s['stage_updates'] == 20 and s['scheduler_horizon'] == 100
    assert s['effective_batch'] == 32 and s['max_completion_length'] == 2048
    assert s['training_seed'] == 42
    assert s['optimizer_reset_each_stage'] and s['scheduler_reset_each_stage']
    assert s['adapter_policy'] == 'one_continuous_default_adapter'
    snapshot_path = local(root, s['development_snapshot_relative'])
    assert digest(snapshot_path) == s['development_snapshot_sha256']
    snapshot, models = development_snapshot(snapshot_path)
    assert set(s['development_evidence']) == set(s['review_by_language']) == set(chosen)
    required_reviews = ['math_gain', 'absolute_level', 'target_language', 'English_tradeoff',
                        'termination', 'teacher_signal', 'cost', 'uncertainty']
    for language in chosen:
        evidence = s['development_evidence'][language]
        seeds = [item['training_seed'] for item in evidence]
        assert len(seeds) == len(set(seeds)) >= 2 and 42 in seeds
        bases = set()
        for item in evidence:
            report_path = local(root, item['report_relative'])
            assert digest(report_path) == item['report_sha256']
            counts = evidence_counts(read(report_path), language, item['training_seed'])
            row = models[language, item['training_seed']]
            assert counts == [row[key] for key in ['target_Base_correct_of200', 'target_OPD_correct_of200',
                                                  'English_Base_correct_of200', 'English_OPD_correct_of200']]
            assert counts[1] > counts[0], 'No forced promotion of a nonpositive repeat'
            bases.add((counts[0], counts[2]))
        assert len(bases) == 1
        for key in required_reviews:
            assert isinstance(s['review_by_language'][language][key], str)
            assert s['review_by_language'][language][key].strip(), key
    assert isinstance(s['selection_rationale'], str) and s['selection_rationale'].strip()
    pool_path = local(root, s['pool_manifest_relative'])
    assert digest(pool_path) == s['pool_manifest_sha256']
    pools = read(pool_path)
    assert len(pools['pool_ids']) == 3 and all(len(ids) == 900 for ids in pools['pool_ids'])
    sets = [set(map(str, ids)) for ids in pools['pool_ids']]
    assert all(len(ids) == 900 for ids in sets) and len(set.union(*sets)) == 2700
    assert not set.union(*sets) & set(map(str, pools['development_ids']))
    development_freeze = local(root, s['development_data_freeze_relative'])
    assert digest(development_freeze) == s['development_data_freeze_sha256']
    assert pools['followup_data_freeze_sha256'] == s['development_data_freeze_sha256']
    assert set(s['stage_evaluation_languages']) == set(chosen) | {'en'}
    assert s['stage_evaluation_protocol'] == 'project_formal4b_n1_8192_native16_v1'
    return {**s, 'selection_record_sha256': expected_sha}


def stage_context(s, direction, stage, language=None, role='sequential_stage'):
    chosen = s['selected_languages']
    assert role in ['sequential_stage', 'matched_single_reference']
    if role == 'matched_single_reference':
        assert direction == 'matched' and stage == 1 and language in chosen
        order_id = s['experiment_id'] + '_matched_' + language
    else:
        assert direction in ['forward', 'reverse'] and 1 <= stage <= 3
        order = chosen if direction == 'forward' else list(reversed(chosen))
        expected = order[stage - 1]
        assert language is None or language == expected
        language = expected
        order_id = s['experiment_id'] + '_' + direction
    return {'model_revision': REVISION, 'teacher_revision': REVISION,
            'selection_record_sha256': s['selection_record_sha256'],
            'experiment_role': role, 'order_id': order_id, 'direction': direction,
            'stage_index': stage, 'language': language,
            'pool_index': s['pool_by_language'][language],
            'pool_manifest_sha256': s['pool_manifest_sha256'],
            'updates': 20, 'scheduler_horizon': 100, 'training_seed': 42,
            'effective_batch': 32, 'max_completion_length': 2048,
            'optimizer_reset': True, 'scheduler_reset': True,
            'adapter_policy': 'one_continuous_default_adapter'}


def required_cases(chosen):
    assert len(chosen) == len(set(chosen)) == 3 and set(chosen) <= LANGUAGES
    return [(benchmark, lang) for lang in chosen for benchmark in ['mgsm_rev2', 'polymath']] + [('gsm8k', 'en')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--selection-sha', required=True)
    a = p.parse_args()
    s = selection(a.root, a.selection, a.selection_sha)
    print(json.dumps({'selected_languages': s['selected_languages'],
                      'required_evaluations_per_checkpoint': required_cases(s['selected_languages']),
                      'GPU_work_started': False}))


if __name__ == '__main__':
    main()
