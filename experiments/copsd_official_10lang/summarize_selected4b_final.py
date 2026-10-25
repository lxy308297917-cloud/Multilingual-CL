"""Summarize completed selected-language analyses without rerunning models.

Requires all eight seven-case acceptances, bound language/TensorBoard receipts,
both canonical full-chain matrices, and six matched single comparisons.
Original2700-pool primary tests remain a separate required experiment.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    logs = root / 'logs/copsd_official_10lang'
    code = Path(__file__).resolve().parent
    state = read(logs / 'sequential4b_analysis_20261024.PRIVATE.json')
    assert state['phase'] == 'completed_selected_language_analysis', state
    assert len(state['completed_analyses']) == 16
    models = ['Base', 'SWmatched', 'TEmatched', 'ZHmatched',
              'forward_s2', 'reverse_s2', 'forward_s3', 'reverse_s3']
    selections, folders, evaluation_events = set(), set(), {}
    evidence, cases, training = {}, {}, {}
    source = lambda label, path: evidence.update({label: digest(path)})
    source('analysis_completion', logs / 'sequential4b_analysis_20261024.PRIVATE.json')
    source('JSONL_CPU_compatibility', code / 'jsonl_physical_line_compat.py')
    for model in models:
        marker = logs / f'selected4b_{model}_FORMAL_COMPLETED_20261024.PRIVATE.json'
        tb = logs / f'selected4b_{model}_FORMAL_TB_20261024.PRIVATE.json'
        lid = logs / f'selected4b_{model}_FORMAL_LANGUAGES_20261024/COMPLETED.PRIVATE.json'
        evaluation, tensorboard, language = read(marker), read(tb), read(lid)
        assert evaluation['all_seven_cases_accepted'] and len(evaluation['cases']) == 7
        assert tensorboard['all_scalars_read_back'] and tensorboard['marker_sha256'] == digest(marker)
        assert language['evaluation_marker_sha256'] == digest(marker)
        assert set(language['cases']) == set(evaluation['cases'])
        selections.add(evaluation['selection_record_sha256'])
        for label, path in [('evaluation', marker), ('TensorBoard', tb), ('language', lid)]:
            source(model + ':' + label, path)
        cases[model] = {}
        for name, case in evaluation['cases'].items():
            assert case['output_relative'] not in folders
            folders.add(case['output_relative'])
            assert language['cases'][name]['items_sha256'] == case['files_sha256']['ITEMS.jsonl']
            assert language['cases'][name]['formal_single_correct'] == case['metrics']['single_correct']
            cases[model][name] = {
                'metrics': case['metrics'], 'language': language['cases'][name],
                'raw_files_sha256': case['files_sha256'],
            }
            cost = case['cost']
            if not cost['reused']:
                assert cost['queue'] not in evaluation_events
                evaluation_events[cost['queue']] = {
                    'GPU_hours': cost['new_GPU_hours'],
                    'generated_tokens': cost['new_generated_tokens'],
                }
    assert len(selections) == 1 and len(folders) == 56 and len(evaluation_events) == 54
    matrices, controls = {}, {}
    for direction in ['forward', 'reverse']:
        path = logs / f'selected4b_{direction}_FORGETTING_20261024.PRIVATE.json'
        value = read(path)
        source(direction + ':full_chain_matrix', path)
        assert value['actual_training_pools_pairwise_disjoint']
        assert value['selection_record_sha256'] in selections
        matrices[direction] = value['matrices']
        for record in value['training_stages']:
            assert record['queue'] not in training
            training[record['queue']] = record
        for index in [1, 2, 3]:
            path = logs / f'selected4b_{direction}_s{index}_MATCHED_COMPARISON_20261024.PRIVATE.json'
            control = read(path)
            assert control['actual_exposure_ID_order_exactly_equal']
            assert control['current_stage_exposures_each'] == 640
            assert control['current_stage_updates_each'] == 20
            assert control['selection_record_sha256'] in selections
            controls[direction + '_s' + str(index)] = control
            source(direction + ':matched_' + str(index), path)
    marker = read(logs / 'selected4b_TEmatched_FORMAL_COMPLETED_20261024.PRIVATE.json')
    path = root / marker['model_provenance']['release_relative']
    assert digest(path) == marker['model_provenance']['release_sha256']
    teacher_reference = read(path)
    training[teacher_reference['queue']] = {key: teacher_reference[key] for key in [
        'stage_context', 'checkpoint_relative', 'adapter_sha256', 'freeze_sha256',
        'online_rollouts', 'generated_tokens', 'valid_response_loss_positions',
        'teacher_forward_calls', 'parent_GPU_hours', 'queue']}
    source('matched_TE_training_release', path)
    assert len(training) == 7
    for record in training.values():
        assert record['online_rollouts'] == record['teacher_forward_calls'] == 640
    train_hours = sum(r['parent_GPU_hours'] for r in training.values())
    eval_hours = sum(r['GPU_hours'] for r in evaluation_events.values())
    return {
        'observed_at': datetime.datetime.now().astimezone().isoformat(),
        'model': 'Qwen/Qwen3-4B', 'model_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
        'protocol': 'project_formal4b_n1_8192_native16_v1',
        'selection_record_sha256': next(iter(selections)),
        'all_56_evaluations_accepted': True, 'eight_full_language_and_TensorBoard_reports_accepted': True,
        'both_complete_chain_matrices_and_six_matched_controls_accepted': True,
        'directional_matrices_and_metrics': matrices, 'cases': cases,
        'matched_current_stage_controls': controls,
        'unique_training_records': training,
        'selected_new_compute_cost': {
            'unique_training_segments': 7, 'unique_new_evaluation_jobs': 54,
            'training_allocated_GPU_hours': train_hours,
            'evaluation_allocated_GPU_hours': eval_hours,
            'combined_allocated_GPU_hours': train_hours + eval_hours,
            'training_exposures': sum(r['online_rollouts'] for r in training.values()),
            'training_generated_tokens': sum(r['generated_tokens'] for r in training.values()),
            'training_valid_loss_positions': sum(r['valid_response_loss_positions'] for r in training.values()),
            'teacher_forward_calls': sum(r['teacher_forward_calls'] for r in training.values()),
            'new_evaluation_generated_tokens': sum(r['generated_tokens'] for r in evaluation_events.values()),
            'scope': 'Distinct planned selected-study events only. First-stage matched-reference training reused, English evaluation shared by the two math matrices, older Base SW/English caches not newly charged. Excludes original2700 primary runs, development, failures, other experiments and pre-job GPU holds. Overlaps queue ledger; never add these subtotals to it. Teacher cost is included in training allocation, not independently timed CUDA or money.',
        },
        'source_evidence_sha256': evidence, 'collector_sha256': digest(Path(__file__)),
        'remaining_full_goal': 'Original2700-pool ten-primary formal evaluations, additional Base language analyses, final all-project cost/reproduction delivery.',
        'full_goal_complete': False,
        'interpretation': [
            'Forward TE acquisition already failed at stage2; a small or negative later forgetting metric does not demonstrate successful retention.',
            'Forward final TE is5/500 PolyMath with494truncations; JSONL reading repair did not change these original scores or explain the learning failure.',
            'Math accuracy and joint target-language counts must be reported together; whole-section language classification is not fluency or per-token purity.',
            'Matched controls equal current-phase exposures/updates, not cumulative budgets: sequential40/60updates versus independent20. Actual token/cost differences remain reported.',
            'One training seed for these matched/sequential models. Per-question intervals condition on fixed models, not seed-population inference or multiple-comparison correction.',
            '20updates per phase under100scheduler and n1/8192/native16 are project adaptations; separate completed1.7B official AfriMGSM pass12 evidence remains distinct.',
            'Engineering success and complete selected analysis do not establish universal method benefit or complete the ten-language experiment.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = summarize(args.root.resolve())
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'directions': 2, 'models': len(result['cases']),
                      'cost': result['selected_new_compute_cost'], 'full_goal_complete': False}))


if __name__ == '__main__':
    main()
