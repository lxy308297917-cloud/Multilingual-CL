"""Audit a full 4B Base/stage1/stage2/stage3 chain before computing forgetting.

Uses completed seven-case markers, original outputs, actual question pools and
accepted training releases. Missing scores are errors, never zero values.
"""
import argparse
import json
from pathlib import Path

from collect_sequential4b_evaluation import collect as audit_evaluation
from collect_sequential4b_stage import collect as audit_training
from formal4b_contract import digest, read
from forgetting_metrics import calculate
from sequential4b_selection import local, selection, stage_context


def accepted_evaluation(root, path, selection_sha):
    marker = read(path)
    assert marker['all_seven_cases_accepted']
    assert marker['selection_record_sha256'] == selection_sha
    plan = local(root, marker['frozen_plan_relative'])
    assert digest(plan) == marker['frozen_plan_sha256']
    queue_map = local(root, marker['queue_map_relative'])
    assert digest(queue_map) == marker['queue_map_sha256']
    assert audit_evaluation(root, plan, queue_map, None) == marker
    return marker, read(plan)


def accepted_training(root, plan):
    provenance = plan['model_provenance']
    release = local(root, provenance['release_relative'])
    assert digest(release) == provenance['release_sha256']
    marker = read(release)
    freeze = local(root, marker['freeze_relative'])
    assert digest(freeze) == marker['freeze_sha256']
    assert audit_training(root, freeze, marker['queue'], None) == marker
    return marker, read(freeze)


def identity_rows(folder):
    rows = [json.loads(line) for line in (folder / 'ITEMS.jsonl').read_text().splitlines()]
    return [(row['id'], row['question'], row['answer'], row['prompt'],
             row['samples'][0]['seed']) for row in rows]


def build(root, selection_path, direction, markers):
    root = Path(root)
    selection_sha = digest(selection_path)
    selected = selection(root, selection_path, selection_sha)
    assert direction in ['forward', 'reverse']
    order = selected['selected_languages']
    if direction == 'reverse':
        order = list(reversed(order))
    assert len(markers) == 4 and len({Path(path).resolve() for path in markers}) == 4
    audits = [accepted_evaluation(root, path, selection_sha) for path in markers]
    assert audits[0][1]['model_provenance'] == {'original_Base': True}
    baseline = audits[0][0]['cases']
    identities = {name: identity_rows(local(root, case['output_relative']))
                  for name, case in baseline.items()}
    stage_labels = ['Base'] + ['after_' + language for language in order]
    scores = {benchmark: {} for benchmark in ['mgsm_rev2', 'polymath']}
    cost_events, training_records, pools, diagnostics = {}, [], [], {}
    previous_cp = None
    previous_adapter = None
    for index, ((marker, plan), label) in enumerate(zip(audits, stage_labels)):
        assert set(marker['cases']) == set(baseline)
        diagnostics[label] = {name: {key: case['metrics'].get(key, 0)
                                    for key in ['items', 'single_correct', 'generated_tokens',
                                                'truncated', 'normal_terminations',
                                                'answer_extraction_failures', 'scores_by_difficulty']}
                              for name, case in marker['cases'].items()}
        for name, case in marker['cases'].items():
            folder = local(root, case['output_relative'])
            assert identity_rows(folder) == identities[name], 'Changed questions, prompts or request seeds'
            if not case['cost']['reused']:
                event_id = 'evaluation:' + case['cost']['queue']
                event = {'GPU_hours': case['cost']['new_GPU_hours'],
                         'generated_tokens': case['cost']['new_generated_tokens'],
                         'queue_state_sha256': case['cost']['queue_state_sha256']}
                assert event_id not in cost_events or cost_events[event_id] == event
                cost_events[event_id] = event
        english = marker['cases']['gsm8k_en']['metrics']
        for benchmark, matrix in scores.items():
            row = {}
            for language in order:
                metrics = marker['cases'][benchmark + '_' + language]['metrics']
                row[language] = {'correct': metrics['single_correct'], 'total': metrics['items']}
            row['en'] = {'correct': english['single_correct'], 'total': english['items']}
            matrix[label] = row
        if index == 0:
            continue
        training, frozen = accepted_training(root, plan)
        expected = stage_context(selected, direction, index)
        if index == 1 and training['stage_context']['experiment_role'] == 'matched_single_reference':
            assert selected['first_stage_reuse_policy'] == 'reuse_exact_matched_single'
            expected = stage_context(selected, 'matched', 1, order[0], 'matched_single_reference')
        assert training['stage_context'] == expected
        cp = local(root, training['checkpoint_relative'])
        spec, = {tuple((case['adapter_relative'], case['adapter_sha256'])) for case in plan['cases'].values()}
        assert local(root, spec[0]).resolve() == cp.resolve() and spec[1] == training['adapter_sha256']
        if index == 1:
            assert not frozen.get('parent_checkpoint_relative')
        else:
            assert local(root, frozen['parent_checkpoint_relative']).resolve() == previous_cp.resolve()
            assert frozen['parent_adapter_sha256'] == previous_adapter
        data = local(root, frozen['training_data_relative'])
        assert digest(data) == frozen['training_data_sha256']
        rows = read(data)
        manifest = read(local(root, selected['pool_manifest_relative']))
        ids = [row['problem_id'] for row in rows]
        assert ids == manifest['pool_ids'][selected['pool_by_language'][order[index - 1]]]
        assert len(set(ids)) == 900 and not set(ids) & set(manifest['development_ids'])
        pools.append(set(ids))
        previous_cp, previous_adapter = cp, training['adapter_sha256']
        training_records.append({key: training[key] for key in [
            'stage_context', 'checkpoint_relative', 'adapter_sha256', 'freeze_sha256',
            'online_rollouts', 'generated_tokens', 'valid_response_loss_positions',
            'teacher_forward_calls', 'parent_GPU_hours', 'queue']})
        event_id = 'training:' + training['queue']
        cost_events[event_id] = {'GPU_hours': training['parent_GPU_hours'],
                                'generated_tokens': training['generated_tokens'],
                                'valid_response_loss_positions': training['valid_response_loss_positions']}
    assert all(not pools[i] & pools[j] for i in range(3) for j in range(i))
    return {'order_id': selected['experiment_id'] + '_' + direction,
            'selection_record_sha256': selection_sha,
            'protocol': selected['stage_evaluation_protocol'],
            'matrices': {benchmark: calculate({'order': order, 'scores': matrix})
                         for benchmark, matrix in scores.items()},
            'source_markers': [{'relative': str(Path(path).relative_to(root)), 'sha256': digest(path)} for path in markers],
            'training_stages': training_records, 'actual_training_pools_pairwise_disjoint': True,
            'generation_diagnostics': diagnostics,
            'unique_cost_events': cost_events,
            'unique_new_GPU_hours': sum(event['GPU_hours'] for event in cost_events.values()),
            'scope': 'One complete audited direction. English appears in both math matrices but its evaluation cost counts once. Retain negative forgetting. Union cost-event IDs across directions and matched singles before reporting aggregate cost. Language analyses, matched-single comparisons and failed/recovery work remain separate required reports.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--direction', choices=['forward', 'reverse'], required=True)
    parser.add_argument('--markers', type=Path, nargs=4, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = build(args.root, args.selection, args.direction, args.markers)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'order_id': result['order_id'], 'complete_math_matrices': 2}))


if __name__ == '__main__':
    main()
