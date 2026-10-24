"""Compare a complete 4B stage to its exact current-pool single reference.

Equal current-stage exposure does not equal equal cumulative sequential budget.
"""
import argparse
import json
from pathlib import Path

from build_sequential4b_matrix import accepted_evaluation, accepted_training, identity_rows
from formal4b_contract import digest, read
from sequential4b_selection import selection, local, stage_context


def compare(root, selected_path, direction, index, stage_marker, single_marker,
            stage_exposures, single_exposures):
    root = Path(root)
    selection_sha = digest(selected_path)
    s = selection(root, selected_path, selection_sha)
    expected = stage_context(s, direction, index)
    stage, stage_plan = accepted_evaluation(root, stage_marker, selection_sha)
    single, single_plan = accepted_evaluation(root, single_marker, selection_sha)
    st, sf = accepted_training(root, stage_plan)
    si, rf = accepted_training(root, single_plan)
    reference_context = stage_context(s, 'matched', 1, expected['language'], 'matched_single_reference')
    assert si['stage_context'] == reference_context
    if index == 1:
        assert s['first_stage_reuse_policy'] == 'reuse_exact_matched_single'
        assert st['stage_context'] == reference_context
        assert st['adapter_sha256'] == si['adapter_sha256']
        assert local(root, st['checkpoint_relative']).resolve() == local(root, si['checkpoint_relative']).resolve()
    else:
        assert st['stage_context'] == expected
    assert sf['train_arguments'] == rf['train_arguments']
    assert sf['training_seed'] == sf['data_seed'] == rf['training_seed'] == rf['data_seed'] == 42
    assert sf['training_data_sha256'] == rf['training_data_sha256']
    assert read(local(root, sf['training_data_relative'])) == read(local(root, rf['training_data_relative']))
    exposures = []
    for path, release, freeze in [(stage_exposures, st, sf), (single_exposures, si, rf)]:
        manifest = read(path)
        assert manifest['run_id'] == freeze['run_id']
        assert manifest['freeze_sha256'] == release['freeze_sha256']
        assert manifest['exposures'] == manifest['unique_original_ids'] == 640
        assert manifest['development_overlap'] == 0 and manifest['student_and_teacher_prompt_pairs_all_exact']
        source = local(root, freeze['checkpoint_relative']) / 'EXPOSURE_PROMPTS.PRIVATE.json'
        assert digest(source) == manifest['recorded_prompt_sha256']
        exposures.append(manifest['actual_exposure_original_ids_in_order'])
    assert exposures[0] == exposures[1], 'Report actual data mismatch instead of claiming exact matched exposure'
    import numpy as np
    comparisons = {}
    for name, result in stage['cases'].items():
        reference = single['cases'][name]
        stage_folder, ref_folder = [local(root, v['output_relative']) for v in [result, reference]]
        assert identity_rows(stage_folder) == identity_rows(ref_folder)
        rows = [[json.loads(line) for line in (folder / 'ITEMS.jsonl').read_text().splitlines()]
                for folder in [ref_folder, stage_folder]]
        before, after = [np.array([r['samples'][0]['correct'] for r in rs], dtype=np.int8) for rs in rows]
        delta = after - before
        rng = np.random.default_rng(20261024)
        bootstrap = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1) * 100
        comparisons[name] = {'items': len(delta), 'single_correct': int(before.sum()),
                             'stage_correct': int(after.sum()), 'gained_vs_single': int((delta == 1).sum()),
                             'lost_vs_single': int((delta == -1).sum()),
                             'delta_percentage_points': float(delta.mean() * 100),
                             'paired_question_bootstrap95_CI_pp': np.quantile(bootstrap, [.025, .975]).tolist()}
    return {'direction': direction, 'stage_index': index, 'training_language': expected['language'],
            'selection_record_sha256': selection_sha,
            'stage_marker_sha256': digest(stage_marker), 'single_marker_sha256': digest(single_marker),
            'stage_exposure_manifest_sha256': digest(stage_exposures),
            'single_exposure_manifest_sha256': digest(single_exposures),
            'current_stage_exposures_each': 640, 'current_stage_updates_each': 20,
            'actual_exposure_ID_order_exactly_equal': True, 'stage1_literal_reuse': index == 1,
            'sequential_cumulative_updates': 20 * index, 'single_cumulative_updates': 20,
            'training_cost_records': {key: {field: release[field] for field in [
                'parent_GPU_hours', 'generated_tokens', 'valid_response_loss_positions', 'teacher_forward_calls']}
                for key, release in [('stage_current_segment', st), ('single', si)]},
            'comparisons': comparisons,
            'scope': 'Full seven-case fixed-protocol comparison. Current900pool,640actualorderedexposures,seed,optimizer/loss/hyperparameter arguments matched. Sequential cumulative budget is larger forstage2/3; differences cannot establish forgetting prevention or remove additional-data effects. Per-question CIs describe fixed models, not seed populations or multiple-comparison corrected inference. Stage1 reuse is one computation, not two costs.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--direction', choices=['forward', 'reverse'], required=True)
    parser.add_argument('--stage-index', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--stage-marker', type=Path, required=True)
    parser.add_argument('--single-marker', type=Path, required=True)
    parser.add_argument('--stage-exposures', type=Path, required=True)
    parser.add_argument('--single-exposures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = compare(args.root, args.selection, args.direction, args.stage_index,
                     args.stage_marker, args.single_marker, args.stage_exposures, args.single_exposures)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'direction': args.direction, 'stage': args.stage_index, 'complete_cases': len(result['comparisons'])}))


if __name__ == '__main__':
    main()
