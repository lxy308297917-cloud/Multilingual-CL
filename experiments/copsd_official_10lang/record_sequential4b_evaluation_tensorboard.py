"""Export an accepted seven-case 4B evaluation to separate TensorBoard logs.

Reaudits provenance before writing and reads every scalar back afterward.
Training event directories and frozen training/evaluation code are untouched.
"""
import argparse
import json
import math
from pathlib import Path

from build_sequential4b_matrix import accepted_evaluation
from formal4b_contract import digest, read


def record(root, marker_path, directory, receipt_path):
    root, marker_path = Path(root), Path(marker_path)
    directory, receipt_path = Path(directory), Path(receipt_path)
    assert not directory.exists() and not receipt_path.exists()
    marker = read(marker_path)
    marker, plan = accepted_evaluation(root, marker_path, marker['selection_record_sha256'])
    assert directory.resolve().is_relative_to((root / 'logs').resolve())
    context = marker['model_provenance'].get('stage_context')
    step = 20 * context['stage_index'] if context else 0
    scalars = {}
    for name, case in marker['cases'].items():
        metrics = case['metrics']
        prefix = 'formal/' + name
        scalars[prefix + '/accuracy'] = metrics['single_accuracy']
        scalars[prefix + '/correct'] = metrics['single_correct']
        for field in ['truncated', 'normal_terminations', 'answer_extraction_failures', 'generated_tokens']:
            scalars[prefix + '/' + field] = metrics.get(field, 0)
        for band, values in metrics['scores_by_difficulty'].items():
            scalars[prefix + '/difficulty_' + band + '/accuracy'] = values['single_accuracy']
        scalars[prefix + '/new_GPU_hours'] = case['cost']['new_GPU_hours']
        scalars[prefix + '/reused'] = int(case['cost']['reused'])
    for benchmark in ['mgsm_rev2', 'polymath']:
        values = [case['metrics']['single_accuracy'] for name, case in marker['cases'].items()
                  if name.startswith(benchmark + '_')]
        assert len(values) == 3
        scalars['formal/' + benchmark + '/three_target_macro_accuracy'] = sum(values) / 3
    scalars['cost/new_evaluation_GPU_hours'] = marker['new_evaluation_GPU_hours']
    scalars['cost/new_generated_tokens'] = marker['new_generated_tokens']
    assert all(math.isfinite(value) for value in scalars.values())
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    writer = SummaryWriter(str(directory))
    try:
        for tag, value in scalars.items():
            writer.add_scalar(tag, value, step)
        writer.add_text('provenance/model_id', marker['model_id'], step)
        writer.add_text('provenance/protocol', marker['protocol'], step)
        writer.add_text('provenance/evaluation_marker_sha256', digest(marker_path), step)
        writer.flush()
    finally:
        writer.close()
    events = EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()
    assert set(events.Tags()['scalars']) == set(scalars)
    for tag, expected in scalars.items():
        values = events.Scalars(tag)
        assert len(values) == 1 and values[0].step == step
        # TensorBoard stores scalar events as float32, including token counts.
        assert math.isclose(values[0].value, expected, rel_tol=2e-7, abs_tol=2e-7), tag
    receipt = {'model_id': marker['model_id'], 'global_stage_display_step': step,
               'marker_relative': str(marker_path.relative_to(root)),
               'marker_sha256': digest(marker_path),
               'TensorBoard_relative': str(directory.relative_to(root)),
               'scalar_count': len(scalars), 'all_scalars_read_back': True,
               'source_sha256': digest(Path(__file__)),
               'scope': 'Formal evaluation export only. Display step counts20updates per sequential stage; actual optimizer/scheduler reset each stage. Reused Base evaluation costs stay zero. No new inference or scoring and no training event modification.'}
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return receipt


def record_pair(root, marker_path, directory, receipt_path):
    root, marker_path = Path(root), Path(marker_path)
    directory, receipt_path = Path(directory), Path(receipt_path)
    assert not directory.exists() and not receipt_path.exists()
    assert directory.resolve().is_relative_to((root / 'logs').resolve())
    marker = read(marker_path)
    assert marker['proves_complete_seven_case_stage'] is False
    assert marker['source_sha256'] == digest(Path(__file__).with_name('collect_sequential4b_formal_pair.py'))
    for condition in ['Base', 'model']:
        case = marker[condition]
        folder = root / case['output_relative']
        for name, expected in case['files_sha256'].items():
            assert digest(folder / name) == expected
        assert read(folder / 'DONE.json') == case['metrics']
    step = marker['model']['config']['checkpoint_step']
    assert step == 20
    scalars = {'formal/Base_accuracy': marker['Base']['metrics']['single_accuracy'],
               'formal/model_accuracy': marker['model']['metrics']['single_accuracy'],
               'formal/paired_delta_pp': marker['delta_percentage_points'],
               'formal/question_CI95_lower_pp': marker['paired_question_bootstrap95_CI_pp'][0],
               'formal/question_CI95_upper_pp': marker['paired_question_bootstrap95_CI_pp'][1],
               'formal/gained': marker['gained'], 'formal/lost': marker['lost']}
    for condition in ['Base', 'model']:
        for field in ['truncated', 'normal_terminations', 'answer_extraction_failures']:
            scalars['formal/' + condition + '_' + field] = marker[condition]['metrics'].get(field, 0)
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    writer = SummaryWriter(str(directory))
    try:
        for tag, value in scalars.items():
            assert math.isfinite(value)
            writer.add_scalar(tag, value, step)
        writer.flush()
    finally:
        writer.close()
    events = EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()
    assert set(events.Tags()['scalars']) == set(scalars)
    for tag, expected in scalars.items():
        values = events.Scalars(tag)
        assert len(values) == 1 and values[0].step == step
        assert math.isclose(values[0].value, expected, rel_tol=2e-7, abs_tol=2e-7), tag
    receipt = {'model_id': marker['model_id'], 'case': marker['case'],
               'marker_sha256': digest(marker_path), 'local_checkpoint_step': step,
               'TensorBoard_relative': str(directory.relative_to(root)),
               'scalar_count': len(scalars), 'all_scalars_read_back': True,
               'scope': 'One partial formal pair only. Local checkpoint step20, not cumulative sequential updates; does not prove a complete stage or language preservation. No new inference or modification of training logs.'}
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--evaluation', type=Path)
    mode.add_argument('--pair', type=Path)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.pair:
        result = record_pair(args.root, args.pair, args.directory, args.receipt)
    else:
        result = record(args.root, args.evaluation, args.directory, args.receipt)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
