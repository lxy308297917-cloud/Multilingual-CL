"""Compare a sequential stage with its independent same-pool single reference.

Consumes completed seven-suite markers. Does not train, generate, select
languages or conflate accumulated sequential training with single exposure.
"""
import argparse
import datetime
import json
from pathlib import Path

from build_sequential_matrix import audit_marker
from sequential_evaluation import digest, read, selection, write_new


def paired_outputs(reference, stage):
    assert len(reference) == len(stage)
    gained = lost = 0
    for a, b in zip(reference, stage):
        assert all(a[key] == b[key] for key in ['id', 'question', 'answer', 'prompt'])
        assert len(a['samples']) == len(b['samples']) == 1
        x, y = a['samples'][0], b['samples'][0]
        assert x['seed'] == y['seed']
        gained += not x['correct'] and y['correct']
        lost += x['correct'] and not y['correct']
    return {'gained_vs_single': gained, 'lost_vs_single': lost, 'net_correct_vs_single': gained - lost}


def training_identity(root, plan):
    cp = Path(plan['checkpoint'])
    state = read(root / 'gpu-queue/queues' / plan['training_queue'] / 'state.json')
    spec_path = Path(state['job']['config']['job']['args'][0])
    spec = read(spec_path)
    context = read(cp.parent / 'STAGE_CONTEXT.PRIVATE.json')['context']
    assert digest(spec_path) == read(cp.parent / 'STAGE_CONTEXT.PRIVATE.json')['stage_spec_sha256']
    assert digest(spec['training_data']) == context['training_data_sha256']
    assert digest(spec['pool_manifest']) == context['pool_manifest_sha256']
    rows = read(spec['training_data'])
    pool = read(spec['pool_manifest'])
    ids = [row['problem_id'] for row in rows]
    assert ids == pool['pool_ids'][context['pool_index']]
    assert len(ids) == len(set(ids)) and not set(ids) & set(pool['development_ids'])
    assert context['updates'] == 20 and len(ids) >= 640
    if plan['experiment_role'] == 'matched_single_reference':
        assert spec.get('parent_checkpoint') is None and context['parent_checkpoint'] is None
    import torch
    args = torch.load(cp / 'training_args.bin', map_location='cpu', weights_only=False)
    keys = ['learning_rate', 'max_grad_norm', 'max_steps', 'seed', 'data_seed', 'optim',
        'lr_scheduler_type', 'warmup_steps', 'warmup_ratio', 'weight_decay',
        'adam_beta1', 'adam_beta2', 'adam_epsilon', 'bf16', 'fp16',
        'gradient_accumulation_steps', 'per_device_train_batch_size',
        'gradient_checkpointing', 'gradient_checkpointing_kwargs', 'max_length',
        'max_completion_length', 'beta', 'temperature']
    signature = {key: getattr(args, key) for key in keys}
    assert args.max_steps == 100 and args.max_completion_length == 2048
    assert args.gradient_accumulation_steps * args.per_device_train_batch_size == 32
    progress = read(cp / 'COST_SEGMENT_PROGRESS.json')
    assert progress['global_step'] == 20
    return {'context': context, 'signature': signature,
        'spec_sha256': digest(spec_path), 'pool_IDs': ids,
        'cost_at_checkpoint_final_segment_only': progress,
        'fresh_Base_at_experiment_start': plan['experiment_role'] == 'matched_single_reference'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--selection-sha', required=True)
    p.add_argument('--stage-marker', type=Path, required=True)
    p.add_argument('--single-marker', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    code = Path(__file__).parent
    selected = selection(a.selection, a.selection_sha)
    stage = audit_marker(a.root, code, a.stage_marker, a.selection_sha)
    single = audit_marker(a.root, code, a.single_marker, a.selection_sha, 'matched_single_reference')
    stage_plan, single_plan = stage[1], single[1]
    assert stage_plan['stage_index'] in [1, 2, 3]
    assert stage_plan['protocol'] == single_plan['protocol'] == selected['stage_evaluation_protocol']
    # New research plans explicitly record their role.
    assert stage_plan['experiment_role'] == 'sequential_stage'
    st, si = training_identity(a.root, stage_plan), training_identity(a.root, single_plan)
    language = st['context']['language']
    assert language == si['context']['language'] == single_plan['reference_language']
    assert st['signature'] == si['signature'] and st['pool_IDs'] == si['pool_IDs']
    for key in ['pool_index', 'model_revision', 'updates', 'seed', 'training_data_sha256', 'pool_manifest_sha256']:
        assert st['context'][key] == si['context'][key]
    by_single = {(row['benchmark'], row['language']): row for row in single[2]}
    folder = a.root / 'logs/copsd_official_10lang/evaluation_vllm8192'
    comparisons = []
    for row in stage[2]:
        other = by_single[(row['benchmark'], row['language'])]
        assert row['canonical_config'] == other['canonical_config']
        assert row['input_sha256'] == other['input_sha256'] and row['items'] == other['items']
        def outputs(run):
            return [json.loads(line) for line in (folder / run / 'ITEMS.jsonl').read_text().splitlines()]
        paired = paired_outputs(outputs(other['run_id']), outputs(row['run_id']))
        assert paired['net_correct_vs_single'] == row['correct'] - other['correct']
        comparisons.append({'benchmark': row['benchmark'], 'language': row['language'],
            'items': row['items'], 'sequential_correct': row['correct'], 'single_correct': other['correct'],
            'accuracy_delta_percentage_points': 100 * paired['net_correct_vs_single'] / row['items'], **paired})
    write_new(a.output, {'observed_at': datetime.datetime.now().isoformat(),
        'selection_record_sha256': a.selection_sha, 'protocol': stage_plan['protocol'],
        'training_language': language, 'stage_order_id': stage_plan['order_id'],
        'stage_index': stage_plan['stage_index'], 'matched_current_stage_exposures': 640,
        'earlier_sequential_updates': 20 * (stage_plan['stage_index'] - 1),
        'source_marker_sha256': {'stage': digest(a.stage_marker), 'single': digest(a.single_marker)},
        'training_spec_sha256': {'stage': st['spec_sha256'], 'single': si['spec_sha256']},
        'comparisons': comparisons,
        'final_training_segment_costs_only': {key: value['cost_at_checkpoint_final_segment_only']
            for key, value in [('stage', st), ('single', si)]},
        'scope': 'Same current-language pool, question-exposure budget and training settings; earlier sequential stages add intended previous-language exposure. Actual on-policy tokens can differ. This paired comparison is not a full forgetting matrix or full recovery-cost ledger; evaluation costs remain in original markers and must not be repeatedly added when reusing a single across both directions. No model generation or language selection.'})
    print(json.dumps({'seven_paired_suites': len(comparisons), 'training_language': language}))


if __name__ == '__main__':
    main()
