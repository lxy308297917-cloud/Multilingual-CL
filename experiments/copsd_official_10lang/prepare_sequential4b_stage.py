"""Freeze one selected-language 4B research job, without acquiring any GPU."""
import argparse
import json
from pathlib import Path
import re

from prepare_sequential_pools import read, sha
from sequential4b_data import bind
from sequential4b_selection import local, selection, stage_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--direction', choices=['matched', 'forward', 'reverse'], required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--language')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--container-relative', required=True)
    parser.add_argument('--parent-relative')
    parser.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    root, code = args.root.resolve(), Path(__file__).parent
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id)
    assert not args.freeze.exists()
    selection_sha = sha(args.selection)
    selected = selection(root, args.selection, selection_sha)
    role = 'matched_single_reference' if args.direction == 'matched' else 'sequential_stage'
    context = stage_context(selected, args.direction, args.stage, args.language, role)
    container = local(root, args.container_relative)
    assert container.is_relative_to(root / 'models/checkpoint')
    # Provision a new owned checkpoint container separately, inspecting both
    # filesystems before calling this builder. It must already exist.
    assert container.is_dir()
    data = root / 'data/copsd_official_20261015/sequential4b_selected' / (args.run_id + '.json')
    receipt = bind(root, args.selection, selection_sha, args.direction, args.stage,
                   data, args.language, role)
    reference = read(code / 'ru4b_single20_frozen_20261024.PUBLIC.json')
    prepared = read(code / 'sw4b_resource2_prepared_20261022.PUBLIC.json')
    training_args = list(reference['train_arguments'])
    target = context['language'].upper()
    training_args[training_args.index('--train_language') + 1] = target
    training_args[training_args.index('--wandb_project') + 1] = 'COPSD-4B-sequential-matched'
    source_names = set(reference['code_sha256']) | {
        'prepare_sequential4b_stage.py', 'prepare_sequential_pools.py',
        'collect_sequential4b_stage.py',
        'sequential4b_context.py', 'sequential4b_stage_entry.py',
        'sequential4b_stage_job.py', 'run_sequential4b_stage.sh',
        'sequential4b_selection.py', 'sequential4b_data.py',
        'sequential_adapter.py', 'sequential_stage_entry.py', 'formal4b_contract.py'}
    frozen = {
        'protocol': 'project_selected4b_disjoint900_stage20_horizon100_v1',
        'engineering_only': False, 'training_role': role,
        'official_commit': reference['official_commit'],
        'model_revision': selected['model_revision'],
        'selection_record_relative': str(args.selection.relative_to(root)),
        'selection_record_sha256': selection_sha, 'stage_context': context,
        'direction': args.direction, 'stage_index': args.stage,
        'target_language': target, 'run_id': args.run_id,
        'updates': 20, 'scheduler_horizon': 100,
        'training_seed': 42, 'data_seed': 42,
        'micro_batch': 1, 'gradient_accumulation': 32, 'effective_batch': 32,
        'max_completion_length': 2048, 'max_num_seqs': 32,
        'training_data_relative': receipt['training_data_relative'],
        'training_data_sha256': receipt['training_data_sha256'],
        'model_lock_relative': prepared['model_lock_relative'],
        'model_lock_sha256': prepared['model_lock_sha256'],
        'checkpoint_relative': str((container / args.run_id).relative_to(root)),
        'checkpoint_actual_container': str(container.resolve()),
        'disk_required_bytes': 4 * 2**30,
        'train_arguments': training_args, 'wandb_project': 'COPSD-4B-sequential-matched',
        'code_sha256': {name: sha(code / name) for name in sorted(source_names)},
        'installed_source_sha256': reference['installed_source_sha256'],
        'max_account_GPUs': 4, 'training_role_GPUs': 1, 'online_rollout_role_GPUs': 1,
        'parent_checkpoint_relative': args.parent_relative,
        'parent_adapter_sha256': sha(local(root, args.parent_relative) / 'adapter_model.safetensors')
            if args.parent_relative else None,
        'teacher_on_training_GPU_via_disabled_adapter': True,
        'CPU_teacher_forward': False, 'online_student_rollouts': True,
        'optimizer_reset_each_stage': True, 'scheduler_reset_each_stage': True,
        'exposures': 640,
        'scope': 'One explicit research-stage job. No GPU submission, automatic retries, language selection or evaluation result is produced by this builder.'}
    with args.freeze.open('x') as stream:
        json.dump(frozen, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    from sequential4b_context import prepare
    _, rows, _, _ = prepare(root, args.freeze)
    print(json.dumps({'CPU_preflight': True, 'GPU_started': False,
                      'rows': len(rows), 'run_id': args.run_id,
                      'context': context}))


if __name__ == '__main__':
    main()
