"""Accept a completed 4B research stage; keep efficacy claims in evaluations."""
import argparse
import datetime
import json
import math
from pathlib import Path

from prepare_sequential_pools import read, sha
from sequential4b_selection import local, selection, stage_context


def collect(root, freeze, queue, output):
    root, freeze = Path(root), Path(freeze)
    output = Path(output) if output is not None else None
    code = Path(__file__).parent
    assert output is None or not output.exists()
    f = read(freeze)
    assert f['engineering_only'] is False and f['updates'] == 20
    for name, digest in f['code_sha256'].items():
        assert sha(code / name) == digest, name
    s = selection(root, local(root, f['selection_record_relative']), f['selection_record_sha256'])
    expected = stage_context(s, f['direction'], f['stage_index'],
                             f['target_language'].lower(), f['training_role'])
    assert f['stage_context'] == expected
    assert queue and '/' not in queue and '..' not in queue
    state = read(root / 'gpu-queue/queues' / queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    job = state['job']
    assert len(set(job['gpu_uuids'])) == 2 and job['config']['max_user_gpus'] == 4
    assert job['config']['job']['args'] == [str(freeze.resolve())]
    assert job['script_sha256'] == sha(code / 'run_sequential4b_stage.sh')
    folder = local(root, f['checkpoint_relative'])
    proof = read(folder / 'SEQUENTIAL4B_RESEARCH_UPDATES_DONE.json')
    assert proof['engineering_only'] is False and proof['stage_context'] == expected
    assert proof['global_step'] == proof['completed_updates'] == 20
    assert proof['micro_steps'] == 640 and proof['effective_batch'] == 32
    assert proof['scheduler_horizon'] == 100
    assert proof['all_HF_parameters_on_CUDA'] and proof['base_unchanged_bitwise']
    assert proof['teacher_via_disabled_original_Base']
    context = read(folder / 'SEQUENTIAL4B_STAGE_CONTEXT.PRIVATE.json')
    assert context['context'] == expected and context['freeze_sha256'] == sha(freeze)
    reset = proof['stage_optimizer_start']
    assert reset['mode'] == 'stage_reset' and reset['global_step'] == 0
    assert reset['optimizer_state_empty'] and reset['scheduler_last_epoch'] == 0
    handoff = proof['stage_handoff']
    assert handoff['adapter_count'] == 1
    parent_relative = f.get('parent_checkpoint_relative')
    if parent_relative:
        parent = local(root, parent_relative)
        assert sha(parent / 'adapter_model.safetensors') == f['parent_adapter_sha256']
        assert handoff['previous_global_step'] == 20
        assert handoff['adapter_values_match_previous_checkpoint']
        assert handoff['parameter_objects_preserved'] and handoff['frozen_Base_unchanged']
        previous = read(parent.parent / 'SEQUENTIAL4B_RESEARCH_UPDATES_DONE.json')
        assert proof['syncs'][0]['adapter_hash'] == previous['syncs'][-1]['adapter_hash']
    else:
        assert handoff['fresh_original_Base'] and proof['parent_step'] is None
    syncs = proof['syncs']
    assert [row['global_step'] for row in syncs] == list(range(21))
    assert all(row['parameter_count'] == 398 for row in syncs)
    assert len({row['adapter_hash'] for row in syncs}) == 21
    assert read(folder / 'SERVER_CLIENT_SHUTDOWN_DONE.json')['communicator_closed']
    parent_done = read(folder / 'SERVER_PARENT_DONE.json')
    assert parent_done['training_exit_code'] == 0 and parent_done['two_role_job_completed']
    assert parent_done['server_cleanup_completed']
    cost = proof['cost']
    assert cost['loss_calls'] == cost['teacher_forward_calls'] == cost['student_forward_calls'] == 640
    groups = proof['batch_generation']
    assert [row['global_step'] for row in groups] == list(range(20))
    assert all(row['rollouts'] == 32 for row in groups)
    generated = sum(row['generated_tokens'] for row in groups)
    assert 0 < cost['valid_response_loss_positions'] <= generated <= 640 * 2048
    data = local(root, f['training_data_relative'])
    assert sha(data) == f['training_data_sha256'] and len(read(data)) == 900
    assert len(read(folder / 'EXPOSURE_PROMPTS.PRIVATE.json')) == 640
    import torch
    cp = folder / 'checkpoint-20'
    trainer_state = read(cp / 'trainer_state.json')
    assert trainer_state['global_step'] == 20
    for item in trainer_state['log_history']:
        assert all(math.isfinite(value) for value in item.values() if isinstance(value, float))
    args = torch.load(cp / 'training_args.bin', map_location='cpu', weights_only=False)
    assert args.max_steps == 100 and args.gradient_accumulation_steps == 32
    assert args.per_device_train_batch_size == 1 and args.learning_rate == 5e-6
    assert args.max_grad_norm == .1 and 'adamw' in args.optim.value.lower()
    assert args.seed == args.data_seed == 42
    assert torch.load(cp / 'scheduler.pt', map_location='cpu', weights_only=False)['last_epoch'] == 20
    optimizer = torch.load(cp / 'optimizer.pt', map_location='cpu', weights_only=False)
    assert {int(value['step']) for value in optimizer['state'].values()} == {20}
    rng = torch.load(cp / 'rng_state.pth', map_location='cpu', weights_only=False)
    assert {'python', 'numpy', 'cpu', 'cuda'} <= set(rng)
    from safetensors.torch import load_file
    weights = load_file(str(cp / 'adapter_model.safetensors'), device='cpu')
    assert len(weights) == 504 and all(torch.isfinite(value).all() for value in weights.values())
    adapter = read(cp / 'adapter_config.json')
    assert adapter['r'] == 64 and adapter['lora_alpha'] == 128
    assert Path(adapter['base_model_name_or_path']).resolve() == (root / 'models/Qwen3-4B').resolve()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(root / 'logs/copsd_official_10lang/tensorboard' / f['run_id']))
    events.Reload()
    for tag in ['train/loss', 'train/learning_rate']:
        values = events.Scalars(tag)
        assert values and any(value.step == 20 for value in values)
        assert all(math.isfinite(value.value) for value in values)
    hours = (datetime.datetime.fromisoformat(job['finished_at']) -
             datetime.datetime.fromisoformat(job['started_at'])).total_seconds() / 3600
    result = {'engineering_only': False, 'global_step': 20, 'stage_context': expected,
              'selection_record_sha256': f['selection_record_sha256'],
              'freeze_sha256': sha(freeze), 'adapter_sha256': sha(cp / 'adapter_model.safetensors'),
              'freeze_relative': str(freeze.relative_to(root)),
              'collector_sha256': sha(Path(__file__)),
              'checkpoint_relative': str(cp.relative_to(root)),
              'online_rollouts': 640, 'generated_tokens': generated,
              'valid_response_loss_positions': cost['valid_response_loss_positions'],
              'teacher_forward_calls': 640, 'current_student_weight_syncs': 21,
              'original_Base_bitwise_unchanged': True,
              'actual_stage_optimizer_scheduler_reset': True,
              'checkpoint20_optimizer_scheduler_RNG_verified': True,
              'TensorBoard_loss_learning_rate_read_back': True,
              'parent_minutes': hours * 60, 'parent_GPU_hours': hours * 2,
              'queue': queue, 'queue_state': state,
              'scope': 'Local checkpoint release and training accounting only. No mathematical efficacy, language preservation or forgetting score is inferred from completion.'}
    if output is not None:
        with output.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root, args.freeze, args.queue, args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ['queue_state']}))


if __name__ == '__main__':
    main()
