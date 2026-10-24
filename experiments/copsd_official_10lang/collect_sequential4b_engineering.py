"""Accept the engineering-only 4B stage2 smoke; never report math efficacy."""
import argparse
import datetime
import json
import math
from pathlib import Path

from single4b_compact_entry import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    code = Path(__file__).parent
    f = read(a.freeze)
    assert f['engineering_only'] and f['updates'] == f['stop_after_updates'] == 2
    for name, digest in f['code_sha256'].items():
        assert sha(code / name) == digest, name
    state = read(a.root / 'gpu-queue/queues/copsd-sequential4b-engineering2-v1/state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    job = state['job']
    assert len(job['gpu_uuids']) == 2 and job['config']['max_user_gpus'] == 4
    assert job['config']['job']['args'] == [str(a.freeze.resolve())]
    assert job['script_sha256'] == sha(code / 'run_sequential4b_engineering.sh')
    folder = a.root / f['checkpoint_relative']
    proof = read(folder / 'SEQUENTIAL4B_ENGINEERING_UPDATES_DONE.json')
    assert proof['engineering_only'] and proof['global_step'] == proof['completed_updates'] == 2
    assert proof['micro_steps'] == 64 and proof['effective_batch'] == 32
    assert proof['scheduler_horizon'] == 100 and proof['parent_step'] == 20
    assert proof['all_HF_parameters_on_CUDA'] and proof['base_unchanged_bitwise']
    assert proof['teacher_via_disabled_original_Base']
    handoff = proof['stage_handoff']
    assert handoff['adapter_count'] == 1 and handoff['previous_global_step'] == 20
    assert handoff['adapter_values_match_previous_checkpoint']
    assert handoff['parameter_objects_preserved'] and handoff['frozen_Base_unchanged']
    reset = proof['stage_optimizer_start']
    assert reset['mode'] == 'stage_reset' and reset['global_step'] == 0
    assert reset['optimizer_state_empty'] and reset['scheduler_last_epoch'] == 0
    parent = a.root / f['parent_checkpoint_relative']
    previous = read(parent.parent / 'SERVER_PILOT20_UPDATES_DONE.json')
    syncs = proof['syncs']
    assert [s['global_step'] for s in syncs] == [0, 1, 2]
    assert all(s['parameter_count'] == 398 for s in syncs)
    assert syncs[0]['adapter_hash'] == previous['syncs'][-1]['adapter_hash']
    assert len({s['adapter_hash'] for s in syncs}) == 3
    assert read(folder / 'SERVER_CLIENT_SHUTDOWN_DONE.json')['communicator_closed']
    parent_done = read(folder / 'SERVER_PARENT_DONE.json')
    assert parent_done['training_exit_code'] == 0 and parent_done['two_role_job_completed']
    assert parent_done['server_cleanup_completed']
    cost = proof['cost']
    assert cost['loss_calls'] == cost['teacher_forward_calls'] == cost['student_forward_calls'] == 64
    groups = proof['batch_generation']
    assert [g['global_step'] for g in groups] == [0, 1]
    assert all(g['rollouts'] == 32 for g in groups)
    generated = sum(g['generated_tokens'] for g in groups)
    assert 0 < cost['valid_response_loss_positions'] <= generated <= 64 * 2048
    import torch
    cp = folder / 'checkpoint-2'
    assert read(cp / 'trainer_state.json')['global_step'] == 2
    args = torch.load(cp / 'training_args.bin', map_location='cpu', weights_only=False)
    assert args.max_steps == 100 and args.gradient_accumulation_steps == 32
    assert args.per_device_train_batch_size == 1 and args.learning_rate == 5e-6
    assert args.max_grad_norm == .1 and 'adamw' in args.optim.value.lower()
    assert torch.load(cp / 'scheduler.pt', map_location='cpu', weights_only=False)['last_epoch'] == 2
    optimizer = torch.load(cp / 'optimizer.pt', map_location='cpu', weights_only=False)
    assert {int(v['step']) for v in optimizer['state'].values()} == {2}
    rng = torch.load(cp / 'rng_state.pth', map_location='cpu', weights_only=False)
    assert {'python', 'numpy', 'cpu', 'cuda'} <= set(rng)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(a.root / 'logs/copsd_official_10lang/tensorboard' / f['run_id']))
    events.Reload()
    for tag in ['train/loss', 'train/learning_rate']:
        values = events.Scalars(tag)
        assert values and any(v.step == 2 for v in values)
        assert all(math.isfinite(v.value) for v in values)
    hours = (datetime.datetime.fromisoformat(job['finished_at']) -
             datetime.datetime.fromisoformat(job['started_at'])).total_seconds() / 3600
    result = {
        'completed_at': datetime.datetime.now().isoformat(), 'engineering_only': True,
        'model': 'Qwen/Qwen3-4B', 'updates': 2, 'scheduler_horizon': 100,
        'effective_batch': 32, 'max_completion_length': 2048,
        'online_rollouts': 64, 'generated_tokens': generated,
        'valid_response_loss_positions': cost['valid_response_loss_positions'],
        'teacher_forward_calls': 64, 'current_student_weight_syncs': 3,
        'same_adapter_values_and_parameter_objects_loaded': True,
        'initial_rollout_weights_match_learned_parent': True,
        'original_Base_bitwise_unchanged': True,
        'actual_Trainer_optimizer_and_scheduler_reset_to0': True,
        'checkpoint2_optimizer_scheduler_RNG_verified': True,
        'TensorBoard_loss_learning_rate_read_back': True,
        'parent_minutes': hours * 60, 'parent_GPU_hours': hours * 2,
        'research_sequential_training_started': False,
        'scope': 'Engineering prerequisite only, using a previously learned RU20 adapter and reusing its training pool. Official-core clipped loss, AdamW and full2048 budget retained. Does not select RU, establish efficacy/forgetting, test fresh within-stage resumption, or validate a production three-language data/evaluation controller. Debug checkpoints must not seed research stages.',
    }
    with a.output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    private = a.root / 'logs/copsd_official_10lang/sequential4b_engineering2_acceptance_20261024.PRIVATE.json'
    with private.open('x') as stream:
        stream.write(json.dumps({'queue_state': state, 'freeze_sha256': sha(a.freeze),
                                'proof_sha256': sha(folder / 'SEQUENTIAL4B_ENGINEERING_UPDATES_DONE.json'),
                                'collector_sha256': sha(Path(__file__)),
                                'checkpoint_adapter_sha256': sha(cp / 'adapter_model.safetensors')}, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
