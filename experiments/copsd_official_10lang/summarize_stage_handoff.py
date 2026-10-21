"""Audit actual SW stage handoff or its one-update checkpoint recovery."""
import argparse
import datetime
import json
from pathlib import Path

from export_eval_dense_cpu import digest, read, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--queue', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    code = Path(__file__).parent
    spec = read(a.config)
    assert spec['engineering_only'] and spec['language'] == 'sw' and spec['updates'] == 2
    assert spec['entry_sha256'] == digest(code / 'sequential_stage_entry.py')
    assert spec['wrapper_sha256'] == digest(code / 'run_sequential_stage.sh')
    state = read(a.root / 'gpu-queue/queues' / a.queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert [str(Path(x).resolve()) for x in job['config']['job']['args']] == [str(a.config.resolve())]
    assert digest(job['script']) == job['script_sha256'] == spec['wrapper_sha256']
    model = a.root / 'models/checkpoint/copsd_sequential_stages_v1' / spec['run_id']
    proof = read(model / 'COLLOCATE_TRAIN_UPDATES_DONE.json')
    context = read(model / 'STAGE_CONTEXT.PRIVATE.json')
    assert context['stage_spec_sha256'] == digest(a.config)
    assert context['context'] == proof['sequential_context'] and context['context']['engineering_only']
    assert read(model / 'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
    start = 1 if spec.get('resume_checkpoint') else 0
    assert proof['start_step'] == start and proof['global_step'] == 2
    assert proof['completed_updates'] == 2 - start and proof['micro_steps'] == 32 * (2 - start)
    assert proof['scheduler_horizon'] == 100 and proof['effective_batch'] == 32
    assert proof['max_completion_length'] == 2048 and proof['base_unchanged_bitwise']
    initial = proof['optimization_start']
    assert initial['global_step'] == initial['scheduler_last_epoch'] == start
    assert initial['optimizer_state_empty'] == (start == 0)
    assert initial['mode'] == ('within_stage_resume' if start else 'stage_reset')
    assert context['initialization']['adapter_count'] == 1
    assert context['initialization']['adapter_values_match_previous_checkpoint']
    assert context['initialization']['frozen_Base_unchanged']
    source = Path(spec.get('resume_checkpoint') or spec['parent_checkpoint'])
    source_proof = read(source.parent / 'COLLOCATE_TRAIN_UPDATES_DONE.json')
    source_step = read(source / 'trainer_state.json')['global_step']
    expected_sync = [s for s in source_proof['syncs'] if s['global_step'] == source_step]
    assert len(expected_sync) == 1
    assert [s['global_step'] for s in proof['syncs']] == list(range(start, 3))
    for key in ['adapter_hash', 'engine_hash']:
        assert proof['syncs'][0][key] == expected_sync[0][key]
        assert proof['syncs'][0][key] != proof['syncs'][-1][key]
    cost = read(model / 'COST_SEGMENT_DONE.json')
    assert cost['start_step'] == start and cost['global_step'] == 2
    assert cost['loss_calls'] == cost['student_forward_calls'] == cost['teacher_forward_calls'] == 32 * (2 - start)
    groups = read(model / 'UPDATE_BATCH_GENERATION.json')['groups']
    assert [g['global_step'] for g in groups] == list(range(start, 2))
    assert all(g['rollouts'] == 32 for g in groups)
    generated = sum(g['generated_tokens'] for g in groups)
    assert 0 < cost['valid_response_loss_positions'] <= generated
    log = Path(job['log_file']).read_text(errors='replace')
    calls = [json.loads(line.split('COLLOCATE_MICROSTEP ', 1)[1])
        for line in log.splitlines() if 'COLLOCATE_MICROSTEP ' in line]
    assert len(calls) == len({r['micro_step'] for r in calls}) == cost['loss_calls']
    assert all(r['loss_finite'] for r in calls)
    import torch
    for step in range(start + 1, 3):
        cp = model / f'checkpoint-{step}'
        assert read(cp / 'trainer_state.json')['global_step'] == step
        assert torch.load(cp / 'scheduler.pt', map_location='cpu', weights_only=False)['last_epoch'] == step
        optimizer = torch.load(cp / 'optimizer.pt', map_location='cpu', weights_only=False)
        assert {int(v['step']) for v in optimizer['state'].values()} == {step}
        rng = torch.load(cp / 'rng_state.pth', map_location='cpu', weights_only=False)
        assert all(key in rng for key in ['python', 'numpy', 'cpu', 'cuda'])
        assert (cp / 'adapter_model.safetensors').stat().st_size > 0
    args = torch.load(model / 'checkpoint-2/training_args.bin', map_location='cpu', weights_only=False)
    assert args.max_steps == 100 and args.learning_rate == 5e-6 and args.max_grad_norm == .1
    assert args.per_device_train_batch_size == 1 and args.gradient_accumulation_steps == 32
    assert args.seed == args.data_seed == 42 and 'tensorboard' in args.report_to
    assert list(Path(args.logging_dir).glob('events.out.tfevents.*'))
    wall = (datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds()
    result = {'observed_at': datetime.datetime.now().isoformat(), 'model': 'Qwen3-1.7B',
        'run_id': spec['run_id'], 'queue': a.queue, 'engineering_only': True,
        'parent_completed_exit0': True, 'start_local_step': start, 'end_local_step': 2,
        'actual_new_updates': 2 - start, 'actual_question_exposures': cost['loss_calls'],
        'source_checkpoint_step': source_step, 'actual_loaded_adapter_and_rollout_engine_match_source_checkpoint': True,
        'one_continual_adapter_and_original_frozen_Base_verified': True,
        'actual_optimization_start': initial,
        'saved_checkpoint_optimizer_scheduler_RNG_loaded_on_CPU': True,
        'all_actual_microstep_losses_finite': True, 'tensorboard_events_written': True,
        'training_generated_tokens': generated, 'valid_response_loss_positions': cost['valid_response_loss_positions'],
        'teacher_forward_calls': cost['teacher_forward_calls'],
        'teacher_forward_cuda_seconds': cost['teacher_forward_cuda_seconds'],
        'training_loop_seconds': proof['train_seconds_including_diagnostic_hashing'],
        'parent_GPU_hours_counted_once': wall / 3600, 'source_spec_sha256': digest(a.config),
        'checkpoint_adapter_sha256': digest(model / 'checkpoint-2/adapter_model.safetensors'),
        'scope': 'Actual1.7B GPU engineering validation using existing SW20 parent. Same-language handoff only, not selected three-language training, disjoint-parent question evidence or mathematics efficacy. Resume preserves actual HF optimizer/scheduler/RNG and initial policy; online engine sampling is not promised bitwise identical. Data ID order/seed are saved; no claim that final resumed weights equal uninterrupted stochastic training.'}
    write_new(a.output, result)
    print(json.dumps({'new_updates': 2 - start, 'all_losses_finite': True,
        'source_policy_matches': True, 'parent_exit_code': 0}))


if __name__ == '__main__':
    main()
