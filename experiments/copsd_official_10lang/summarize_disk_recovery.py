"""Read-only audit of one completed disk-recovery job and its actual costs.

Cached stages from failed parents are explicitly identified; they are not
misrepresented as new generation or as successful old parent pipelines.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from disk_recovery_driver import audit_stage, digest, read
from summarize_saved_candidates8192 import pair


def parent_clock(job):
    start = datetime.datetime.fromisoformat(job['started_at'])
    end = datetime.datetime.fromisoformat(job['finished_at'])
    assert end >= start
    return (end - start).total_seconds() / 3600


def interrupted_parent(root, queue):
    folder = root / 'logs/copsd_official_10lang/disk_full_20261020/queue_snapshots'
    state = read(folder / (queue + '.STATE.json'))
    job = state.get('job')
    if not job:
        assert not state.get('held')
        return {'allocated_GPU_hours_approx': 0., 'waiting_without_allocated_GPU': True}
    records = [json.loads(x) for x in (folder / (queue + '.JOURNAL.jsonl')).read_text().splitlines()]
    failed = [r for r in records if 'Failed with result' in str(r.get('MESSAGE', ''))]
    assert failed and len(job['gpu_uuids']) == 1
    event = max(int(r['__REALTIME_TIMESTAMP']) for r in failed)
    start = datetime.datetime.fromisoformat(job['started_at'])
    if start.tzinfo is None:
        start = start.replace(tzinfo=ZoneInfo('Asia/Shanghai'))
    end = datetime.datetime.fromtimestamp(event / 1e6, tz=datetime.timezone.utc)
    assert end >= start
    return {'allocated_GPU_hours_approx': (end - start).total_seconds() / 3600,
            'waiting_without_allocated_GPU': False,
            'method': 'Old job start to archived service-failure event, one GPU. Approximate allocation clock; old child normal exit was not collected.',
            'unserialized_generation_work_not_reconstructed': True,
            'old_parent_success_claimed': False}


def audit_ja_training(op, root):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    model = Path(op['new_model_directory'])
    cp = model / 'checkpoint-100'
    proof = read(model / 'COLLOCATE_TRAIN_UPDATES_DONE.json')
    cost = read(model / 'COST_SEGMENT_DONE.json')
    assert read(cp / 'trainer_state.json')['global_step'] == 100
    assert proof['start_step'] == cost['start_step'] == 80
    assert proof['global_step'] == cost['global_step'] == 100
    assert proof['completed_updates'] == 20 and proof['micro_steps'] == cost['loss_calls'] == 628
    assert proof['base_unchanged_bitwise'] and proof['effective_batch'] == 32
    assert proof['max_completion_length'] == 2048 and cost['loss_temperature'] == 1.1
    assert [s['global_step'] for s in proof['syncs']] == list(range(80, 101))
    reference = read(root / 'logs/copsd_official_10lang/ja80_parent_policy_20261020.PRIVATE.json')
    for key in ['adapter_hash', 'engine_hash']:
        assert proof['syncs'][0][key] == reference['syncs'][0][key]
    assert read(model / 'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
    weights = load_file(str(cp / 'adapter_model.safetensors'), device='cpu')
    assert weights and all(torch.isfinite(t).all() for t in weights.values())
    optimizer = torch.load(cp / 'optimizer.pt', map_location='cpu', weights_only=False)
    assert {int(v['step']) for v in optimizer['state'].values()} == {100}
    assert all(torch.isfinite(t).all() for state in optimizer['state'].values()
               for t in state.values() if isinstance(t, torch.Tensor))
    assert torch.load(cp / 'scheduler.pt', map_location='cpu', weights_only=False)['last_epoch'] == 100
    rng = torch.load(cp / 'rng_state.pth', map_location='cpu', weights_only=False)
    assert all(k in rng for k in ['python', 'numpy', 'cpu', 'cuda']) and rng['cpu'].numel() > 0
    saved = torch.load(cp / 'training_args.bin', map_location='cpu', weights_only=False)
    parent = Path(op['commands'][0][-1])
    prior = torch.load(parent / 'training_args.bin', map_location='cpu', weights_only=False)
    fields = ['learning_rate', 'max_grad_norm', 'per_device_train_batch_size',
              'gradient_accumulation_steps', 'max_steps', 'seed', 'data_seed', 'optim',
              'lr_scheduler_type', 'lr_scheduler_kwargs', 'warmup_steps', 'warmup_ratio',
              'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon',
              'gradient_checkpointing', 'gradient_checkpointing_kwargs', 'bf16', 'fp16',
              'max_length', 'max_completion_length']
    assert all(getattr(saved, f) == getattr(prior, f) for f in fields)
    assert digest(cp / 'adapter_model.safetensors') == digest(model / 'adapter_model.safetensors')
    groups = read(model / 'UPDATE_BATCH_GENERATION.json')['groups']
    assert [g['global_step'] for g in groups] == list(range(80, 100))
    assert sum(g['rollouts'] for g in groups) == 628
    generated = sum(g['generated_tokens'] for g in groups)
    assert 0 < cost['valid_response_loss_positions'] <= generated
    return {'new_updates': 20, 'start_step': 80, 'global_step': 100,
            'new_question_exposures': 628, 'new_generated_tokens': generated,
            'new_valid_response_loss_positions': cost['valid_response_loss_positions'],
            'new_training_clock_seconds': proof['train_seconds_including_diagnostic_hashing'],
            'new_teacher_forward_cuda_seconds': cost['teacher_forward_cuda_seconds'],
            'new_student_forward_cuda_seconds': cost['student_forward_cuda_seconds'],
            'actual_initial_student_and_rollout_sync_match_saved80': True,
            'final_adapter_actually_loaded_and_finite': True,
            'final_optimizer_scheduler_RNG_actually_loaded': True,
            'saved_scientific_fields_match_parent80': fields,
            'frozen_Base_unchanged_new_segment': True,
            'final_committed_exposure_chain': 2560 + 628,
            'known_discarded_completed_updates': 2,
            'known_attempted_loss_calls_lower_bound_including_discarded_work': 2624 + 628,
            'sampling_stream_bitwise_continuation_promised': False,
            'scope': 'New80-to100 process only. Old saved80 plus new20 commits3188 exposures; actual interrupted work additionally includes at least64 discarded loss calls and unreconstructed rollout work.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--operation', choices=['ja80', 'th', 'te', 'ja20'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    code = Path(__file__).parent
    lock = read(code / 'disk_recovery_runtime_20261020.CONFIG.json')
    op = lock['operations'][a.operation]
    state = read(a.root / 'gpu-queue/queues' / op['queue'] / 'state.json')
    if state['phase'] != 'completed' or (state.get('job') or {}).get('exit_code') != 0:
        raise RuntimeError('Recovery parent has not completed exit0; stale running state and CPU preflights are not results')
    job = state['job']
    assert Path(job['script']).resolve() == (code / 'run_disk_recovery.sh').resolve()
    assert job['config']['job']['args'] == [a.operation]
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert job['script_sha256'] == digest(job['script'])
    for name, sha in lock['code_sha256'].items():
        assert digest(code / name) == sha, name
    for name, sha in op.get('frozen_file_sha256', {}).items():
        assert digest(name) == sha, name
    stages = {name: audit_stage(spec, a.root, code) for name, spec in op['result_stages'].items()}
    comparisons = {'target': pair(stages['base'], stages['target'])}
    if 'English' in stages:
        comparisons['English'] = pair(stages['English_Base'], stages['English'])
    for records in zip(*[v[2] for v in stages.values()]):
        assert len({r['id'] for r in records}) == len({str(r['answer']) for r in records}) == 1
    assert read(op['completion_marker'])['paired_scores'] == comparisons
    # The shared development helper names its trained score adapter20_correct.
    # JA80 recovery ends at100; preserve the frozen marker audit, then label
    # the public report with the actual evaluated checkpoint step.
    if a.operation == 'ja80':
        # The older2048 evaluator has no checkpoint_step config field.
        # Verify the actual evaluated adapter path and its saved Trainer state.
        evaluated = Path(stages['target'][0]['adapter']).resolve()
        assert evaluated == (Path(op['new_model_directory']) / 'checkpoint-100').resolve()
        assert read(evaluated / 'trainer_state.json')['global_step'] == 100
        comparisons['target']['adapter100_correct'] = comparisons['target'].pop('adapter20_correct')
        comparisons['target']['checkpoint_step'] = 100
    new_roles = {'ja80': ['target'], 'th': ['English'], 'te': ['base', 'target', 'English'],
                 'ja20': ['base', 'target', 'English']}[a.operation]
    responses, tokens, seconds = 0, 0, 0.
    reused = {name: 100 for name in stages if name not in new_roles}
    provenance = None
    for name in new_roles:
        m = stages[name][1]
        if a.operation == 'te' and name == 'base':
            provenance = read(Path(op['result_stages']['base']['directory']) / 'RECOVERY.PROVENANCE.json')
            assert provenance['cached_records_preserved_exactly'] and provenance['reused_items'] == 73
            assert provenance['new_generated_items'] == 27
            assert provenance['old_partial_items_sha256'] == op['partial_Base_sha256']['ITEMS.jsonl']
            assert provenance['wrapper_sha256'] == digest(code / 'resume_te_base8192.py')
            responses += 27
            tokens += provenance['new_generated_tokens']
            seconds += provenance['new_generation_seconds']
            reused['base_partial'] = 73
        else:
            responses += m['items']
            tokens += m['generated_tokens']
            seconds += m['generation_seconds']
    old = interrupted_parent(a.root, op['queue'])
    output = {'observed_at': datetime.datetime.now().isoformat(), 'operation': a.operation,
        'new_parent_identity_for_cost_deduplication': {'queue': op['queue'],
            'started_at': job['started_at'], 'script_sha256': job['script_sha256']},
        'old_cost_reference_not_additional_compute': 'disk_failure_cost_audit_20261020.PUBLIC.json#' + a.operation,
        'protocol': stages['target'][0]['protocol'], 'paired_development100': comparisons,
        'metrics': {name: value[1] for name, value in stages.items()},
        'new_generation_cost': {'new_responses': responses, 'new_generated_tokens': tokens,
            'new_generation_seconds': seconds, 'reused_responses_excluded': reused},
        'new_unique_parent_allocated_GPU_hours': parent_clock(job),
        'old_interrupted_parent_cost': old,
        'old_plus_recovery_parent_GPU_hours_approx': old['allocated_GPU_hours_approx'] + parent_clock(job),
        'new_parent_exit0_scientific_sources_inputs_prompts_seeds_counts_and_marker_verified': True,
        'cached_English_Base_cost_not_added_again': True,
        'TE_partial_cost_uses_only_new27': provenance is not None,
        'new_training': audit_ja_training(op, a.root) if a.operation == 'ja80' else False,
        'scope': 'Recovered development report, not new formal tests, stable final language selection or sequential results. Old failed parent costs are approximated from failure events, not normal exits. Cached outputs retain their old cost attribution; TE full100 scored-token totals are not all new tokens. Do not sum this report with its own overlapping preflight/partial reports as additional compute.'}
    with a.output.open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(comparisons))


if __name__ == '__main__':
    main()
