"""Audit the completed second-seed BN clip intervention and English retention.

Read-only analysis: refuses incomplete parent pipelines and never submits jobs.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

from summarize_saved_candidates8192 import load_eval, pair, read


def load_live_target(root, queue, run, adapter):
    """Completed target stage, with its still-live English parent explicitly separate."""
    s = read(root / 'gpu-queue/queues' / queue / 'state.json')
    if s['phase'] == 'completed':
        return load_eval(root, queue, run, 'bn', adapter)
    assert s['phase'] == 'running' and s['job']['exit_code'] is None
    job = s['job']
    proc = Path('/proc') / str(job['pid'])
    assert proc.stat().st_uid == os.getuid()
    assert str(job['script']).encode() in proc.joinpath('cmdline').read_bytes().split(b'\x00')
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert hashlib.sha256(Path(job['script']).read_bytes()).hexdigest() == job['script_sha256']
    d = root / 'logs/copsd_official_10lang/development_vllm8192' / run
    c, m = read(d / 'CONFIG.json'), read(d / 'DONE.json')
    assert c['protocol'] == m['protocol'] == 'project_dev_single_vllm8192_v2'
    assert c['language'] == 'bn' and c['max_new_tokens'] == 8192 and c['samples'] == m['n'] == 1
    assert c['max_num_seqs'] == m['actual_max_num_seqs'] == 16
    assert c['checkpoint_step'] == 20 and Path(c['adapter']).resolve() == adapter.resolve()
    assert hashlib.sha256(Path(__file__).with_name('evaluate_dev8192.py').read_bytes()).hexdigest() == c['evaluator_sha256']
    rows = [json.loads(x) for x in (d / 'ITEMS.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == m['items'] == 100 and not m['formal_test']
    assert all(len(r['samples']) == 1 for r in rows)
    assert sum(r['samples'][0]['correct'] for r in rows) == m['single_correct']
    assert sum(len(r['samples'][0]['token_ids']) for r in rows) == m['generated_tokens']
    assert all(len(r['samples'][0]['token_ids']) == r['samples'][0]['tokens'] for r in rows)
    return c, m, rows, None  # A live parent has no completed allocated-clock cost.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-only', action='store_true',
                   help='Audit finished target stages while English parents remain live; never claim whole-pipeline completion.')
    a = p.parse_args()
    assert not a.output.exists()
    root = a.root
    base = load_eval(root, 'copsd-bn-base-dev8192-v1',
                     'bn_base_dev100_8192_20261020_v1', 'bn', None)
    english = load_eval(root, 'copsd-en-sw-base-dev8192-v1',
                        'en_sw_base_dev100_8192_20261020_v1', 'en', None)
    runs, costs, proofs, training_args = {}, {}, {}, {}
    import torch
    fields = ['learning_rate', 'max_grad_norm', 'per_device_train_batch_size',
              'gradient_accumulation_steps', 'max_steps', 'seed', 'data_seed',
              'optim', 'lr_scheduler_type', 'lr_scheduler_kwargs', 'warmup_steps',
              'warmup_ratio', 'weight_decay', 'adam_beta1', 'adam_beta2',
              'adam_epsilon', 'gradient_checkpointing', 'gradient_checkpointing_kwargs',
              'bf16', 'fp16', 'max_length', 'max_completion_length']
    for mode in ['clip', 'unclip']:
        name = f'bn_17b_{mode}20_horizon100_seed43_20261020_v2'
        queue = f'copsd-bn-{mode}20-seed43-dev8192-v2'
        model = root / 'models/checkpoint/copsd_bn_clip_control_v1' / name
        cp = model / 'checkpoint-20'
        if a.target_only:
            target = load_live_target(root, queue, name + '_dev8192', cp)
            runs[mode] = {'target': target}
        else:
            target = load_eval(root, queue, name + '_dev8192', 'bn', cp)
            en = load_eval(root, queue, 'en_' + name + '_dev8192', 'en', cp)
            runs[mode] = {'target': target, 'English': en}
            assert target[3] == en[3]  # Same parent clock, count it once.
        proof = read(model / 'COLLOCATE_TRAIN_UPDATES_DONE.json')
        cost = read(model / 'COST_SEGMENT_DONE.json')
        actual = read(model / 'ABLATION_ACTUAL.json')
        assert proof['global_step'] == 20 and proof['scheduler_horizon'] == 100
        assert proof['base_unchanged_bitwise']
        assert proof['micro_steps'] == cost['loss_calls'] == 640
        assert proof['max_completion_length'] == 2048 and proof['effective_batch'] == 32
        assert cost['loss_temperature'] == 1.1
        assert cost['seed_applied_before_model_and_PEFT'] == 43
        assert len(proof['syncs']) == 21
        assert read(model / 'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
        assert actual['beta'] == 0 and not actual['core_loss_replaced']
        assert actual['actual_component_clip'] == (.05 if mode == 'clip' else None)
        args = torch.load(cp / 'training_args.bin', map_location='cpu', weights_only=False)
        assert args.seed == args.data_seed == 43
        training_args[mode] = args
        assert torch.load(cp / 'scheduler.pt', map_location='cpu', weights_only=False)['last_epoch'] == 20
        optimizer = torch.load(cp / 'optimizer.pt', map_location='cpu', weights_only=False)
        assert {int(v['step']) for v in optimizer['state'].values()} == {20}
        rng = torch.load(cp / 'rng_state.pth', map_location='cpu', weights_only=False)
        assert all(k in rng for k in ['python', 'numpy', 'cpu', 'cuda'])
        groups = read(model / 'UPDATE_BATCH_GENERATION.json')['groups']
        assert len(groups) == 20 and [g['global_step'] for g in groups] == list(range(20))
        assert sum(g['rollouts'] for g in groups) == 640
        generated = sum(g['generated_tokens'] for g in groups)
        assert 0 < cost['valid_response_loss_positions'] <= generated
        costs[mode] = {
            'updates': 20, 'scheduler_horizon': 100, 'question_exposures': 640,
            'training_generated_tokens': generated,
            'valid_response_loss_positions': cost['valid_response_loss_positions'],
            'training_clock_seconds': proof['train_seconds_including_diagnostic_hashing'],
            'teacher_forward_cuda_seconds': cost['teacher_forward_cuda_seconds'],
            'student_forward_cuda_seconds': cost['student_forward_cuda_seconds'],
            'pipeline_GPU_hours_training_target_English_once': target[3],
        }
        proofs[mode] = proof
    for field in fields:
        assert getattr(training_args['clip'], field) == getattr(training_args['unclip'], field), field
    for key in ['adapter_hash', 'engine_hash']:
        assert proofs['clip']['syncs'][0][key] == proofs['unclip']['syncs'][0][key]
    pairs = {mode: {'target': pair(base, rr['target'])} for mode, rr in runs.items()}
    if not a.target_only:
        for mode, rr in runs.items():
            pairs[mode]['English'] = pair(english, rr['English'])
    intervention = pair(runs['clip']['target'], runs['unclip']['target'])
    new_metrics = [x[1] for rr in runs.values() for x in rr.values()]
    out = {
        'observed_at': datetime.datetime.now().isoformat(),
        'protocol': 'project_dev_single_vllm8192_v2',
        'training_language': 'bn', 'training_seed': 43, 'evaluation_seed': 42,
        'target_stage_only': a.target_only,
        'English_and_complete_parent_exit_required': a.target_only,
        'paired_development100': pairs,
        'unclipped_vs_clipped_target': {
            'gained': intervention['gained'], 'lost': intervention['lost'],
            'delta_correct': intervention['delta_correct']},
        'metrics': {mode: {lang: x[1] for lang, x in rr.items()} for mode, rr in runs.items()},
        'cached_Base_metrics': {'BN': base[1], 'English': english[1]},
        'training_and_pipeline_cost': costs,
        'new_evaluation_cost': {
            'responses': len(new_metrics) * 100,
            'generated_tokens': sum(m['generated_tokens'] for m in new_metrics),
            'generation_seconds': sum(m['generation_seconds'] for m in new_metrics),
            'new_pipeline_GPU_hours': None if a.target_only else sum(c['pipeline_GPU_hours_training_target_English_once'] for c in costs.values()),
            'cached_Base_costs_excluded': True,
            'scope': 'Two unique parent clocks include training plus both evaluations; never sum the same parent once per evaluation.'},
        'parents_completed_exit0': not a.target_only,
        'completed_target_stages_scientific_args_initial_weights_recovery_state_verified': True,
        'scope': 'Second training-seed replication of the BN component-clip intervention; seed changes adapter initialization and training order. Same independent development100, not an additional test set or formal-test result. Policies may generate different trajectories and token costs. English retention must be reported beside target gains.'}
    with a.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(pairs, ensure_ascii=False))


if __name__ == '__main__':
    main()
