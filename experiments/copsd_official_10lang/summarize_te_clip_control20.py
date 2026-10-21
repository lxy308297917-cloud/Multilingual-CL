"""Read-only TE clip intervention audit; requires both complete training/eval parents."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from summarize_saved_candidates8192 import load_eval, pair, read


def collect(root):
    import torch
    code = Path(__file__).parent
    frozen = read(code / 'te_clip_control20_frozen_20261021.PUBLIC.json')
    for name, expected in frozen['code_sha256'].items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((root / frozen['training_data_relative']).read_bytes()).hexdigest() == frozen['training_data_sha256']
    old = read(code / 'remaining_saved20_screen8192_frozen_20261020.PUBLIC.json')['languages']['te']
    base = load_eval(root, old['queue'], 'te_saved20_screen_base_recovered_dev100_8192_20261020_v2', 'te', None)
    english = load_eval(root, 'copsd-en-sw-base-dev8192-v1', 'en_sw_base_dev100_8192_20261020_v1', 'en', None)
    runs, proofs, costs, arguments, adapters, sources = {}, {}, {}, {}, {}, {}
    fields = ['learning_rate', 'max_grad_norm', 'per_device_train_batch_size',
              'gradient_accumulation_steps', 'max_steps', 'seed', 'data_seed', 'optim',
              'lr_scheduler_type', 'lr_scheduler_kwargs', 'warmup_steps', 'warmup_ratio',
              'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon',
              'gradient_checkpointing', 'gradient_checkpointing_kwargs',
              'bf16', 'fp16', 'max_length', 'max_completion_length']
    for mode, spec in frozen['conditions'].items():
        model = root / spec['checkpoint_output_relative']
        cp = model / 'checkpoint-20'
        # load_eval rejects live, failed or incomplete parents before any score is reported.
        target = load_eval(root, spec['queue'], spec['evaluation_run_ids']['te'], 'te', cp)
        en = load_eval(root, spec['queue'], spec['evaluation_run_ids']['en'], 'en', cp)
        assert target[3] == en[3]
        proof = read(model / 'COLLOCATE_TRAIN_UPDATES_DONE.json')
        cost = read(model / 'COST_SEGMENT_DONE.json')
        actual = read(model / 'ABLATION_ACTUAL.json')
        assert actual['protocol'] == frozen['protocol'] and actual['language'] == 'te'
        assert actual['actual_component_clip'] == spec['component_clip']
        assert actual['beta'] == 0 and not actual['core_loss_replaced']
        assert proof['global_step'] == 20 and proof['scheduler_horizon'] == 100
        assert proof['micro_steps'] == cost['loss_calls'] == 640
        assert proof['base_unchanged_bitwise'] and len(proof['syncs']) == 21
        assert proof['max_completion_length'] == 2048 and proof['effective_batch'] == 32
        assert cost['loss_temperature'] == 1.1 and cost['seed_applied_before_model_and_PEFT'] == 42
        assert read(model / 'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
        arguments[mode] = torch.load(cp / 'training_args.bin', map_location='cpu', weights_only=False)
        args = arguments[mode]
        assert args.seed == args.data_seed == 42 and args.max_steps == 100
        assert args.learning_rate == 5e-6 and args.max_grad_norm == .1
        assert torch.load(cp / 'scheduler.pt', map_location='cpu', weights_only=False)['last_epoch'] == 20
        optimizer = torch.load(cp / 'optimizer.pt', map_location='cpu', weights_only=False)
        assert {int(v['step']) for v in optimizer['state'].values()} == {20}
        rng = torch.load(cp / 'rng_state.pth', map_location='cpu', weights_only=False)
        assert all(k in rng for k in ['python', 'numpy', 'cpu', 'cuda'])
        adapter = read(cp / 'adapter_config.json')
        for key in ['target_modules', 'exclude_modules', 'modules_to_save']:
            if isinstance(adapter.get(key), list): adapter[key] = sorted(adapter[key])
        adapters[mode] = adapter
        groups = read(model / 'UPDATE_BATCH_GENERATION.json')['groups']
        assert [g['global_step'] for g in groups] == list(range(20))
        assert sum(g['rollouts'] for g in groups) == 640
        tokens = sum(g['generated_tokens'] for g in groups)
        assert 0 < cost['valid_response_loss_positions'] <= tokens
        runs[mode] = {'target': target, 'English': en}
        proofs[mode] = proof
        costs[mode] = {'updates': 20, 'question_exposures': 640, 'generated_training_tokens': tokens,
                       'valid_response_loss_positions': cost['valid_response_loss_positions'],
                       'training_clock_seconds': proof['train_seconds_including_diagnostic_hashing'],
                       'teacher_forward_cuda_seconds': cost['teacher_forward_cuda_seconds'],
                       'student_forward_cuda_seconds': cost['student_forward_cuda_seconds'],
                       'pipeline_GPU_hours_training_and_both_evals_counted_once': target[3]}
        sources[mode] = {lang: {'run_id': name, 'items_sha256': hashlib.sha256((root / 'logs/copsd_official_10lang/development_vllm8192' / name / 'ITEMS.jsonl').read_bytes()).hexdigest()} for lang, name in spec['evaluation_run_ids'].items()}
    for field in fields:
        assert getattr(arguments['clip'], field) == getattr(arguments['unclip'], field), field
    assert adapters['clip'] == adapters['unclip']
    for key in ['adapter_hash', 'engine_hash']:
        assert proofs['clip']['syncs'][0][key] == proofs['unclip']['syncs'][0][key]
    for records in zip(base[2], english[2], *[ev[2] for rr in runs.values() for ev in rr.values()]):
        assert len({r['id'] for r in records}) == len({r['answer'] for r in records}) == 1
    pairs = {mode: {'target': pair(base, rr['target']), 'English': pair(english, rr['English'])} for mode, rr in runs.items()}
    interventions = {lang: pair(runs['clip'][lang], runs['unclip'][lang]) for lang in ['target', 'English']}
    metrics = [ev[1] for rr in runs.values() for ev in rr.values()]
    return {'observed_at': datetime.datetime.now().isoformat(), 'protocol': frozen['protocol'],
            'parents_completed_exit0': True, 'training_seed': 42, 'paired_development100': pairs,
            'unclipped_vs_clipped': interventions,
            'metrics': {mode: {lang: ev[1] for lang, ev in rr.items()} for mode, rr in runs.items()},
            'cached_Base_metrics': {'TE': base[1], 'English': english[1]},
            'training_and_pipeline_cost': costs, 'source_records': sources,
            'new_evaluation_cost': {'responses': 400, 'generated_tokens': sum(m['generated_tokens'] for m in metrics),
                                    'generation_seconds': sum(m['generation_seconds'] for m in metrics),
                                    'cached_Base_cost_excluded': True,
                                    'pipeline_GPU_hours': sum(x['pipeline_GPU_hours_training_and_both_evals_counted_once'] for x in costs.values())},
            'fixed_Base_initial_policy_recovery_state_and_scientific_args_verified': True,
            'scope': frozen['scope'], 'interpretation_limits': frozen['decision_policy']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    out = collect(a.root)
    with a.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(out['paired_development100'], ensure_ascii=False))


if __name__ == '__main__': main()
