"""Collect completed ZH/JA seed43 repeats; never select languages or submit jobs."""
import argparse
import datetime
import json
from pathlib import Path

from disk_recovery_driver import audit_stage, digest, read
from summarize_base8192 import RUNS as BASE_RUNS
from summarize_saved_candidates8192 import load_eval, pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--language', choices=['zh', 'ja'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    root, lang, code = a.root, a.language, Path(__file__).parent
    frozen = read(code / f'{lang}_seed43_prefix20_frozen_20261021.PUBLIC.json')
    for name, sha in frozen['code_sha256'].items():
        assert digest(code / name) == sha, name
    queue = f'copsd-{lang}-seed43-prefix20-dev8192-v1'
    track = 'copsd_prefix20_replicates_v1' if lang == 'zh' else 'copsd_ja_prefix20_replicates_v1'
    suffix = '' if lang == 'zh' else '_repaired'
    model = root / 'models/checkpoint' / track / f'{lang}_17b_prefix20_horizon100_seed43{suffix}_20261021_v1'
    cp = model / 'checkpoint-20'
    folder = root / 'logs/copsd_official_10lang/development_vllm8192'
    target_id = f'{lang}_17b_seed43_prefix20_dev100_8192_20261021_v1'
    en_id = f'en_{lang}_seed43_prefix20_dev100_8192_20261021_v1'
    target = load_eval(root, queue, target_id, lang, cp)
    English = load_eval(root, queue, en_id, 'en', cp)
    assert target[3] == English[3]  # One parent pipeline, counted once.

    def audited(run_id, language, adapter, step):
        return audit_stage({'directory': str(folder / run_id), 'language': language,
            'adapter': adapter, 'step': step, 'protocol': 'project_dev_single_vllm8192_v2',
            'budget': 8192, 'evaluator': 'evaluate_dev8192.py'}, root, code)

    target = (*audited(target_id, lang, str(cp), 20)[:3], target[3])
    English = (*audited(en_id, 'en', str(cp), 20)[:3], English[3])
    base = audited(BASE_RUNS[lang], lang, None, None)
    base_en = audited(BASE_RUNS['en'], 'en', None, None)
    for group in zip(base[2], target[2], base_en[2], English[2]):
        assert len({r['id'] for r in group}) == len({str(r['answer']) for r in group}) == 1
    proof, cost = [read(model / name) for name in ['COLLOCATE_TRAIN_UPDATES_DONE.json', 'COST_SEGMENT_DONE.json']]
    assert proof['start_step'] == 0 and proof['global_step'] == proof['completed_updates'] == 20
    assert proof['scheduler_horizon'] == 100 and proof['stop_after_updates'] == 20
    assert proof['micro_steps'] == cost['loss_calls'] == cost['student_forward_calls'] == cost['teacher_forward_calls'] == 640
    assert proof['effective_batch'] == 32 and proof['max_completion_length'] == 2048
    assert proof['base_unchanged_bitwise'] and len(proof['syncs']) == 21
    assert cost['seed_applied_before_model_and_PEFT'] == 43 and cost['loss_temperature'] == 1.1
    assert read(model / 'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
    reference_model = root / 'models/checkpoint/copsd_prefix20_replicates_v1/sw_17b_prefix20_horizon100_seed43_20261020_v1'
    initial = read(reference_model / 'COLLOCATE_TRAIN_UPDATES_DONE.json')['syncs'][0]
    assert all(proof['syncs'][0][key] == initial[key] for key in ['adapter_hash', 'engine_hash'])
    import torch
    args = torch.load(cp / 'training_args.bin', map_location='cpu', weights_only=False)
    assert args.seed == args.data_seed == 43 and args.max_steps == 100
    assert args.learning_rate == 5e-6 and args.max_grad_norm == .1
    assert args.per_device_train_batch_size == 1 and args.gradient_accumulation_steps == 32
    reference_args = torch.load(reference_model / 'checkpoint-20/training_args.bin', map_location='cpu', weights_only=False)
    assert args.optim == reference_args.optim
    assert torch.load(cp / 'scheduler.pt', map_location='cpu', weights_only=False)['last_epoch'] == 20
    optimizer = torch.load(cp / 'optimizer.pt', map_location='cpu', weights_only=False)
    assert {int(v['step']) for v in optimizer['state'].values()} == {20}
    rng = torch.load(cp / 'rng_state.pth', map_location='cpu', weights_only=False)
    assert all(k in rng for k in ['python', 'numpy', 'cpu', 'cuda'])
    groups = read(model / 'UPDATE_BATCH_GENERATION.json')['groups']
    assert [g['global_step'] for g in groups] == list(range(20))
    assert all(g['rollouts'] == 32 for g in groups)
    generated = sum(g['generated_tokens'] for g in groups)
    assert 0 < cost['valid_response_loss_positions'] <= generated
    result = {'observed_at': datetime.datetime.now().isoformat(), 'training_language': lang,
        'training_seed': 43, 'evaluation_seed': 42, 'protocol': 'project_dev_single_vllm8192_v2',
        'parents_completed_exit0': True, 'paired_development100': {
            'target': pair(base, target), 'English': pair(base_en, English)},
        'metrics': {'target': target[1], 'English': English[1]},
        'cached_Base_metrics': {'target': base[1], 'English': base_en[1]},
        'actual_initial_policy_matches_cached_seed43_original_Base_LoRA': True,
        'frozen_Base_optimizer_scheduler_RNG_and_budget_verified': True,
        'actual_loss_settings': {'beta': 0, 'component_clip': .05,
            'scope': 'Inherited frozen official-core source and launcher; no loss intervention.'},
        'actual_optimizer': args.optim.value,
        'frozen_launcher_source_hashes_verified': True,
        'raw_items_sha256': {name: digest(folder / name / 'ITEMS.jsonl') for name in [target_id, en_id]},
        'checkpoint_adapter_sha256': digest(cp / 'adapter_model.safetensors'),
        'cost': {'updates': 20, 'scheduler_horizon': 100, 'question_exposures': 640,
            'training_generated_tokens': generated, 'valid_response_loss_positions': cost['valid_response_loss_positions'],
            'training_seconds_including_diagnostic_hashing': proof['train_seconds_including_diagnostic_hashing'],
            'teacher_forward_cuda_seconds': cost['teacher_forward_cuda_seconds'],
            'new_eval_responses': 200, 'new_eval_generated_tokens': target[1]['generated_tokens'] + English[1]['generated_tokens'],
            'new_eval_generation_seconds': target[1]['generation_seconds'] + English[1]['generation_seconds'],
            'parent_pipeline_GPU_hours_counted_once': target[3], 'cached_Base_cost_excluded': True},
        'scope': 'Controlled fresh-Base seed43 prefix20 follow-up on the same independent development100. Seed changes initialization and training order; historical seed42 references are not guaranteed identically seeded before PEFT. Not another independent test set, final language selection, a formal result or a sequential run. JA retains the established single-record literal-prefix repair.'}
    with a.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['paired_development100']))


if __name__ == '__main__':
    main()
