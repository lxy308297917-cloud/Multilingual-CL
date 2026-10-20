"""Verify completed beta0/beta1 controls with clipping disabled in both."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists()
    import torch
    names = ['sw_17b_prefix20_horizon100_unclipped_seed42_20261020_v1',
             'sw_17b_reversekl_prefix20_horizon100_seed42_20261020_v1']
    tracks = ['copsd_clip_ablation_v1', 'copsd_kl_direction_ablation_v1']
    queues = ['copsd-sw-clip-ablation20-v1', 'copsd-sw-reversekl20-v1']
    proofs, arguments, adapters, costs = [], [], [], []
    for track, name, queue in zip(tracks, names, queues):
        s = read(a.root/'gpu-queue/queues'/queue/'state.json')
        assert s['phase'] == 'completed' and s['job']['exit_code'] == 0
        j = s['job']
        assert hashlib.sha256(Path(j['script']).read_bytes()).hexdigest() == j['script_sha256']
        p = a.root/'models/checkpoint'/track/name
        proof = read(p/'COLLOCATE_TRAIN_UPDATES_DONE.json')
        cost = read(p/'COST_SEGMENT_DONE.json')
        assert proof['global_step'] == 20 and proof['scheduler_horizon'] == 100
        assert proof['micro_steps'] == cost['loss_calls'] == 640 and proof['base_unchanged_bitwise']
        assert proof['effective_batch'] == 32 and proof['max_completion_length'] == 2048
        assert len(proof['syncs']) == 21
        assert read(p/'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
        cp = p/'checkpoint-20'
        assert read(cp/'trainer_state.json')['global_step'] == 20
        for f in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_model.safetensors']:
            assert (cp/f).stat().st_size > 0
        arguments.append(torch.load(cp/'training_args.bin', map_location='cpu', weights_only=False))
        adapter = read(cp/'adapter_config.json')
        for field in ['target_modules', 'exclude_modules', 'modules_to_save']:
            if isinstance(adapter.get(field), list): adapter[field] = sorted(adapter[field])
        adapters.append(adapter)
        groups = read(p/'UPDATE_BATCH_GENERATION.json')['groups']
        assert len(groups) == 20 and sum(g['rollouts'] for g in groups) == 640
        tokens = sum(g['generated_tokens'] for g in groups)
        assert 0 < cost['valid_response_loss_positions'] <= tokens
        assert cost['seed_applied_before_model_and_PEFT'] == 42 and cost['loss_temperature'] == 1.1
        costs.append({'updates': 20, 'scheduler_horizon': 100, 'question_exposures': 640,
            'generated_tokens': tokens, 'valid_response_loss_positions': cost['valid_response_loss_positions'],
            'training_seconds': proof['train_seconds_including_diagnostic_hashing'],
            'teacher_forward_cuda_seconds': cost['teacher_forward_cuda_seconds'],
            'student_forward_cuda_seconds': cost['student_forward_cuda_seconds']})
        proofs.append(proof)
    assert adapters[0] == adapters[1]
    assert arguments[0].beta == 0 and arguments[1].beta == 1
    assert arguments[0].temperature == arguments[1].temperature == 1.1
    for field in ['learning_rate', 'max_grad_norm', 'per_device_train_batch_size',
                  'gradient_accumulation_steps', 'max_steps', 'seed', 'data_seed', 'optim',
                  'lr_scheduler_type', 'lr_scheduler_kwargs', 'warmup_steps', 'warmup_ratio',
                  'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'gradient_checkpointing',
                  'gradient_checkpointing_kwargs', 'bf16', 'fp16', 'max_length', 'max_completion_length']:
        assert getattr(arguments[0], field) == getattr(arguments[1], field), field
    for key in ['adapter_hash', 'engine_hash']:
        assert proofs[0]['syncs'][0][key] == proofs[1]['syncs'][0][key]
    actual = read(a.root/'models/checkpoint'/tracks[1]/names[1]/'ABLATION_ACTUAL.json')
    assert actual['actual_beta'] == 1 and actual['actual_component_clip'] is None and not actual['core_loss_replaced']
    dev = a.root/'logs/copsd_official_10lang/development_vllm'
    suites = {
        'sw': ['sw_17b_dev100_vllm_base_20261020_v1',
               'sw_17b_dev100_vllm_prefix20_unclipped_seed42_20261020_v1',
               'sw_17b_dev100_vllm_reversekl20_seed42_20261020_v1'],
        'en': ['en_17b_dev100_vllm_base_20261020_v1',
               'en_17b_dev100_vllm_sw_unclipped20_seed42_20261020_v1',
               'en_17b_dev100_vllm_sw_reversekl20_seed42_20261020_v1']}
    evaluations = {}
    for lang, runs in suites.items():
        configs = [read(dev/n/'CONFIG.json') for n in runs]
        cc = [{k: v for k, v in c.items() if k not in ['adapter', 'output']} for c in configs]
        assert cc[0] == cc[1] == cc[2]
        for c, track, name in zip(configs[1:], tracks, names):
            assert Path(c['adapter']).resolve() == (a.root/'models/checkpoint'/track/name/'checkpoint-20').resolve()
        rows = [[json.loads(s) for s in (dev/n/'ITEMS.jsonl').read_text().splitlines()] for n in runs]
        metrics = [read(dev/n/'DONE.json') for n in runs]
        for rr, m in zip(rows, metrics):
            assert len(rr) == len({r['id'] for r in rr}) == m['items'] == 100
            assert sum(r['samples'][0]['correct'] for r in rr) == m['single_correct']
            assert sum(len(r['samples'][0]['token_ids']) for r in rr) == m['generated_tokens']
        for triple in zip(*rows):
            for key in ['id', 'question', 'answer', 'prompt']:
                assert triple[0][key] == triple[1][key] == triple[2][key]
            assert len({r['samples'][0]['seed'] for r in triple}) == 1
        bc, rc = [[r['samples'][0]['correct'] for r in rr] for rr in rows[1:]]
        evaluations[lang] = {'correct_out_of100': dict(zip(['Base', 'beta0', 'beta1'], [m['single_correct'] for m in metrics])),
            'metrics': dict(zip(['Base', 'beta0', 'beta1'], metrics)),
            'beta1_vs_beta0': {'gained': sum(not x and y for x, y in zip(bc, rc)),
                              'lost': sum(x and not y for x, y in zip(bc, rc))}}
    result = {'observed_at': datetime.datetime.now().isoformat(), 'protocol': 'project_SW_KL_direction_one_factor_v1',
        'evaluations': evaluations, 'training_cost': dict(zip(['beta0', 'beta1'], costs)),
        'initial_adapter_inference_weights_args_adapter_configs_and_dev_pairs_verified': True,
        'actual_beta1_no_clip_verified': True, 'new_core_loss_implementation': False,
        'scope': 'Single seed,20 updates of100-step scheduler,640 exposures, independent SW/EN development100. Project ablation, not published beta0/clip0.05 reproduction, stable multilingual efficacy or sequential forgetting.'}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v['correct_out_of100'] for k, v in evaluations.items()}))


if __name__ == '__main__':
    main()
