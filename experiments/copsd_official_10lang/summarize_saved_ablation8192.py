"""Verify matched saved-model development100 evaluation at8192 tokens."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--language', choices=['sw', 'en'], default='sw')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    root, lang = a.root, a.language
    training = {
        'clip20': 'copsd_prefix20_replicates_v1/sw_17b_prefix20_horizon100_seed42_20261020_v1',
        'unclip20': 'copsd_clip_ablation_v1/sw_17b_prefix20_horizon100_unclipped_seed42_20261020_v1',
        'reverse20': 'copsd_kl_direction_ablation_v1/sw_17b_reversekl_prefix20_horizon100_seed42_20261020_v1'}
    proofs = [read(root/'models/checkpoint'/name/'COLLOCATE_TRAIN_UPDATES_DONE.json') for name in training.values()]
    for proof in proofs:
        assert proof['global_step'] == 20 and proof['scheduler_horizon'] == 100
        assert proof['micro_steps'] == 640 and proof['base_unchanged_bitwise']
        assert len(proof['syncs']) == 21
    for key in ['adapter_hash', 'engine_hash']:
        assert len({proof['syncs'][0][key] for proof in proofs}) == 1
    labels = ['base', *training]
    configs, rows, metrics, clocks = {}, {}, {}, []
    for label in labels:
        # Queue names retain sw because all evaluated adapters were SW-trained.
        queue = f'copsd-sw-{label}-dev8192-v1' if lang == 'sw' else f'copsd-en-sw-{label}-dev8192-v1'
        s = read(root/'gpu-queue/queues'/queue/'state.json')
        assert s['phase'] == 'completed' and s['job']['exit_code'] == 0
        job = s['job']
        assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
        assert hashlib.sha256(Path(job['script']).read_bytes()).hexdigest() == job['script_sha256']
        run = f'sw_{label}_dev100_8192_20261020_v1' if lang == 'sw' else f'en_sw_{label}_dev100_8192_20261020_v1'
        d = root/'logs/copsd_official_10lang/development_vllm8192'/run
        cfg, m = read(d/'CONFIG.json'), read(d/'DONE.json')
        assert cfg['protocol'] == m['protocol'] == 'project_dev_single_vllm8192_v2'
        assert cfg['language'] == lang and cfg['items'] == m['items'] == 100
        assert cfg['samples'] == m['n'] == 1 and cfg['max_new_tokens'] == 8192
        assert cfg['max_num_seqs'] == m['actual_max_num_seqs'] == 16
        assert hashlib.sha256(Path(__file__).with_name('evaluate_dev8192.py').read_bytes()).hexdigest() == cfg['evaluator_sha256']
        assert not m['formal_test']
        if label == 'base':
            assert cfg['adapter'] is None and cfg['checkpoint_step'] is None
        else:
            expected = root/'models/checkpoint'/training[label]/'checkpoint-20'
            assert Path(cfg['adapter']).resolve() == expected.resolve() and cfg['checkpoint_step'] == 20
        rr = [json.loads(x) for x in (d/'ITEMS.jsonl').read_text().splitlines()]
        assert len(rr) == len({r['id'] for r in rr}) == 100
        assert all(len(r['samples']) == 1 for r in rr)
        assert sum(r['samples'][0]['correct'] for r in rr) == m['single_correct']
        assert sum(len(r['samples'][0]['token_ids']) for r in rr) == m['generated_tokens']
        assert all(len(r['samples'][0]['token_ids']) == r['samples'][0]['tokens'] for r in rr)
        configs[label], rows[label], metrics[label] = cfg, rr, m
        clocks.append((datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()/3600)
    canonical = [{k: v for k, v in c.items() if k not in ['adapter', 'checkpoint_step', 'output']} for c in configs.values()]
    assert all(x == canonical[0] for x in canonical)
    for quartet in zip(*rows.values()):
        for key in ['id', 'question', 'answer', 'prompt']:
            assert len({r[key] for r in quartet}) == 1
        assert len({r['samples'][0]['seed'] for r in quartet}) == 1
    comparisons = {}
    for first, second in [('base', 'clip20'), ('base', 'unclip20'), ('base', 'reverse20'),
                          ('clip20', 'unclip20'), ('unclip20', 'reverse20')]:
        x, y = [[r['samples'][0]['correct'] for r in rows[label]] for label in [first, second]]
        comparisons[first+'_to_'+second] = {'gained': sum(not i and j for i, j in zip(x, y)),
            'lost': sum(i and not j for i, j in zip(x, y)),
            'delta_correct': metrics[second]['single_correct']-metrics[first]['single_correct']}
    report = {'observed_at': datetime.datetime.now().isoformat(), 'protocol': 'project_dev_single_vllm8192_v2',
        'language': lang, 'training_language': 'sw', 'correct_out_of100': {k: v['single_correct'] for k, v in metrics.items()},
        'metrics': metrics, 'paired_changes': comparisons,
        'all_parent_queues_exit0_and_actual_scheduler_verified': True,
        'initial_training_adapter_inference_policy_and_20update_budget_matched': True,
        'evaluation_inputs_prompts_seeds_configs_and_raw_counts_paired': True,
        'cost': {'model_loads': 4, 'responses': 400,
            'generated_tokens': sum(m['generated_tokens'] for m in metrics.values()),
            'generation_seconds': sum(m['generation_seconds'] for m in metrics.values()),
            'allocated_GPU_hours_job_clocks_sum': sum(clocks), 'new_training': False},
        'scope': 'Independent development100, one training seed; saved clipped/unclipped/direction controls at the same8192 budget. Projectn1, not officialpass12 or formal-test evidence. No stable multilingual or sequential effect established.'}
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['correct_out_of100']))


if __name__ == '__main__':
    main()
