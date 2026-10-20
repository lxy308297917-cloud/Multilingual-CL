"""Fill the fixed development grid 0/20/50/100 using saved adapters only."""
import argparse
import json
import os
from pathlib import Path
import subprocess


TRAINING = {
    'sw': 'copsd_official_17b_single_v1/sw_17b_project2900_seed42_20261015_v1',
    'th': 'copsd_official_17b_single_4gpu_v1/th_17b_project2900_seed42_4gpu_20261018_v2',
    'bn': 'copsd_vllm_update_batch_single_v1/bn_17b_project2900_vllm_update_batch_seed42_20261020_v1',
    **{lang: f'copsd_vllm_seeded_cost_single_v2/{lang}_17b_project2900_vllm_seeded_cost_seed42_20261020_v2'
       for lang in ['zh', 'de', 'fr']},
}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=TRAINING)
    parser.add_argument('--root', type=Path, default=Path.home())
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root, lang = args.root.resolve(), args.language
    base = root / 'logs/copsd_official_10lang/development_vllm'
    jobs = []
    for step in [20, 50]:
        cp = root / 'models/checkpoint' / TRAINING[lang] / f'checkpoint-{step}'
        assert read(cp / 'trainer_state.json')['global_step'] == step
        assert (cp / 'adapter_model.safetensors').stat().st_size > 0
        cached = step == 20 and lang in ['sw', 'th']
        run_id = f'{lang}_17b_dev100_vllm_ckpt{step}_20261020_' + ('v1' if cached else 'curve_v1')
        output = base / run_id
        if cached:
            assert read(output / 'DONE.json')['items'] == 100
            assert Path(read(output / 'CONFIG.json')['adapter']).resolve() == cp.resolve()
        else:
            assert not output.exists(), f'Refusing to overwrite/retry {output}'
        jobs.append((step, cp, run_id, cached))
    print(json.dumps({'language': lang, 'grid': [0, 20, 50, 100],
                      'new_evaluators': sum(not x[3] for x in jobs),
                      'check_only': args.check_only}), flush=True)
    if args.check_only:
        return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert gpu.startswith('GPU-') and ',' not in gpu
    script = Path(__file__).with_name('run_dev_vllm_17b.sh')
    for step, cp, run_id, cached in jobs:
        if not cached:
            subprocess.run(['bash', str(script), lang, 'adapter', run_id, str(cp)], check=True)
    final_version = 'v2' if lang in ['zh', 'de', 'fr'] else 'v1'
    names = [f'{lang}_17b_dev100_vllm_base_20261020_v1',
             jobs[0][2], jobs[1][2], f'{lang}_17b_dev100_vllm_ckpt100_20261020_{final_version}']
    configs = [read(base / name / 'CONFIG.json') for name in names]
    canonical = [{k: v for k, v in c.items() if k not in ['adapter', 'output']} for c in configs]
    assert all(c == canonical[0] for c in canonical)
    assert configs[0]['adapter'] is None and configs[0]['protocol'] == 'project_dev_single_vllm_v1'
    rows = [[json.loads(line) for line in (base / name / 'ITEMS.jsonl').read_text().splitlines()] for name in names]
    metrics = [read(base / name / 'DONE.json') for name in names]
    for rr, metric in zip(rows, metrics):
        assert len(rr) == len({r['id'] for r in rr}) == metric['items'] == 100
        assert not metric['formal_test']
        assert sum(r['samples'][0]['correct'] for r in rr) == metric['single_correct']
        for first, other in zip(rows[0], rr):
            assert all(first[k] == other[k] for k in ['id', 'question', 'answer', 'prompt'])
            assert first['samples'][0]['seed'] == other['samples'][0]['seed']
    report = {'language': lang, 'protocol': configs[0]['protocol'],
              'grid': [dict(step=step, metrics=m) for step, m in zip([0, 20, 50, 100], metrics)],
              'paired_inputs_prompts_seeds_configs_verified': True,
              'new_generation_seconds': sum(metrics[i]['generation_seconds'] for i in [1, 2] if not jobs[i-1][3]),
              'new_generated_tokens': sum(metrics[i]['generated_tokens'] for i in [1, 2] if not jobs[i-1][3]),
              'scope': 'Independent development only; saved checkpoints, no new training. Fixed grid chosen after endpoint regression; single seed, exploratory checkpoint selection with winner bias. Formal tests not used for selection. Cached Base/20/final costs excluded from new cost.'}
    target = root / f'logs/copsd_official_10lang/{lang}_dev_fixed_curve_20261020.PUBLIC.json'
    with target.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'language': lang, 'correct': [m['single_correct'] for m in metrics]}), flush=True)


if __name__ == '__main__':
    main()
