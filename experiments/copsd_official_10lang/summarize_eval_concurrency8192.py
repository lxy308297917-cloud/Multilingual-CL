"""Collect completed native-LoRA concurrency measurements and raw-output proof."""
import argparse
import datetime
import json
from pathlib import Path

from export_eval_dense_cpu import digest, read, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    code = Path(__file__).parent
    frozen = read(code / 'eval_concurrency8192_frozen_20261021.PUBLIC.json')
    for name, sha in frozen['code_sha256'].items():
        assert digest(code / name) == sha
    state = read(a.root / 'gpu-queue/queues/copsd-eval8192-concurrency-v1/state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    assert len(state['job']['gpu_uuids']) == 1 and state['job']['config']['max_user_gpus'] == 4
    folder = a.root / frozen['output_relative']
    summary = read(folder / 'SUMMARY.PUBLIC.json')
    baseline = None
    for concurrency in [16, 32, 64]:
        path = folder / f'concurrency{concurrency}'
        cfg, done = read(path / 'CONFIG.json'), read(path / 'DONE.json')
        assert cfg['CUDA_VISIBLE_DEVICES'] == state['job']['gpu_uuids'][0] == summary['same_GPU']
        assert cfg['actual_concurrency_requested'] == done['concurrency'] == concurrency
        assert done['items'] == cfg['items'] == 64
        assert cfg['reference']['max_new_tokens'] == 8192
        rows = [json.loads(x) for x in (path / 'ITEMS.jsonl').read_text().splitlines()]
        assert len(rows) == len({r['id'] for r in rows}) == 64
        assert digest(path / 'ITEMS.jsonl') == done['items_sha256']
        count = sum(len(r['token_ids']) for r in rows)
        assert count == done['generated_tokens'] and all(len(r['token_ids']) <= 8192 for r in rows)
        assert sum(r['finish_reason'] == 'length' for r in rows) == done['truncated']
        assert done['tokens_per_generation_second'] == count / done['generation_seconds']
        assert done['samples_per_generation_second'] == 64 / done['generation_seconds']
        assert all(summary['metrics'][str(concurrency)][k] == v for k, v in done.items())
        if baseline is None:
            baseline = rows
        for first, second in zip(baseline, rows):
            assert all(first[k] == second[k] for k in ['id', 'question', 'answer', 'prompt', 'seed'])
        assert summary['metrics'][str(concurrency)]['exact_token_sequence_matches_vs16'] == sum(
            first['token_ids'] == second['token_ids'] for first, second in zip(baseline, rows))
    job = state['job']
    wall = (datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds()
    result = {**summary, 'collected_at': datetime.datetime.now().isoformat(),
        'parent_completed_exit0': True, 'raw_prompts_IDs_seeds_tokens_and_counts_verified': True,
        'best_observed_resource_concurrency': 64,
        'new_cost': {'responses': 192,
            'generated_tokens': sum(m['generated_tokens'] for m in summary['metrics'].values()),
            'generation_seconds': sum(m['generation_seconds'] for m in summary['metrics'].values()),
            'parent_GPU_hours_counted_once': wall / 3600},
        'source_summary_sha256': digest(folder / 'SUMMARY.PUBLIC.json'),
        'scope': summary['scope'] + ' Raw192 responses and successful parent verified. Same questions/seeds do not make differing batch outputs identical. Resource64 was chosen only by measured throughput/memory, without math scoring or formal-test selection. Cost overlaps the saved benchmark and is not additional generation.'}
    write_new(a.output, result)
    print(json.dumps({'best_concurrency': 64,
        'token_throughput_ratio_64_vs16': summary['metrics']['64']['tokens_per_generation_second'] / summary['metrics']['16']['tokens_per_generation_second'],
        'batch_wall_speedup_64_vs16': summary['metrics']['64']['generation_time_speedup_vs16'], 'parent_exit_code': 0}))


if __name__ == '__main__':
    main()
