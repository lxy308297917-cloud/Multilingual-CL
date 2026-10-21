"""Audit completed dense-export throughput; never mix it into math scores."""
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
    lock = read(code / 'eval_dense8192_frozen_20261021.PUBLIC.json')
    for name, sha in lock['code_sha256'].items():
        assert digest(code / name) == sha, name
    state = read(a.root / 'gpu-queue/queues/copsd-eval8192-dense-v1/state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    folder = a.root / lock['output_relative']
    native = a.root / lock['native16_output_relative']
    done, cfg = read(folder / 'DONE.json'), read(folder / 'CONFIG.json')
    rows = [json.loads(line) for line in (folder / 'ITEMS.jsonl').read_text().splitlines()]
    original = [json.loads(line) for line in (native / 'ITEMS.jsonl').read_text().splitlines()]
    assert len(rows) == len(original) == done['items'] == 64
    assert len({r['id'] for r in rows}) == 64
    assert cfg['concurrency'] == 16 and cfg['max_new_tokens'] == 8192
    assert cfg['CUDA_VISIBLE_DEVICES'] == state['job']['gpu_uuids'][0]
    assert cfg['CUDA_VISIBLE_DEVICES'] == read(native / 'CONFIG.json')['CUDA_VISIBLE_DEVICES']
    assert done['same_physical_GPU_as_native16']
    assert digest(folder / 'ITEMS.jsonl') == done['items_sha256']
    assert sum(len(row['token_ids']) for row in rows) == done['generated_tokens']
    assert all(len(row['token_ids']) <= 8192 for row in rows)
    assert done['tokens_per_generation_second'] == done['generated_tokens'] / done['generation_seconds']
    assert sum(row['finish_reason'] == 'length' for row in rows) == done['truncated']
    for b, r in zip(original, rows):
        assert all(b[k] == r[k] for k in ['id', 'question', 'answer', 'prompt', 'seed'])
    assert sum(b['token_ids'] == r['token_ids'] for b, r in zip(original, rows)) == done['token_sequences_identical_to_native16']
    native_summary = read(code / 'eval_concurrency8192_completed_20261021.PUBLIC.json')
    assert native_summary['parent_completed_exit0']
    best_native = native_summary['metrics']['64']
    job = state['job']
    wall = (datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds()
    result = {**done, 'collected_at': datetime.datetime.now().isoformat(), 'parent_completed_exit0': True,
        'raw64_inputs_prompts_seeds_counts_and_same_GPU_verified': True,
        'comparison_to_best_native64': {
            'native_generation_seconds': best_native['generation_seconds'],
            'dense16_generation_seconds': done['generation_seconds'],
            'batch_wall_speedup_dense16_vs_native64': best_native['generation_seconds'] / done['generation_seconds'],
            'native_generated_tokens': best_native['generated_tokens'], 'dense_generated_tokens': done['generated_tokens'],
            'native_truncated': best_native['truncated'], 'dense_truncated': done['truncated']},
        'resource_decision': 'Use native-LoRA64 for the next separately frozen evaluation preset. Dense16 saves only about5% batch wall time over native64 on this run, requires independent weight exports and changes BF16 model numerics. No math accuracy was measured; do not claim dense efficacy or adopt it as a score-compatible replacement.',
        'new_dense_cost_only': {'responses': 64, 'generated_tokens': done['generated_tokens'],
            'generation_seconds': done['generation_seconds'], 'parent_GPU_hours_counted_once': wall / 3600,
            'cached_native16_and_native64_costs_excluded': True}}
    write_new(a.output, result)
    print(json.dumps({'dense_tokens_per_second': done['tokens_per_generation_second'],
        'dense_wall_speedup_over_native64': result['comparison_to_best_native64']['batch_wall_speedup_dense16_vs_native64'],
        'next_preset': 'native-LoRA64', 'parent_exit_code': 0}))


if __name__ == '__main__':
    main()
