"""Audit length probes, including exact same-trajectory prefix censoring."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time

from evaluate_vllm import official_api


def read(p):
    return json.loads(p.read_text())


def load_rows(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def common_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return min(len(a), len(b))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--language', choices=['sw', 'bn', 'fr'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    root, lang = a.root, a.language
    state = read(root/'gpu-queue/queues'/f'copsd-{lang}-length32-v1/state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    job = state['job']
    assert hashlib.sha256(Path(job['script']).read_bytes()).hexdigest() == job['script_sha256']
    base = root/'logs/copsd_official_10lang/length_sensitivity'
    tokenizer = None
    results, configs, records, costs = {}, {}, {}, []
    for mode in ['base', 'ckpt100']:
        d = base/f'{lang}_{mode}_dev32_2048_8192_20261020_v1'
        cfg = read(d/'CONFIG.PRIVATE.json')
        assert cfg['new_budgets'] == [2048, 8192]
        assert hashlib.sha256(Path(__file__).with_name('evaluate_dev_length_sensitivity.py').read_bytes()).hexdigest() == cfg['code_sha256']
        source = Path(cfg['source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == cfg['official_source_sha256']
        api = official_api(source)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True)
        configs[mode] = cfg
        done = read(d/'DONE.json')
        rows = {cap: load_rows(d/f'max{cap}_ITEMS.PRIVATE.jsonl') for cap in [2048, 8192]}
        for cap, rr in rows.items():
            m = done['metrics'][str(cap)]
            assert len(rr) == len({r['id'] for r in rr}) == m['items'] == 32
            assert sum(r['correct'] for r in rr) == m['single_correct']
            assert sum(len(r['token_ids']) for r in rr) == m['generated_tokens']
            assert sum(r['finish_reason'] == 'length' for r in rr) == m['truncated']
            costs.append(m)
        for x, y in zip(rows[2048], rows[8192]):
            assert all(x[k] == y[k] for k in ['id', 'question', 'answer', 'prompt', 'seed'])
        lcp = [common_prefix(x['token_ids'], y['token_ids']) for x, y in zip(rows[2048], rows[8192])]
        prefix_analysis = []
        grading_tick = time.monotonic()
        for row in rows[8192]:
            whole = tokenizer.decode(row['token_ids'], skip_special_tokens=True)
            whole_answer = api['extract_boxed_answer'](whole)
            assert bool(api['grade_answer'](whole_answer, row['answer'])) == row['correct']
            prefix = tokenizer.decode(row['token_ids'][:2048], skip_special_tokens=True)
            answer = api['extract_boxed_answer'](prefix)
            correct = bool(api['grade_answer'](answer, row['answer']))
            prefix_analysis.append({'id': row['id'], 'prefix2048_correct': correct,
                'prefix2048_boxed_answer': answer, 'full_correct': row['correct']})
        path = d/'PREFIX2048_ANALYSIS.PRIVATE.json'
        grading_seconds = time.monotonic()-grading_tick
        if path.exists():
            assert read(path) == prefix_analysis
        else:
            path.write_text(json.dumps(prefix_analysis, ensure_ascii=False, indent=2)+'\n')
        result = {**done, 'generated_prefix_common_length_min_median_max': [min(lcp), statistics.median(lcp), max(lcp)],
            'same_long_trajectory_prefix_scoring': {
                'first2048_correct': sum(r['prefix2048_correct'] for r in prefix_analysis),
                'full8192_correct': sum(r['full_correct'] for r in prefix_analysis),
                'gained_after2048': sum(not r['prefix2048_correct'] and r['full_correct'] for r in prefix_analysis),
                'lost_after2048': sum(r['prefix2048_correct'] and not r['full_correct'] for r in prefix_analysis),
                'full_decoded_scores_equal_original': True,
                'CPU_grader_calls': 64, 'prefix_and_full_decoding_grading_wall_seconds': grading_seconds,
                'scope': 'Exact censoring of the same long generated trajectories; distinct from independently sampled short-budget runs.'}}
        results[mode] = result
        records[mode] = rows
    for field in ['model', 'input', 'language', 'samples', 'temperature', 'top_p', 'top_k',
                  'seed', 'source', 'input_sha256', 'official_source_sha256', 'code_sha256']:
        assert configs['base'][field] == configs['ckpt100'][field], field
    assert configs['base']['adapter'] is None and configs['ckpt100']['adapter'] is not None
    for cap in [2048, 8192]:
        for x, y in zip(records['base'][cap], records['ckpt100'][cap]):
            assert all(x[k] == y[k] for k in ['id', 'question', 'answer', 'prompt', 'seed'])
    hours = (datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()/3600
    report = {'observed_at': datetime.datetime.now().isoformat(), 'language': lang,
        'protocol': 'project_development_length_sensitivity32_v1', 'queue_phase': 'completed', 'exit_code': 0,
        'models': results, 'inputs_prompts_seeds_and_source_configs_paired': True,
        'all_raw_counts_and_same_long_trace_prefix_scores_verified': True,
        'new_cost': {'engine_loads': 2, 'responses': sum(m['items'] for m in costs),
            'generated_tokens': sum(m['generated_tokens'] for m in costs),
            'generation_seconds': sum(m['generation_seconds'] for m in costs),
            'single_GPU_hours_job_clock': hours, 'new_training': False, 'cached_inference_excluded': True},
        'scope': 'First32 independent development questions, one model seed and sampling seed; not formal test or official8192/pass12 reproduction. Short/long runs may diverge before2048 despite equal seeds; exact prefix censoring is separately reported. No checkpoint or language selection based on these outputs.'}
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({m: {'fresh': {k: v['single_correct'] for k, v in r['metrics'].items()},
        'same_trace': r['same_long_trajectory_prefix_scoring']} for m, r in results.items()}))


if __name__ == '__main__':
    main()
