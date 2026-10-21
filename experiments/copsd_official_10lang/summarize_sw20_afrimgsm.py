"""Collect official AfriMGSM pass12 for two preselected SW20 adapters.

Audits original official per-generation records without rescoring. Base is
cached, not regenerated. This report cannot be used for language selection.
"""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path

from evaluate_transformers import load_items
from sequential_evaluation import digest, read, write_new


def audit(root, code, run_id, adapter):
    directory = root / 'logs/copsd_official_10lang/evaluation_vllm' / run_id
    config, metrics, official = [read(directory / name) for name in ['CONFIG.json', 'DONE.json', 'OFFICIAL_RESULTS.json']]
    assert config['protocol'] == metrics['protocol'] == 'afrimgsm_release_vllm_v1'
    assert config['samples'] == metrics['n'] == official['val_n'] == 12
    assert config['max_new_tokens'] == official['max_new_tokens'] == 1024
    assert config['evaluator_sha256'] == digest(code / 'evaluate_vllm.py')
    source = root / 'COPSD-official-20261015/african_langs_eval/evaluate_math.py'
    assert config['official_source_sha256'] == digest(source)
    assert config['input_sha256'] == digest(Path(config['input']))
    assert config['base_model'] == str(root / 'models/Qwen3-1.7B')
    assert config['adapter'] == (str(adapter) if adapter else None)
    assert config['scope'] == 'frozen_formal_test' and metrics['formal_full_test']
    inputs = list(load_items(Path(config['input']), 'afrimgsm', 'swa'))
    rows = [json.loads(line) for line in (directory / 'ITEMS.jsonl').read_text().splitlines()]
    assert len(inputs) == len(rows) == len(official['results']) == metrics['items'] == 250
    totals = Counter()
    for item, row, record in zip(inputs, rows, official['results']):
        assert all(item[key] == row[key] for key in ['id', 'question', 'answer'])
        assert row['id'] == str(record['problem_id'])
        assert row['question'] == record['problem'] and str(row['answer']) == str(record['ground_truth'])
        assert row['prompt'] == record['prompt']
        assert len(row['samples']) == len(record['generations']) == 12
        correct = 0
        for index, (sample, original) in enumerate(zip(row['samples'], record['generations'])):
            assert sample['sample'] == index and sample['tokens'] == len(sample['token_ids']) <= 1024
            assert sample['correct'] == original['correct']
            assert sample['official_generated_text'] == original['full_generation']
            correct += sample['correct']
            totals['generated_tokens'] += sample['tokens']
            totals['normal_terminations'] += sample['finish_reason'] == 'stop'
            totals['truncated'] += sample['finish_reason'] == 'length'
            totals['answer_extraction_failures'] += sample['boxed_answer'] is None
        assert record['num_correct'] == correct and record['pass_at_n'] == (correct > 0)
        totals['single_correct'] += row['samples'][0]['correct']
        totals['pass_n_correct'] += correct > 0
        totals['all_samples_correct'] += correct
        totals['majority_correct'] += record['majority_vote_correct']
    assert all(metrics[key] == totals[key] for key in ['generated_tokens', 'normal_terminations',
        'truncated', 'answer_extraction_failures', 'single_correct', 'pass_n_correct'])
    assert totals['all_samples_correct'] == official['average_at_n']
    assert totals['majority_correct'] == official['majority_vote_at_n']
    assert totals['pass_n_correct'] == official['pass_at_n']
    assert metrics['generation_count'] == official['total_solutions'] == 3000
    return {'metrics': metrics, 'average_at12_correct_of3000': totals['all_samples_correct'],
        'majority_vote_at12_correct_of250': totals['majority_correct'],
        'source_sha256': {name: digest(directory / name) for name in ['CONFIG.json', 'DONE.json', 'ITEMS.jsonl', 'OFFICIAL_RESULTS.json']}}, config, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=['seed42', 'seed43'], default=['seed42', 'seed43'])
    a = p.parse_args()
    assert len(a.models) == len(set(a.models))
    code = Path(__file__).parent
    frozen = read(code / 'sw20_afrimgsm_frozen_20261021.PUBLIC.json')
    for name, sha in frozen['code_sha256'].items():
        assert digest(code / name) == sha
    state = read(a.root / 'gpu-queue/queues' / frozen['queue'] / 'state.json')
    complete = state['phase'] == 'completed' and state['job']['exit_code'] == 0
    if set(a.models) == {'seed42', 'seed43'}:
        assert complete
    else:
        assert complete or state['phase'] == 'running'
    job = state['job']
    assert Path(job['script']).resolve() == code / 'run_sw20_afrimgsm_pair.sh'
    assert job['script_sha256'] == frozen['code_sha256']['run_sw20_afrimgsm_pair.sh']
    assert not job['config']['job']['args'] and len(job['gpu_uuids']) == 1
    results, configs, raw = {}, {}, {}
    results['base'], configs['base'], raw['base'] = audit(a.root, code, frozen['cached_Base_run_id'], None)
    for name, sha in frozen['cached_Base_sha256'].items():
        assert results['base']['source_sha256'][name] == sha
    paired = {}
    for stage in frozen['stages']:
        model = stage['model']
        if model not in a.models:
            continue
        adapter = a.root / stage['checkpoint_relative']
        assert digest(adapter / 'adapter_model.safetensors') == stage['adapter_sha256']
        assert read(adapter / 'trainer_state.json')['global_step'] == 20
        results[model], configs[model], raw[model] = audit(a.root, code, stage['run_id'], adapter)
        assert {k: v for k, v in configs[model].items() if k not in ['adapter', 'output_dir']} == {
            k: v for k, v in configs['base'].items() if k not in ['adapter', 'output_dir']}
        gained = lost = 0
        for base, trained in zip(raw['base'], raw[model]):
            assert all(base[k] == trained[k] for k in ['id', 'question', 'answer', 'prompt'])
            b, t = [any(s['correct'] for s in row['samples']) for row in [base, trained]]
            gained += not b and t
            lost += b and not t
        paired[model] = {'gained_pass12': gained, 'lost_pass12': lost, 'net_pass12_correct': gained - lost}
    wall = ((datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds()
        if complete else None)
    write_new(a.output, {'observed_at': datetime.datetime.now().isoformat(), 'protocol': frozen['protocol'],
        'parent_completed_exit0': complete, 'results': results, 'paired_pass12': paired,
        'requested_models': a.models, 'both_training_seeds_included': set(a.models) == {'seed42', 'seed43'},
        'new_cost_excluding_cached_Base': {'responses': 3000 * len(a.models),
            'generated_tokens': sum(results[k]['metrics']['generated_tokens'] for k in a.models),
            'generation_seconds': sum(results[k]['metrics']['generation_seconds'] for k in a.models),
            'parent_GPU_hours_counted_once': wall / 3600 if complete and len(a.models) == 2 else None,
            'scope': 'Requested completed benchmark stages only. Parent allocation is included only when both seeds and parentexit0 are complete; subset reports overlap the eventual full report.'},
        'scope': frozen['scope'], 'selection_policy': frozen['selection_policy'],
        'overlap_caveat': frozen['overlap_caveat']})
    print(json.dumps({k: v['metrics']['pass_n_correct'] for k, v in results.items()}))


if __name__ == '__main__':
    main()
