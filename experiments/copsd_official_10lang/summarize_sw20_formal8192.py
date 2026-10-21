"""Audit frozen SW20 formal stages; distinguish stage output from parent success."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

from evaluate_transformers import load_items
from evaluate_vllm import question_seed


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--benchmark', choices=['mgsm_rev2', 'polymath', 'gsm8k'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=['base', 'seed42', 'seed43'],
                   default=['base', 'seed42', 'seed43'],
                   help='Explicit completed subset; Base plus at least one trained model.')
    a = p.parse_args()
    assert len(set(a.models)) == len(a.models) and 'base' in a.models and len(a.models) >= 2
    assert not a.output.exists()
    code = Path(__file__).parent
    frozen = read(code / 'sw20_formal8192_frozen_20261021.PUBLIC.json')
    for name, sha in frozen['code_sha256'].items():
        assert digest(code / name) == sha, name
    configurations, records, metrics, sources, parents = {}, {}, {}, {}, {}
    stages = [s for s in frozen['stages'] if s['benchmark'] == a.benchmark and s['model'] in a.models]
    assert {s['model'] for s in stages} == set(a.models) and len(stages) == len(a.models)
    for stage in stages:
        model = stage['model']
        directory = a.root / 'logs/copsd_official_10lang/evaluation_vllm8192' / stage['run_id']
        c, m = read(directory / 'CONFIG.json'), read(directory / 'DONE.json')
        assert c['protocol'] == m['protocol'] == frozen['protocol']
        assert c['language'] == stage['language'] and c['benchmark'] == a.benchmark
        assert c['model_revision'] == '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
        assert c['model'] == str(a.root / 'models/Qwen3-1.7B')
        assert c['checkpoint_step'] == stage['step']
        assert c['adapter_sha256'] == stage.get('adapter_sha256')
        if stage['adapter_relative']:
            checkpoint = a.root / stage['adapter_relative']
            assert Path(c['adapter']).resolve() == checkpoint.resolve()
            assert digest(checkpoint / 'adapter_model.safetensors') == c['adapter_sha256']
            assert read(checkpoint / 'trainer_state.json')['global_step'] == 20
        else:
            assert c['adapter'] is None
        assert c['samples'] == m['n'] == 1 and c['max_new_tokens'] == 8192
        assert c['max_num_seqs'] == m['actual_max_num_seqs'] == 16
        assert c['input_sha256'] == digest(c['input'])
        assert c['evaluator_sha256'] == digest(code / 'evaluate_formal8192.py')
        official_source = a.root / 'COPSD-official-20261015/polymath_eval/evaluate_math.py'
        assert c['official_source_sha256'] == digest(official_source)
        rows = [json.loads(line) for line in (directory / 'ITEMS.jsonl').read_text().splitlines()]
        assert len(rows) == len({r['id'] for r in rows}) == m['items'] == stage['items']
        assert m['formal_full_test'] and m['generation_count'] == len(rows)
        kind = 'mgsm_rev2' if a.benchmark == 'mgsm_rev2' else 'polymath_project'
        inputs = list(load_items(Path(c['input']), kind, c['language']))
        assert len(inputs) == len(rows)
        totals = Counter()
        for row, item in zip(rows, inputs):
            assert all(row[k] == item[k] for k in ['id', 'question', 'answer'])
            assert len(row['samples']) == 1
            sample = row['samples'][0]
            assert sample['seed'] == question_seed(42, c['language'], row['id'])
            assert sample['tokens'] == len(sample['token_ids']) <= 8192
            totals['single_correct'] += sample['correct']
            totals['generated_tokens'] += sample['tokens']
            totals['truncated'] += sample['finish_reason'] == 'length'
            totals['normal_terminations'] += sample['finish_reason'] == 'stop'
            totals['answer_extraction_failures'] += sample['boxed_answer'] is None
        assert all(m[k] == v for k, v in totals.items())
        assert m['single_accuracy'] == totals['single_correct'] / len(rows)
        configurations[model] = {k: v for k, v in c.items() if k not in ['adapter', 'adapter_sha256', 'checkpoint_step']}
        records[model], metrics[model] = rows, m
        sources[model] = {'run_id': stage['run_id'], 'raw_items_sha256': digest(directory / 'ITEMS.jsonl'),
                          'config_sha256': digest(directory / 'CONFIG.json'), 'DONE_sha256': digest(directory / 'DONE.json')}
        queue = {'base': 'copsd-sw20-formal8192-confirm-v1',
                 'seed42': 'copsd-sw20-formal8192-seed42-v1', 'seed43': 'copsd-sw20-formal8192-seed43-v1'}[model]
        state = read(a.root / 'gpu-queue/queues' / queue / 'state.json')
        parents[model] = {'queue': queue, 'observed_phase': state['phase'],
                          'exit_code': state['job'].get('exit_code')}
    assert all(c == configurations['base'] for c in configurations.values())
    paired = {}
    for model in [name for name in a.models if name != 'base']:
        gained = lost = 0
        for base, trained in zip(records['base'], records[model]):
            assert all(base[k] == trained[k] for k in ['id', 'question', 'answer', 'prompt'])
            b, t = base['samples'][0]['correct'], trained['samples'][0]['correct']
            gained += not b and t
            lost += b and not t
        paired[model] = {'Base_correct': metrics['base']['single_correct'],
            'trained20_correct': metrics[model]['single_correct'], 'gained': gained, 'lost': lost,
            'delta_correct': gained - lost}
    out = {'observed_at': datetime.datetime.now().isoformat(), 'protocol': frozen['protocol'],
        'benchmark': a.benchmark, 'metrics': metrics, 'paired_counts': paired,
        'sources': sources, 'parents': parents, 'all_requested_stage_outputs_verified': True,
        'requested_models': a.models,
        'all_three_models_included': set(a.models) == {'base', 'seed42', 'seed43'},
        'all_requested_parent_suites_completed_exit0': all(p['observed_phase'] == 'completed' and p['exit_code'] == 0 for p in parents.values()),
        'all_three_parent_suites_completed_exit0': set(a.models) == {'base', 'seed42', 'seed43'} and all(p['observed_phase'] == 'completed' and p['exit_code'] == 0 for p in parents.values()),
        'stage_cost_only': {'responses': sum(m['items'] for m in metrics.values()),
            'generated_tokens': sum(m['generated_tokens'] for m in metrics.values()),
            'generation_seconds': sum(m['generation_seconds'] for m in metrics.values()),
            'scope': 'Unique completed requested benchmark stages only; excludes other suites, training, loading and parent-clock allocation. Do not add to overlapping later aggregate reports.'},
        'new_generation_from_this_analysis': False,
        'scope': 'Frozen project n1/8192 stage comparison, not official pass12. Completed benchmark stages do not imply the other benchmark stages or English have completed. Both checkpoints were preselected using independent development; the formal inputs are existing project test sets, not newly held-out tests. No language selection from these formal scores.'}
    with a.output.open('x') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(paired))


if __name__ == '__main__':
    main()
