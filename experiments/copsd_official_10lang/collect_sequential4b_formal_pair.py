"""Audit one completed formal Base/model pair before the seven-case suite ends.

A partial pair never stands in for a completed stage or forgetting matrix.
"""
import argparse
from collections import Counter
import datetime
import importlib.metadata
import json
from pathlib import Path

from formal4b_contract import prepare, digest, read
from sequential4b_selection import local
from evaluate_vllm import question_seed


def case(root, freeze, name, queue):
    f, cases = prepare(root, freeze)
    assert not f['engineering_preflight_only']
    selected = cases[name]
    cached = f.get('cached_cases', {}).get(name)
    if cached:
        assert queue is None and f['model_id'] == 'Base'
        audit_path = local(root, f['Base_cache_audit_relative'])
        assert digest(audit_path) == f['Base_cache_audit_sha256']
        assert read(audit_path)['cases'][name] == cached
        folder = local(root, cached['output_relative'])
        for file_name, expected in cached['file_sha256'].items():
            assert digest(folder / file_name) == expected
        hours = 0.
    else:
        assert queue and '/' not in queue and '..' not in queue
        state = read(root / 'gpu-queue/queues' / queue / 'state.json')
        assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
        job = state['job']
        assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
        assert job['config']['job']['args'] == [str(Path(freeze).resolve()), name]
        assert job['script_sha256'] == digest(Path(__file__).with_name('run_formal4b_case.sh'))
        folder = selected['output']
        hours = (datetime.datetime.fromisoformat(job['finished_at']) -
                 datetime.datetime.fromisoformat(job['started_at'])).total_seconds() / 3600
    config, done = read(folder / 'CONFIG.json'), read(folder / 'DONE.json')
    packages = {key: importlib.metadata.version(key) for key in ['torch', 'transformers', 'vllm', 'math-verify']}
    if not cached:
        assert config == {**selected['config'], 'packages': packages}
    else:
        assert config['packages'] == packages and config['adapter'] is None
    assert read(folder / 'PHASE.json')['phase'] == 'done'
    assert done['formal_full_test'] and done['actual_max_num_seqs'] == 16 and done['n'] == 1
    assert done['items'] == done['generation_count'] == selected['count']
    rows = [json.loads(line) for line in (folder / 'ITEMS.jsonl').read_text().splitlines()]
    assert len(rows) == selected['count']
    totals, bands = Counter(), {}
    for expected, row in zip(selected['items'], rows):
        assert all(row[key] == value for key, value in expected.items())
        assert len(row['samples']) == 1
        sample = row['samples'][0]
        assert type(sample['correct']) is bool
        assert sample['seed'] == question_seed(42, selected['spec']['language'], row['id'])
        assert sample['tokens'] == len(sample['token_ids']) <= 8192
        totals['single_correct'] += sample['correct']
        totals['generated_tokens'] += sample['tokens']
        totals['truncated'] += sample['finish_reason'] == 'length'
        totals['normal_terminations'] += sample['finish_reason'] == 'stop'
        totals['answer_extraction_failures'] += sample['boxed_answer'] is None
        if selected['difficulties']:
            level = selected['difficulties'][row['id']]
            assert row['split'] == level
            band = bands.setdefault(level, Counter())
            band['items'] += 1
            band['single_correct'] += sample['correct']
    assert all(done[key] == value for key, value in totals.items())
    assert done['single_accuracy'] == totals['single_correct'] / selected['count']
    assert done['scores_by_difficulty'] == {
        level: {**dict(band), 'single_accuracy': band['single_correct'] / band['items']}
        for level, band in bands.items()}
    result = {'metrics': done, 'config': config,
              'output_relative': str(folder.relative_to(root)),
              'files_sha256': {key: digest(folder / key) for key in ['CONFIG.json', 'DONE.json', 'ITEMS.jsonl']},
              'cost': {'new_GPU_hours': hours, 'reused': bool(cached),
                       'new_generated_tokens': 0 if cached else done['generated_tokens']}}
    return f, result, rows


def collect(root, base_freeze, model_freeze, name, base_queue, model_queue, output):
    root, output = Path(root), Path(output)
    assert not output.exists()
    bf, base, br = case(root, base_freeze, name, base_queue)
    mf, model, mr = case(root, model_freeze, name, model_queue)
    assert bf['selection_record_sha256'] == mf['selection_record_sha256']
    assert bf['model_id'] == 'Base' and mf['model_id'] != 'Base'
    fields = ['benchmark', 'language', 'model', 'model_revision', 'input_sha256',
              'items', 'samples', 'max_new_tokens', 'max_num_seqs', 'model_context_limit',
              'temperature', 'top_p', 'top_k', 'seed', 'seed_policy',
              'official_source_sha256', 'released_length_script_sha256', 'packages']
    assert all(base['config'][key] == model['config'][key] for key in fields)
    for a, b in zip(br, mr):
        assert all(a[key] == b[key] for key in ['id', 'question', 'answer', 'prompt'])
        assert a['samples'][0]['seed'] == b['samples'][0]['seed']
    import numpy as np
    before = np.array([row['samples'][0]['correct'] for row in br], dtype=np.int8)
    after = np.array([row['samples'][0]['correct'] for row in mr], dtype=np.int8)
    delta = after - before
    rng = np.random.default_rng(20261024)
    bootstrap = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1) * 100
    result = {'model_id': mf['model_id'], 'case': name, 'items': len(delta),
              'Base_correct': int(before.sum()), 'model_correct': int(after.sum()),
              'gained': int((delta == 1).sum()), 'lost': int((delta == -1).sum()),
              'delta_percentage_points': float(delta.mean() * 100),
              'paired_question_bootstrap95_CI_pp': np.quantile(bootstrap, [.025, .975]).tolist(),
              'Base': base, 'model': model,
              'selection_record_sha256': mf['selection_record_sha256'],
              'base_freeze_sha256': digest(base_freeze), 'model_freeze_sha256': digest(model_freeze),
              'source_sha256': digest(Path(__file__)),
              'proves_complete_seven_case_stage': False,
              'scope': 'One completed formal pair only, not a complete seven-case stage, language-preservation result or forgetting matrix. Original scores retained. Question CI describes fixed models, not trainingseed population or multiplicity-adjusted inference. Report cost events separately to avoid recounting this pair in the later full-suite collector.'}
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base-freeze', type=Path, required=True)
    parser.add_argument('--model-freeze', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--base-queue')
    parser.add_argument('--model-queue', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    r = collect(args.root, args.base_freeze, args.model_freeze, args.case,
                args.base_queue, args.model_queue, args.output)
    print(json.dumps({key: r[key] for key in ['model_id', 'case', 'items', 'Base_correct', 'model_correct', 'gained', 'lost', 'delta_percentage_points', 'paired_question_bootstrap95_CI_pp']}))


if __name__ == '__main__':
    main()
