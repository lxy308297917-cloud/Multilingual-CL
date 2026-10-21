"""Aggregate fixed100 formal reports, resolving each metric to its raw output.

Never select a run by its score. Exact complete DONE dictionaries identify the
already reported runs, then configs/checkpoint identities and rows are checked.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

LANGUAGES = 'sw te th bn zh ja ru de fr es'.split()
PROTOCOL = 'project_single_vllm_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(metric):
    return json.dumps(metric, sort_keys=True, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    code = Path(__file__).parent
    inventory = read(code / 'ten_language_training100_inventory_20261021.PUBLIC.json')
    source_reports, proposed = {}, {}

    def source(name):
        path = code / name
        source_reports[name] = sha(path)
        return read(path)

    for lang in LANGUAGES:
        if lang in ['sw', 'th']:
            continue
        date = '20261021' if lang in ['te', 'ja'] else '20261020'
        name = f'{lang}_completed_formal_suite_{date}.PUBLIC.json'
        report = source(name)
        assert report['parent_suite_phase'] == 'completed' and report['parent_suite_exit_code'] == 0
        for pair in report['pairs']:
            proposed[(lang, pair['benchmark'])] = pair['metrics']
    shared = source('sw_th_remaining_formal_pairs_20261020.PUBLIC.json')
    assert shared['parent_suite_phase'] == 'completed' and shared['parent_suite_exit_code'] == 0
    for pair in shared['pairs']:
        proposed[(pair['language'], pair['benchmark'])] = pair['metrics']
    sw = source('sw_mgsm_rev2_vllm_pair_20261020.PUBLIC.json')
    assert sw['parent_queue_completed_exit0']
    proposed[('sw', 'mgsm_rev2')] = sw['metrics']
    english = source('english_gsm8k_completed_singles_20261020.PUBLIC.json')
    assert english['english_subprocesses_returned_successfully']
    for lang in ['sw', 'th']:
        proposed[(lang, 'gsm8k')] = [english['metrics'][english['compared_models'].index(name)]
                                    for name in ['Base', lang.upper() + '100']]
    assert len(proposed) == 30

    folder = args.root / 'logs/copsd_official_10lang/evaluation_vllm'
    outputs = {}
    for path in folder.glob('*/DONE.json'):
        metric = read(path)
        if metric.get('protocol') == PROTOCOL and metric.get('formal_full_test'):
            outputs.setdefault(key(metric), []).append(path.parent)
    unique, results = {}, {}
    for lang in LANGUAGES:
        results[lang] = {}
        checkpoint = args.root / inventory['languages'][lang]['checkpoint_relative']
        assert read(checkpoint / 'trainer_state.json')['global_step'] == 100
        for benchmark, size in [('mgsm_rev2', 250), ('polymath', 500), ('gsm8k', 1319)]:
            metrics = proposed[(lang, benchmark)]
            configs, rows, run_ids = [], [], []
            for index, metric in enumerate(metrics):
                matches = outputs[key(metric)]
                assert len(matches) == 1, (lang, benchmark, matches)
                path = matches[0]
                config = read(path / 'CONFIG.json')
                assert config['protocol'] == PROTOCOL and config['samples'] == 1
                assert config['max_new_tokens'] == 2048 and config['benchmark'] == benchmark
                assert config['language'] == ('en' if benchmark == 'gsm8k' else lang)
                assert config['base_model'] == str(args.root / 'models/Qwen3-1.7B')
                if index == 0:
                    assert config['adapter'] is None
                else:
                    adapter = Path(config['adapter'])
                    assert adapter.resolve() == checkpoint.resolve()
                assert sha(config['input']) == config['input_sha256']
                assert sha(config['official_source']) == config['official_source_sha256']
                assert sha(code / 'evaluate_vllm.py') == config['evaluator_sha256']
                rr = [json.loads(line) for line in (path / 'ITEMS.jsonl').read_text().splitlines()]
                assert len(rr) == len({r['id'] for r in rr}) == metric['items'] == size
                assert all(len(r['samples']) == 1 for r in rr) and metric['n'] == 1
                assert sum(r['samples'][0]['correct'] for r in rr) == metric['single_correct']
                assert sum(r['samples'][0]['tokens'] for r in rr) == metric['generated_tokens']
                assert metric['single_accuracy'] == metric['single_correct'] / size
                configs.append(config)
                rows.append(rr)
                run_ids.append(path.name)
                unique[path.name] = metric
            comparable = [{k: v for k, v in cfg.items() if k not in ['adapter', 'output_dir']} for cfg in configs]
            assert comparable[0] == comparable[1]
            for before, after in zip(*rows):
                assert all(before[k] == after[k] for k in ['id', 'question', 'answer', 'prompt'])
            results[lang][benchmark] = {'items': size, 'Base_correct': metrics[0]['single_correct'],
                'single100_correct': metrics[1]['single_correct'],
                'delta_percentage_points': 100 * (metrics[1]['single_accuracy'] - metrics[0]['single_accuracy']),
                'raw_run_ids': run_ids, 'paired_gained': sum(not x['samples'][0]['correct'] and y['samples'][0]['correct'] for x, y in zip(*rows)),
                'paired_lost': sum(x['samples'][0]['correct'] and not y['samples'][0]['correct'] for x, y in zip(*rows))}
    macro = {benchmark: {role.replace('_correct', '_accuracy'): sum(results[lang][benchmark][role] / results[lang][benchmark]['items'] for lang in LANGUAGES) / 10
                        for role in ['Base_correct', 'single100_correct']}
             for benchmark in ['mgsm_rev2', 'polymath', 'gsm8k']}
    report = {'observed_at': datetime.datetime.now().isoformat(), 'protocol': PROTOCOL,
        'model': 'Qwen3-1.7B', 'model_revision': inventory['model_revision'],
        'max_new_tokens': 2048, 'samples': 1, 'training_updates': 100,
        'completed_languages': 10, 'results': results, 'descriptive_mean_accuracy': macro,
        'all10_English_scores_below_Base': all(results[l]['gsm8k']['single100_correct'] < results[l]['gsm8k']['Base_correct'] for l in LANGUAGES),
        'source_report_sha256': source_reports,
        'existing_unique_engine_output_cost_not_additional_work': {
            'unique_runs': len(unique), 'generated_responses': sum(m['generation_count'] for m in unique.values()),
            'generated_tokens': sum(m['generated_tokens'] for m in unique.values()),
            'generation_seconds': sum(m['generation_seconds'] for m in unique.values()),
            'evaluation_seconds_including_generation_grading_output': sum(m['evaluation_seconds'] for m in unique.values()),
            'scope': 'Unique saved engines only; cached English Base counted once. Engine generation is included in evaluation, so these two times must not be added. Parent allocation clocks/training/failed work/other protocols are excluded.'},
        'new_generation': False, 'GPU_used': False,
        'scope': 'Complete fixed100 project n1/2048 formal accuracy inventory, not officialpass12, released8192 confirmation, checkpoint/language selection, or sequential results. Raw reported runs resolved by exact full DONE dictionaries, never highest scores, and actual rows/configs/input/source/checkpoint identities checked. Historical4GPU/initializer/JA repair/recovery caveats remain. Translated questions and English questions recur across models; mean accuracies are descriptive, not independent pooled trials. Older formal rows do not all save the per-request seed field; equality uses frozen code, seed rule, IDs and prompts. No new compute or cost sum with overlapping source reports.'}
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'completed_languages': 10, 'macro': macro, 'unique_runs': len(unique)}))


if __name__ == '__main__':
    main()
