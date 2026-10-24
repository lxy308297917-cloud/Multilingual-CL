"""Collect all seven selected-language formal cases with cache provenance."""
import argparse
from collections import Counter
import datetime
import importlib.metadata
import json
from pathlib import Path

from formal4b_contract import prepare, digest, read
from evaluate_vllm import question_seed
from sequential4b_selection import local, selection, required_cases


def collect(root, freeze, queue_map, output):
    root, freeze, queue_map = Path(root), Path(freeze), Path(queue_map)
    output = Path(output) if output is not None else None
    assert output is None or not output.exists()
    frozen, cases = prepare(root, freeze)
    assert frozen['engineering_preflight_only'] is False
    selected = selection(root, local(root, frozen['selection_record_relative']),
                         frozen['selection_record_sha256'])
    assert len(cases) == 7
    assert {(case['spec']['benchmark'], case['spec']['language']) for case in cases.values()} == set(required_cases(selected['selected_languages']))
    cached = frozen.get('cached_cases', {})
    queues = read(queue_map)
    assert set(queues) == set(cases) - set(cached)
    assert len(set(queues.values())) == len(queues)
    if cached:
        audit_path = local(root, frozen['Base_cache_audit_relative'])
        assert digest(audit_path) == frozen['Base_cache_audit_sha256']
        audit = read(audit_path)
        assert audit['original_Base_only'] and frozen['model_id'] == 'Base'
        assert all(audit['cases'][key] == value for key, value in cached.items())
    packages = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'vllm', 'math-verify']}
    accepted, total_hours, new_tokens = {}, 0., 0
    for name, case in cases.items():
        if name in cached:
            folder = local(root, cached[name]['output_relative'])
            for file_name, expected in cached[name]['file_sha256'].items():
                assert digest(folder / file_name) == expected
            cost = {'reused': True, 'new_GPU_hours': 0., 'new_generated_tokens': 0}
        else:
            folder = case['output']
            queue = queues[name]
            assert queue and '/' not in queue and '..' not in queue
            state = read(root / 'gpu-queue/queues' / queue / 'state.json')
            assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
            job = state['job']
            assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
            assert job['config']['job']['args'] == [str(freeze.resolve()), name]
            assert job['script_sha256'] == digest(Path(__file__).with_name('run_formal4b_case.sh'))
            config = read(folder / 'CONFIG.json')
            assert config == {**case['config'], 'packages': packages}
            hours = (datetime.datetime.fromisoformat(job['finished_at']) -
                     datetime.datetime.fromisoformat(job['started_at'])).total_seconds() / 3600
            assert hours > 0
            total_hours += hours
            cost = {'reused': False, 'new_GPU_hours': hours, 'queue': queue,
                    'queue_state_sha256': digest(root / 'gpu-queue/queues' / queue / 'state.json')}
        done = read(folder / 'DONE.json')
        assert read(folder / 'PHASE.json')['phase'] == 'done'
        assert done['formal_full_test'] and done['n'] == 1 and done['actual_max_num_seqs'] == 16
        assert done['items'] == done['generation_count'] == case['count']
        rows = [json.loads(line) for line in (folder / 'ITEMS.jsonl').read_text().splitlines()]
        assert len(rows) == case['count']
        totals, bands = Counter(), {}
        for item, row in zip(case['items'], rows):
            assert all(row[key] == value for key, value in item.items())
            assert len(row['samples']) == 1 and row['prompt']
            sample = row['samples'][0]
            assert type(sample['correct']) is bool
            assert sample['seed'] == question_seed(42, case['spec']['language'], item['id'])
            assert sample['tokens'] == len(sample['token_ids']) <= 8192
            totals['single_correct'] += sample['correct']
            totals['generated_tokens'] += sample['tokens']
            totals['truncated'] += sample['finish_reason'] == 'length'
            totals['normal_terminations'] += sample['finish_reason'] == 'stop'
            totals['answer_extraction_failures'] += sample['boxed_answer'] is None
            if case['difficulties']:
                level = case['difficulties'][item['id']]
                assert row['split'] == level
                band = bands.setdefault(level, Counter())
                band['items'] += 1
                band['single_correct'] += sample['correct']
        assert all(done[key] == value for key, value in totals.items())
        assert done['single_accuracy'] == totals['single_correct'] / case['count']
        expected_bands = {level: {**dict(band), 'single_accuracy': band['single_correct'] / band['items']}
                          for level, band in bands.items()}
        assert done['scores_by_difficulty'] == expected_bands
        if name not in cached:
            new_tokens += done['generated_tokens']
            cost['new_generated_tokens'] = done['generated_tokens']
        accepted[name] = {'metrics': done, 'cost': cost,
                          'output_relative': str(folder.relative_to(root)),
                          'files_sha256': {file_name: digest(folder / file_name)
                                           for file_name in ['CONFIG.json', 'DONE.json', 'ITEMS.jsonl']}}
    result = {'model_id': frozen['model_id'], 'protocol': frozen['protocol'],
              'model_revision': frozen['model_revision'],
              'selection_record_sha256': frozen['selection_record_sha256'],
              'frozen_plan_relative': str(freeze.relative_to(root)), 'frozen_plan_sha256': digest(freeze),
              'queue_map_relative': str(queue_map.relative_to(root)), 'queue_map_sha256': digest(queue_map),
              'collector_sha256': digest(Path(__file__)),
              'model_provenance': frozen['model_provenance'], 'all_seven_cases_accepted': True,
              'cases': accepted, 'new_evaluation_GPU_hours': total_hours,
              'new_generated_tokens': new_tokens, 'reused_cases': sorted(cached),
              'scope': 'Complete seven-case formal accounting. Language analysis and stage-to-stage forgetting matrices are separate required outputs; formal scores never feed language/checkpoint selection.'}
    if output is not None:
        with output.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--queue-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root, args.freeze, args.queue_map, args.output)
    print(json.dumps({key: result[key] for key in ['model_id', 'all_seven_cases_accepted', 'new_evaluation_GPU_hours', 'new_generated_tokens', 'reused_cases']}))


if __name__ == '__main__':
    main()
