"""Audit completed fixed-grid development outputs and count only new inference."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

LANGUAGES = ['sw', 'th', 'bn', 'zh', 'de', 'fr']


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    state = read(args.root / 'gpu-queue/queues/copsd-dev-checkpoint-curves-v1/state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert hashlib.sha256(Path(job['script']).read_bytes()).hexdigest() == job['script_sha256']
    root = args.root / 'logs/copsd_official_10lang'
    curves, fresh, common_ids, common_answers = {}, [], None, None
    for lang in LANGUAGES:
        curve = read(root / f'{lang}_dev_fixed_curve_20261020.PUBLIC.json')
        assert curve['paired_inputs_prompts_seeds_configs_verified']
        assert [r['step'] for r in curve['grid']] == [0, 20, 50, 100]
        version = 'v2' if lang in ['zh', 'de', 'fr'] else 'v1'
        suffixes = ['base_20261020_v1',
            'ckpt20_20261020_' + ('v1' if lang in ['sw', 'th'] else 'curve_v1'),
            'ckpt50_20261020_curve_v1', 'ckpt100_20261020_' + version]
        reference, canonical = None, None
        for index, suffix in enumerate(suffixes):
            path = root / 'development_vllm' / f'{lang}_17b_dev100_vllm_{suffix}'
            rows = [json.loads(line) for line in (path / 'ITEMS.jsonl').read_text().splitlines()]
            config, metric = read(path / 'CONFIG.json'), read(path / 'DONE.json')
            assert metric == curve['grid'][index]['metrics']
            assert config['protocol'] == 'project_dev_single_vllm_v1' and config['samples'] == 1
            assert config['max_new_tokens'] == 2048 and not metric['formal_test']
            reduced = {k: v for k, v in config.items() if k not in ['adapter', 'output']}
            if canonical is None:
                canonical, reference = reduced, rows
            assert canonical == reduced
            assert len(rows) == len({r['id'] for r in rows}) == metric['items'] == 100
            assert all(len(r['samples']) == 1 and len(r['samples'][0]['token_ids']) == r['samples'][0]['tokens'] for r in rows)
            assert sum(r['samples'][0]['correct'] for r in rows) == metric['single_correct']
            assert sum(r['samples'][0]['tokens'] for r in rows) == metric['generated_tokens']
            for base, current in zip(reference, rows):
                assert all(base[k] == current[k] for k in ['id', 'question', 'answer', 'prompt'])
                assert base['samples'][0]['seed'] == current['samples'][0]['seed']
            if index == 0:
                ids, answers = [r['id'] for r in rows], [r['answer'] for r in rows]
                if common_ids is None:
                    common_ids, common_answers = ids, answers
                assert ids == common_ids and answers == common_answers
            if index == 2 or (index == 1 and lang not in ['sw', 'th']):
                fresh.append(metric)
        curves[lang] = [m['metrics']['single_correct'] for m in curve['grid']]
    assert len(fresh) == 10
    result = {'observed_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'protocol': 'project_dev_single_vllm_v1', 'grid': [0, 20, 50, 100],
        'correct_out_of100': curves, 'parent_queue_completed_exit0': True,
        'raw_ids_questions_answers_prompts_seeds_and_configs_paired': True,
        'same_original_question_ids_and_answers_across_languages': True,
        'raw_token_ids_lengths_and_correct_counts_recomputed': True,
        'new_inference': {'engine_processes': len(fresh), 'responses': sum(m['generation_count'] for m in fresh),
            'generated_tokens': sum(m['generated_tokens'] for m in fresh),
            'generation_seconds': sum(m['generation_seconds'] for m in fresh),
            'cached_Base_and100_and_SW_TH20_costs_excluded': True},
        'languages_with_any_observed_gain': [lang for lang, scores in curves.items() if max(scores[1:]) > scores[0]],
        'three_languages_selected': False, 'sequential_training_started': False,
        'scope': 'Held-out development only. Post-endpoint exploratory grid, one training seed per language, checkpoint selection bias; not stable-effectiveness or formal-test selection evidence. Earlier initialization/backend caveats remain documented. No new training.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
