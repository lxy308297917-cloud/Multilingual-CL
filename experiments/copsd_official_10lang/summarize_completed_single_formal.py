"""Pair completed fixed100 formal reports and count only newly executed runs."""
import argparse
import datetime
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home())
    parser.add_argument('--language', choices='sw te th bn zh ja ru de fr es'.split(), required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert all(c.isalnum() or c in '_-' for c in args.tag) and args.tag
    assert all(c.isalnum() or c == '-' for c in args.queue) and args.queue
    state = read(args.root / 'gpu-queue/queues' / args.queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    assert len(state['job']['gpu_uuids']) == 1
    started = datetime.datetime.strptime(state['job']['started_at'], '%Y-%m-%d %H:%M:%S')
    collected = datetime.datetime.strptime(state['updated_at'], '%Y-%m-%d %H:%M:%S')
    job_clock = (collected - started).total_seconds()
    assert job_clock >= 0
    frozen_args = state['job']['config']['job']['args']
    assert frozen_args[:2] == [args.language, args.tag] and len(frozen_args) == 3
    checkpoint = Path(frozen_args[2]).resolve()
    assert checkpoint.is_relative_to((args.root / 'models/checkpoint').resolve())
    assert read(checkpoint / 'trainer_state.json')['global_step'] == 100
    base = args.root / 'logs/copsd_official_10lang/evaluation_vllm'
    pairs, new_metrics = [], []
    for benchmark, language, total in [('mgsm_rev2', args.language, 250),
                                       ('polymath', args.language, 500), ('gsm8k', 'en', 1319)]:
        if benchmark == 'gsm8k':
            names = ['en_17b_gsm8k_base_20261020_v1', f'en_17b_gsm8k_{args.language}100_{args.tag}']
        else:
            names = [f'{language}_17b_{benchmark}_{mode}_{args.tag}' for mode in ['base', 'ckpt100']]
        paths = [base / name for name in names]
        cfg = [read(p / 'CONFIG.json') for p in paths]
        canonical = [{k: v for k, v in c.items() if k not in ['adapter', 'output_dir']} for c in cfg]
        assert canonical[0] == canonical[1]
        assert cfg[0]['adapter'] is None and Path(cfg[1]['adapter']).resolve() == checkpoint
        assert cfg[0]['language'] == language and cfg[0]['benchmark'] == benchmark
        assert cfg[0]['protocol'] == 'project_single_vllm_v1'
        assert cfg[0]['samples'] == 1 and cfg[0]['max_new_tokens'] == 2048
        rows = [[json.loads(line) for line in (p / 'ITEMS.jsonl').read_text().splitlines()] for p in paths]
        assert all(len(rr) == total and len({r['id'] for r in rr}) == total for rr in rows)
        for first, second in zip(*rows):
            assert all(first[k] == second[k] for k in ['id', 'question', 'answer', 'prompt'])
            assert len(first['samples']) == len(second['samples']) == 1
        metrics = [read(p / 'DONE.json') for p in paths]
        for rr, m in zip(rows, metrics):
            assert m['formal_full_test'] and m['items'] == total and m['n'] == 1
            assert sum(r['samples'][0]['correct'] for r in rr) == m['single_correct']
            assert sum(r['samples'][0]['tokens'] for r in rr) == m['generated_tokens']
        if benchmark == 'polymath':
            splits = {r['id']: r['split'] for r in read(Path(cfg[0]['input']))}
            for rr, m in zip(rows, metrics):
                for level in ['low', 'medium', 'high', 'top']:
                    subset = [r for r in rr if splits[r['id']] == level]
                    assert len(subset) == 125
                    assert sum(r['samples'][0]['correct'] for r in subset) == m['scores_by_difficulty'][level]['single_correct']
        bo, tr = [[r['samples'][0]['correct'] for r in rr] for rr in rows]
        pairs.append({'benchmark': benchmark, 'evaluation_language': language, 'items': total,
            'metrics': metrics, 'delta_correct': metrics[1]['single_correct'] - metrics[0]['single_correct'],
            'delta_percentage_points': 100 * (metrics[1]['single_accuracy'] - metrics[0]['single_accuracy']),
            'paired_counts': {'gained': sum(not a and b for a, b in zip(bo, tr)),
                'lost': sum(a and not b for a, b in zip(bo, tr)), 'retained': sum(a and b for a, b in zip(bo, tr))},
            'configs_equal_except_adapter_output': True, 'problem_ids_questions_answers_prompts_match': True,
            'same_frozen_per_question_seed_rule': True,
            'per_sample_seed_field_saved': all('seed' in r['samples'][0] for rr in rows for r in rr),
            'difficulty_counts_recomputed_from_items': benchmark == 'polymath',
            'Base_reused_from_existing_formal_run': benchmark == 'gsm8k'})
        new_metrics.extend(metrics[1:] if benchmark == 'gsm8k' else metrics)
    result = {'observed_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'training_language': args.language, 'model': 'Qwen3-1.7B', 'checkpoint_selection': 'Fixed100-update reference',
        'protocol': 'project_single_vllm_v1', 'formal_full_tests': True,
        'parent_suite_phase': 'completed', 'parent_suite_exit_code': 0, 'pairs': pairs,
        'new_run_cost': {'engine_processes': len(new_metrics),
            'generated_responses': sum(m['generation_count'] for m in new_metrics),
            'generated_tokens': sum(m['generated_tokens'] for m in new_metrics),
            'generation_seconds': sum(m['generation_seconds'] for m in new_metrics),
            'evaluation_seconds_including_grading_and_output': sum(m['evaluation_seconds'] for m in new_metrics),
            'cached_English_Base_generation_not_repeated': True,
            'job_wall_seconds_to_queue_collection_approx': job_clock,
            'single_GPU_hours_job_clock_approx': job_clock / 3600,
            'scope': 'Five new evaluators. Parent job clock includes startup/grading and queue exit collection; excludes reservation preparation and queue waiting. Engine generation/evaluation timings exclude model cold startup. Evaluation includes output, not isolated grader CPU time.'},
        'scope': 'Frozen formal report only; not language/checkpoint selection or sequential forgetting evidence. Cached English Base is excluded from new costs.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'language': args.language, 'pairs': [(p['benchmark'], p['delta_correct']) for p in pairs]}))


if __name__ == '__main__':
    main()
