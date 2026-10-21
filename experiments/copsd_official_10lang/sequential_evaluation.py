"""Freeze/run/audit seven project evaluations after Base or a research stage.

Does not choose languages, start queues, train or retry subprocesses. Cached
stages must be specified before execution; scores never drive cache selection.
The final training-gate marker is written only after queue exit0 is observed.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

PROTOCOL = 'project_formal_single_vllm8192_v1'
PRESETS = {
    PROTOCOL: {'concurrency': 16, 'evaluator': 'evaluate_formal8192.py',
        'wrapper': 'run_formal8192.sh', 'preflight': 'formal8192_CPU_preflight_20261020.PUBLIC.json'},
    'project_formal_single_vllm8192_c64_v2': {'concurrency': 64,
        'evaluator': 'evaluate_formal8192_c64.py', 'wrapper': 'run_formal8192_c64.sh',
        'preflight': 'formal8192_c64_CPU_preflight_20261021.PUBLIC.json'},
}
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
LANGUAGES = set('sw te th bn zh ja ru de fr es'.split())
CODE_FILES = ['sequential_evaluation.py', 'run_sequential_evaluation.sh',
              'evaluate_formal8192.py', 'run_formal8192.sh',
              'evaluate_transformers.py', 'evaluate_vllm.py',
              'formal8192_CPU_preflight_20261020.PUBLIC.json',
              'evaluate_formal8192_c64.py', 'run_formal8192_c64.sh',
              'formal8192_c64_CPU_preflight_20261021.PUBLIC.json',
              'compat/vllm011_uuid/sitecustomize.py']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def selection(path, sha):
    assert digest(path) == sha
    s = read(path)
    assert s['source_protocol'] == 'independent_development_only'
    chosen = s['selected_languages']
    assert len(chosen) == len(set(chosen)) == 3 and set(chosen) <= LANGUAGES
    assert s['pool_by_language'] == {lang: i for i, lang in enumerate(chosen)}
    assert s['stage_updates'] == 20 and s['stage_evaluation_protocol'] in PRESETS
    assert re.fullmatch(r'[A-Za-z0-9_-]+', s['experiment_id'])
    assert s['evidence_reports']
    for evidence in s['evidence_reports']:
        assert digest(evidence['path']) == evidence['sha256']
        report = read(evidence['path'])
        assert report['protocol'] == 'project_dev_single_vllm8192_v2', 'Selection evidence must be development, not formal tests'
    return s


def required_cases(chosen):
    return [(bench, lang) for lang in chosen for bench in ['mgsm_rev2', 'polymath']] + [('gsm8k', 'en')]


def validate_case_set(chosen, cases):
    observed = [(c['benchmark'], c['language']) for c in cases]
    assert observed == required_cases(chosen) and len(observed) == len(set(observed)) == 7
    assert len({c['run_id'] for c in cases}) == 7


def benchmark_input(root, benchmark, language):
    if benchmark == 'mgsm_rev2':
        return root / f'data/math_single_10lang_20261015/mgsm_rev2/MGSM-Rev2/mgsm_{language}.tsv', 250
    if benchmark == 'polymath':
        return root / f'data/copsd_official_20261015/polymath_project_eval_v1/polymath_all500_{language}.json', 500
    assert benchmark == 'gsm8k' and language == 'en'
    return root / 'data/copsd_official_20261015/gsm8k_project_en_zeroshot_v1/gsm8k_en_test_1319.json', 1319


def audit_benchmark(root, code, case, checkpoint):
    """Verify one real output against frozen identity, inputs and raw counters."""
    from evaluate_transformers import load_items
    from evaluate_vllm import question_seed
    directory = root / 'logs/copsd_official_10lang/evaluation_vllm8192' / case['run_id']
    c, m = read(directory / 'CONFIG.json'), read(directory / 'DONE.json')
    protocol = case.get('protocol', PROTOCOL)
    preset = PRESETS[protocol]
    assert c['protocol'] == m['protocol'] == protocol
    assert c['benchmark'] == case['benchmark'] and c['language'] == case['language']
    assert c['model'] == str(root / 'models/Qwen3-1.7B') and c['model_revision'] == REVISION
    data, count = benchmark_input(root, case['benchmark'], case['language'])
    assert Path(c['input']).resolve() == data.resolve() and c['input_sha256'] == case['input_sha256'] == digest(data)
    assert c['evaluator_sha256'] == digest(code / preset['evaluator'])
    official = root / 'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    assert c['official_source_sha256'] == digest(official)
    release = official.with_name('run_eval_4b_all_checkpoints.sh')
    assert c['released_length_script_sha256'] == digest(release)
    assert c['samples'] == m['n'] == 1 and c['max_new_tokens'] == 8192
    assert c['temperature'] == 1 and c['top_p'] == .95 and c['top_k'] == -1
    assert c['seed'] == 42 and c['max_num_seqs'] == m['actual_max_num_seqs'] == preset['concurrency']
    assert c['model_context_limit'] == 40960
    if checkpoint:
        checkpoint = Path(checkpoint)
        assert Path(c['adapter']).resolve() == checkpoint.resolve() and c['checkpoint_step'] == 20
        assert c['adapter_sha256'] == digest(checkpoint / 'adapter_model.safetensors')
        assert read(checkpoint / 'trainer_state.json')['global_step'] == 20
    else:
        assert c['adapter'] is c['adapter_sha256'] is c['checkpoint_step'] is None
    kind = 'mgsm_rev2' if case['benchmark'] == 'mgsm_rev2' else 'polymath_project'
    inputs = list(load_items(data, kind, case['language']))
    rows = [json.loads(line) for line in (directory / 'ITEMS.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == len(inputs) == m['items'] == count
    assert m['formal_full_test'] and m['generation_count'] == count
    totals, levels = Counter(), {}
    if case['benchmark'] == 'polymath':
        levels = {r['id']: r['split'] for r in read(data)}
        assert Counter(levels.values()) == Counter({level: 125 for level in ['low', 'medium', 'high', 'top']})
    by_level = {}
    for row, item in zip(rows, inputs):
        assert all(row[k] == item[k] for k in ['id', 'question', 'answer'])
        assert len(row['samples']) == 1
        sample = row['samples'][0]
        assert type(sample['correct']) is bool
        assert sample['seed'] == question_seed(42, case['language'], row['id'])
        assert sample['tokens'] == len(sample['token_ids']) <= 8192
        totals['single_correct'] += sample['correct']
        totals['generated_tokens'] += sample['tokens']
        totals['truncated'] += sample['finish_reason'] == 'length'
        totals['normal_terminations'] += sample['finish_reason'] == 'stop'
        totals['answer_extraction_failures'] += sample['boxed_answer'] is None
        if levels:
            level = levels[row['id']]
            assert row['split'] == level
            subtotal = by_level.setdefault(level, Counter())
            subtotal['items'] += 1
            subtotal['single_correct'] += sample['correct']
    assert all(m[k] == value for k, value in totals.items())
    assert m['single_accuracy'] == totals['single_correct'] / count
    for level, subtotal in by_level.items():
        assert all(m['scores_by_difficulty'][level][k] == v for k, v in subtotal.items())
    return {'benchmark': case['benchmark'], 'language': case['language'], 'run_id': case['run_id'],
        'items': count, 'correct': totals['single_correct'], 'metrics': m,
        'config_sha256': digest(directory / 'CONFIG.json'), 'raw_items_sha256': digest(directory / 'ITEMS.jsonl'),
        'done_sha256': digest(directory / 'DONE.json'), 'input_sha256': digest(data),
        'canonical_config': {k: v for k, v in c.items() if k not in ['adapter', 'adapter_sha256', 'checkpoint_step']}}


def verify_plan(root, code, plan):
    s = selection(plan['selection_record'], plan['selection_record_sha256'])
    assert plan['protocol'] == s['stage_evaluation_protocol'] and plan['model_revision'] == REVISION
    preset = PRESETS[plan['protocol']]
    assert plan['code_sha256'] == {name: digest(code / name) for name in CODE_FILES}
    assert plan['direction'] in ['forward', 'reverse'] and 0 <= plan['stage_index'] <= 3
    order = s['selected_languages'] if plan['direction'] == 'forward' else list(reversed(s['selected_languages']))
    assert plan['order_id'] == s['experiment_id'] + '_' + plan['direction']
    validate_case_set(s['selected_languages'], plan['cases'])
    runtime_lock = read(root / 'logs/copsd_official_10lang/eval_vllm_runtime_20261019.CONFIG.json')
    assert runtime_lock['official_commit'] == '4f4410730e9b6775a30d20f4b6e141cba2e87590'
    assert digest(runtime_lock['model_lock']['path']) == runtime_lock['model_lock']['sha256']
    official = root / 'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    assert digest(official) == runtime_lock['source_sha256']['polymath_eval/evaluate_math.py']
    prepared = read(code / preset['preflight'])
    assert prepared['evaluator_sha256'] == digest(code / preset['evaluator'])
    for c in plan['cases']:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', c['run_id'])
        assert c.get('protocol', PROTOCOL) == plan['protocol']
        path, count = benchmark_input(root, c['benchmark'], c['language'])
        assert c['input_sha256'] == digest(path) and c['items'] == count
        assert c['input_sha256'] == runtime_lock['input_sha256'][str(path.relative_to(root))]
    checkpoint = Path(plan['checkpoint']) if plan['checkpoint'] else None
    if plan['stage_index'] == 0:
        assert checkpoint is None and plan['training_queue'] is None
    else:
        assert checkpoint and checkpoint.resolve().is_relative_to((root / 'models/checkpoint/copsd_sequential_stages_v1').resolve())
        assert digest(checkpoint / 'adapter_model.safetensors') == plan['checkpoint_adapter_sha256']
        assert read(checkpoint / 'trainer_state.json')['global_step'] == 20
        proof = read(checkpoint.parent / 'COLLOCATE_TRAIN_UPDATES_DONE.json')
        context = proof['sequential_context']
        assert not context['engineering_only'] and proof['base_unchanged_bitwise']
        assert context['selection_record_sha256'] == plan['selection_record_sha256']
        assert context['order_id'] == plan['order_id'] and context['stage_index'] == plan['stage_index']
        assert context['language'] == order[plan['stage_index'] - 1] and context['seed'] == s['training_seed']
        assert context['pool_index'] == s['pool_by_language'][context['language']]
        assert read(checkpoint.parent / 'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
        state = read(root / 'gpu-queue/queues' / plan['training_queue'] / 'state.json')
        assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
        arguments = state['job']['config']['job']['args']
        assert len(arguments) == 1
        config = read(arguments[0])
        assert config['run_id'] == checkpoint.parent.name and not config.get('engineering_only', False)
        assert digest(arguments[0]) == read(checkpoint.parent / 'STAGE_CONTEXT.PRIVATE.json')['stage_spec_sha256']
    return s, checkpoint


def create_plan(root, code, a):
    s = selection(a.selection, a.selection_sha)
    assert re.fullmatch(r'[A-Za-z0-9_-]+', a.run_label)
    assert a.direction in ['forward', 'reverse'] and 0 <= a.stage <= 3
    assert (a.stage == 0) == (a.checkpoint is None)
    assert (a.stage == 0) == (a.training_queue is None)
    cache = read(a.cache) if a.cache else {}
    allowed = {f'{bench}:{lang}' for bench, lang in required_cases(s['selected_languages'])}
    assert set(cache) <= allowed
    cases = []
    for benchmark, language in required_cases(s['selected_languages']):
        key = f'{benchmark}:{language}'
        path, count = benchmark_input(root, benchmark, language)
        cases.append({'benchmark': benchmark, 'language': language, 'items': count,
            'protocol': s['stage_evaluation_protocol'],
            'input_sha256': digest(path), 'cached_before_execution': key in cache,
            'run_id': cache.get(key, f'{a.run_label}_{benchmark}_{language}')})
    checkpoint = a.checkpoint.resolve() if a.checkpoint else None
    plan = {'selection_record': str(a.selection.resolve()), 'selection_record_sha256': a.selection_sha,
        'protocol': s['stage_evaluation_protocol'], 'model_revision': REVISION, 'direction': a.direction,
        'stage_index': a.stage, 'order_id': s['experiment_id'] + '_' + a.direction,
        'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_adapter_sha256': digest(checkpoint / 'adapter_model.safetensors') if checkpoint else None,
        'training_queue': a.training_queue, 'cases': cases,
        'code_sha256': {name: digest(code / name) for name in CODE_FILES}}
    verify_plan(root, code, plan)
    for case in cases:
        path = root / 'logs/copsd_official_10lang/evaluation_vllm8192' / case['run_id']
        if case['cached_before_execution']:
            audit_benchmark(root, code, case, checkpoint)
        else:
            assert not path.exists(), 'Existing output must be explicitly frozen as cache or get a new run label'
    write_new(a.plan, plan)
    print(json.dumps({'frozen_plan': str(a.plan), 'cases': 7,
        'cached_cases': sum(c['cached_before_execution'] for c in cases), 'GPU_started': False}))


def run_plan(root, code, path, prepare_only):
    plan = read(path)
    _, checkpoint = verify_plan(root, code, plan)
    assert shutil.disk_usage(root).free >= 20 * 2**30
    commands = []
    for case in plan['cases']:
        destination = root / 'logs/copsd_official_10lang/evaluation_vllm8192' / case['run_id']
        if case['cached_before_execution']:
            audit_benchmark(root, code, case, checkpoint)
            continue
        assert not destination.exists(), 'Preserve partial output; recover with a new plan and explicit completed cache'
        command = ['bash', str(code / PRESETS[plan['protocol']]['wrapper']), case['benchmark'], case['language'], case['run_id']]
        if checkpoint:
            command += [str(checkpoint), '20']
        commands.append((case, command))
    if prepare_only:
        # One shared import; do not cold-start the same Python stack seven times.
        from evaluate_transformers import load_items
        for case in plan['cases']:
            data, count = benchmark_input(root, case['benchmark'], case['language'])
            kind = 'mgsm_rev2' if case['benchmark'] == 'mgsm_rev2' else 'polymath_project'
            items = list(load_items(data, kind, case['language']))
            assert len(items) == len({x['id'] for x in items}) == count
        print(json.dumps({'CPU_preflight': True, 'new_cases': len(commands), 'GPU_started': False}))
        return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    for case, command in commands:
        assert shutil.disk_usage(root).free >= 20 * 2**30
        subprocess.run(command, check=True)
        audit_benchmark(root, code, case, checkpoint)
    # This is not the training gate: the parent queue has not exited yet.
    write_new(Path(path).with_suffix('.STAGES_DONE.json'), {'plan_sha256': digest(path),
        'all_seven_stages_completed': True, 'parent_exit0_still_required': True})


def collect(root, code, path, queue, output):
    plan = read(path)
    _, checkpoint = verify_plan(root, code, plan)
    state = read(root / 'gpu-queue/queues' / queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    job = state['job']
    assert Path(job['script']).resolve() == (code / 'run_sequential_evaluation.sh').resolve()
    assert digest(job['script']) == job['script_sha256']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert [str(Path(x).resolve()) for x in job['config']['job']['args']] == [str(path.resolve())]
    assert read(path.with_suffix('.STAGES_DONE.json'))['plan_sha256'] == digest(path)
    expected = checkpoint.parent / 'STAGE_EVALUATIONS_DONE.json' if checkpoint else path.parent / 'STAGE_EVALUATIONS_DONE.json'
    assert output.resolve() == expected.resolve()
    completed = [audit_benchmark(root, code, case, checkpoint) for case in plan['cases']]
    fresh = [row for row, case in zip(completed, plan['cases']) if not case['cached_before_execution']]
    wall = (datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds()
    write_new(output, {'completed': True, 'selection_record_sha256': plan['selection_record_sha256'],
        'protocol': plan['protocol'], 'order_id': plan['order_id'], 'stage_index': plan['stage_index'],
        'checkpoint_adapter_sha256': plan['checkpoint_adapter_sha256'],
        'plan_sha256': digest(path), 'plan_path_PRIVATE': str(path), 'evaluation_parent_exit_code': 0,
        'completed_benchmarks': completed,
        'new_evaluation_cost': {'responses': sum(x['items'] for x in fresh),
            'generated_tokens': sum(x['metrics']['generated_tokens'] for x in fresh),
            'generation_seconds': sum(x['metrics']['generation_seconds'] for x in fresh),
            'parent_GPU_hours_counted_once': wall / 3600,
            'cached_cases_excluded': sum(c['cached_before_execution'] for c in plan['cases'])},
        'scope': 'All seven frozen project evaluations verified after parent exit0. Original outputs remain authoritative. Does not choose languages, rescore/filter by language, claim official pass12 or compute a complete sequence before all stages exist.'})
    print(json.dumps({'completed_marker': str(output), 'benchmarks': 7, 'new_cases': len(fresh)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    sub = parser.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--selection', type=Path, required=True)
    plan.add_argument('--selection-sha', required=True)
    plan.add_argument('--direction', choices=['forward', 'reverse'], required=True)
    plan.add_argument('--stage', type=int, choices=range(4), required=True)
    plan.add_argument('--checkpoint', type=Path)
    plan.add_argument('--training-queue')
    plan.add_argument('--cache', type=Path)
    plan.add_argument('--run-label', required=True)
    plan.add_argument('--plan', type=Path, required=True)
    for action in ['prepare', 'run', 'collect']:
        child = sub.add_parser(action)
        child.add_argument('--plan', type=Path, required=True)
        if action == 'collect':
            child.add_argument('--queue', required=True)
            child.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).parent
    if args.action == 'plan':
        create_plan(args.root, code, args)
    elif args.action == 'collect':
        collect(args.root, code, args.plan, args.queue, args.output)
    else:
        run_plan(args.root, code, args.plan, args.action == 'prepare')


if __name__ == '__main__':
    main()
