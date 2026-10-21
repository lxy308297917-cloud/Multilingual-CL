"""Audit completed clean-context student/privileged-teacher diagnostic outputs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from evaluate_vllm import question_seed


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--queue', required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--failed-queue', help='Optional preceding startup failure; account its allocation separately.')
    a = p.parse_args()
    assert not a.output.exists()
    state = read(a.root / 'gpu-queue/queues' / a.queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert sha(Path(job['script'])) == job['script_sha256']
    prepared, cfg, result = [read(x) for x in [a.prepared / 'CONFIG.PRIVATE.json', a.run / 'CONFIG.PRIVATE.json', a.run / 'DONE.json']]
    assert all(cfg[k] == v for k, v in prepared.items())
    assert cfg['protocol'] == result['protocol'] == 'privileged_teacher_conditioning_diagnostic_v1'
    assert cfg['adapter'] is result['teacher_adapter'] is None and result['same_original_Base']
    assert cfg['samples'] == 1 and cfg['max_new_tokens'] == 2048
    assert cfg['temperature'] == 1 and cfg['top_p'] == .95 and cfg['top_k'] == -1 and cfg['seed'] == 42
    assert Path(cfg['model']).resolve() == (a.root / 'models/Qwen3-1.7B').resolve()
    source = Path(cfg['source'])
    assert sha(source / 'multilingual_data_collator.py') == cfg['collator_sha256']
    assert sha(source / 'language_config.py') == cfg['language_config_sha256']
    assert sha(Path(__file__).with_name('evaluate_teacher_conditioning.py')) == cfg['evaluator_sha256']
    assert sha(a.prepared / 'PROMPTS.PRIVATE.json') == cfg['prompts_sha256']
    prompts = read(a.prepared / 'PROMPTS.PRIVATE.json')
    assert len(prompts) == len({r['id'] for r in prompts}) == 100
    rows = {}
    hashes = {}
    for condition in ['student', 'teacher']:
        path = a.run / (condition + '_ITEMS.PRIVATE.jsonl')
        rr = [json.loads(s) for s in path.read_text().splitlines()]
        totals = result['conditions'][condition]
        assert totals == read(a.run / (condition + '_DONE.json'))
        assert len(rr) == totals['items'] == 100
        for r, prompt in zip(rr, prompts):
            assert r['id'] == prompt['id'] and r['answer'] == prompt['answer']
            assert r['seed'] == question_seed(42, cfg['language'], r['id'])
            assert r['finish_reason'] in ['stop', 'length']
            assert 0 < len(r['token_ids']) <= 2048
            if r['correct']: assert r['boxed_answer'] is not None
        for field, values in {
            'correct': [bool(r['correct']) for r in rr],
            'normal_terminations': [r['finish_reason'] == 'stop' for r in rr],
            'truncated': [r['finish_reason'] == 'length' for r in rr],
            'answer_extraction_failures': [r['boxed_answer'] is None for r in rr],
            'generated_tokens': [len(r['token_ids']) for r in rr],
        }.items(): assert sum(values) == totals[field], (condition, field)
        rows[condition] = rr
        hashes[condition] = sha(path)
    s, t = rows['student'], rows['teacher']
    hours = lambda j: (datetime.datetime.fromisoformat(j['finished_at']) - datetime.datetime.fromisoformat(j['started_at'])).total_seconds() / 3600
    cost = {'completed_parent_GPU_hours': hours(job), 'new_responses': 200,
            'generated_tokens': sum(x['generated_tokens'] for x in result['conditions'].values()),
            'generation_seconds': sum(x['generation_seconds'] for x in result['conditions'].values())}
    if a.failed_queue:
        failed = read(a.root / 'gpu-queue/queues' / a.failed_queue / 'state.json')
        assert failed['phase'] == 'failed' and failed['job']['exit_code'] != 0
        old = failed['job']; assert len(old['gpu_uuids']) == 1
        cost['preceding_failed_parent_GPU_hours'] = hours(old)
        cost['completed_plus_failed_parent_GPU_hours'] = hours(old) + hours(job)
    out = {**result, 'observed_at': datetime.datetime.now().isoformat(), 'queue': a.queue,
           'queue_phase': 'completed', 'exit_code': 0,
           'paired_counts': {'gained_under_privileged_context': sum(not x['correct'] and y['correct'] for x, y in zip(s, t)),
                             'lost_under_privileged_context': sum(x['correct'] and not y['correct'] for x, y in zip(s, t)),
                             'correct_under_both': sum(x['correct'] and y['correct'] for x, y in zip(s, t))},
           'source_records_sha256': hashes, 'cost': cost,
           'paired_IDs_answers_seeds_raw_counts_source_and_prompt_hashes_verified': True,
           'comparison_boundary': 'Both conditions use the official training collator and2048 tokens. Not comparable to8192 development scores or a fair reference-free teacher benchmark. Clean-context advantage does not establish supervision quality on on-policy wrong histories.',
           'new_training_or_SFT_labels': False,
           'scheduling': {'max_num_seqs_explicitly_passed': False, 'runtime_scheduler_capacity_saved': False,
                          'source_inferred_vllm011_LLM_CLASS_default_on_3090': 256,
                          'boundary': 'Official loader default used, not the explicit16 dev8192 evaluator. Initial TE freeze metadata16 was corrected as an erratum; do not claim runtime16.'}}
    with a.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'conditions': out['conditions'], 'paired': out['paired_counts'], 'cost': cost}, ensure_ascii=False))


if __name__ == '__main__': main()
