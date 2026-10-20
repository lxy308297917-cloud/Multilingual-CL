"""Verify nine frozen development-selected checkpoint formal evaluations."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from summarize_completed_single_formal import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists()
    selection = read(a.selection)
    state = read(a.root/'gpu-queue/queues/copsd-devselected-formal-v1/state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['job']['args'] == [str(a.selection)]
    assert hashlib.sha256(Path(job['script']).read_bytes()).hexdigest() == job['script_sha256']
    for p, expected in selection['sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected, p
    root = a.root/'logs/copsd_official_10lang/evaluation_vllm'
    pairs, new_metrics = [], []
    for entry in selection['jobs']:
        benchmark, lang, training = [entry[k] for k in ['benchmark', 'language', 'training_language']]
        tag = '20261020_v2' if training == 'fr' else '20261020_v1'
        base = 'en_17b_gsm8k_base_20261020_v1' if lang == 'en' else f'{lang}_17b_{benchmark}_base_{tag}'
        paths = [root/base, root/entry['run_id']]
        configs = [read(p/'CONFIG.json') for p in paths]
        canonical = [{k: v for k, v in c.items() if k not in ['adapter', 'output_dir']} for c in configs]
        assert canonical[0] == canonical[1]
        assert configs[0]['adapter'] is None and Path(configs[1]['adapter']).resolve() == Path(entry['checkpoint']).resolve()
        assert configs[0]['benchmark'] == benchmark and configs[0]['language'] == lang
        assert configs[0]['protocol'] == 'project_single_vllm_v1'
        assert configs[0]['samples'] == 1 and configs[0]['max_new_tokens'] == 2048
        total = {'mgsm_rev2': 250, 'polymath': 500, 'gsm8k': 1319}[benchmark]
        rows = [[json.loads(s) for s in (p/'ITEMS.jsonl').read_text().splitlines()] for p in paths]
        metrics = [read(p/'DONE.json') for p in paths]
        for rr, m in zip(rows, metrics):
            assert len(rr) == len({r['id'] for r in rr}) == total == m['items']
            assert m['formal_full_test'] and m['n'] == 1
            assert sum(r['samples'][0]['correct'] for r in rr) == m['single_correct']
            assert sum(r['samples'][0]['tokens'] for r in rr) == m['generated_tokens']
        for x, y in zip(*rows):
            assert all(x[k] == y[k] for k in ['id', 'question', 'answer', 'prompt'])
            assert len(x['samples']) == len(y['samples']) == 1
        if benchmark == 'polymath':
            splits = {r['id']: r['split'] for r in read(Path(configs[0]['input']))}
            for rr, m in zip(rows, metrics):
                for level in ['low', 'medium', 'high', 'top']:
                    part = [r for r in rr if splits[r['id']] == level]
                    assert len(part) == m['scores_by_difficulty'][level]['items'] == 125
                    assert sum(r['samples'][0]['correct'] for r in part) == m['scores_by_difficulty'][level]['single_correct']
        step = read(Path(entry['checkpoint'])/'trainer_state.json')['global_step']
        assert any(x['language'] == training and x['step'] == step for x in selection['selections'])
        bc, ac = [[r['samples'][0]['correct'] for r in rr] for rr in rows]
        pairs.append({'training_language': training, 'checkpoint_step': step,
            'benchmark': benchmark, 'evaluation_language': lang, 'metrics': metrics,
            'delta_correct': metrics[1]['single_correct']-metrics[0]['single_correct'],
            'gained': sum(not x and y for x, y in zip(bc, ac)),
            'lost': sum(x and not y for x, y in zip(bc, ac)),
            'inputs_prompts_and_configs_paired': True,
            'same_frozen_per_question_seed_rule': True, 'per_sample_seed_field_saved': False,
            'Base_inference_reused': True})
        new_metrics.append(metrics[1])
    assert len(pairs) == 9
    result = {'observed_at': datetime.datetime.now().isoformat(),
        'selection_frozen_at': selection['frozen_at'], 'rule': selection['rule'],
        'selections': selection['selections'], 'queue_phase': 'completed', 'exit_code': 0,
        'protocol': 'project_single_vllm_v1', 'pairs': pairs,
        'new_cost': {'model_loads': 9, 'generated_responses': sum(m['generation_count'] for m in new_metrics),
            'generated_tokens': sum(m['generated_tokens'] for m in new_metrics),
            'generation_seconds': sum(m['generation_seconds'] for m in new_metrics),
            'evaluation_seconds_including_grading_output': sum(m['evaluation_seconds'] for m in new_metrics),
            'single_GPU_hours_job_clock': (datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()/3600,
            'all_Base_costs_excluded': True, 'new_training': False},
        'scope': selection['scope']+' Original fixed100 formal results retained; no sequential models or stable cross-seed evidence.'}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([(p['training_language'], p['benchmark'], p['metrics'][0]['single_correct'], p['metrics'][1]['single_correct']) for p in pairs]))


if __name__ == '__main__':
    main()
