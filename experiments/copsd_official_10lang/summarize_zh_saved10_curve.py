"""Audit the prespecified ZH Base/10/20 independent-development comparison.

Uses two existing training histories and cached Base/20 results. Does not
choose a favorable checkpoint, change stage budgets or submit experiments.
"""
import argparse
import datetime
import json
from pathlib import Path

from disk_recovery_driver import audit_stage
from summarize_base8192 import RUNS
from summarize_saved_candidates8192 import pair
from sequential_evaluation import digest, read, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    code = Path(__file__).parent
    frozen = read(code / 'zh_saved10_development_frozen_20261021.PUBLIC.json')
    for name, sha in frozen['code_sha256'].items():
        assert digest(code / name) == sha
    folder = a.root / 'logs/copsd_official_10lang/development_vllm8192'
    def load(run, lang, adapter=None, step=None):
        return audit_stage({'directory': str(folder / run), 'language': lang,
            'adapter': str(adapter) if adapter else None, 'step': step,
            'protocol': frozen['protocol'], 'budget': 8192, 'evaluator': 'evaluate_dev8192.py'}, a.root, code)
    base = {lang: load(RUNS[lang], lang) for lang in ['zh', 'en']}
    results, sources, fresh, hours = {}, {}, [], 0
    for model, entry in frozen['models'].items():
        state = read(a.root / 'gpu-queue/queues' / entry['queue'] / 'state.json')
        assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
        job = state['job']
        assert job['config']['job']['args'] == [model] and job['config']['max_user_gpus'] == 4
        assert len(job['gpu_uuids']) == 1
        assert Path(job['script']).resolve() == code / 'run_zh_saved10_dev8192.sh'
        assert job['script_sha256'] == frozen['code_sha256']['run_zh_saved10_dev8192.sh']
        cp10 = a.root / entry['checkpoint_relative']
        assert digest(cp10 / 'adapter_model.safetensors') == entry['adapter_sha256']
        cp20 = cp10.parent / 'checkpoint-20'
        old = read(code / ('zh_saved20_dev8192_completed_20261020.PUBLIC.json' if model == 'seed42'
            else 'zh_seed43_prefix20_completed_20261021.PUBLIC.json'))
        run20 = ({'zh': 'zh_early20_candidate_dev100_8192_20261020_v1',
            'en': 'zh_early20_en_candidate_dev100_8192_20261020_v1'} if model == 'seed42' else
            {'zh': 'zh_17b_seed43_prefix20_dev100_8192_20261021_v1',
             'en': 'en_zh_seed43_prefix20_dev100_8192_20261021_v1'})
        results[model], sources[model] = {}, {}
        for lang in ['zh', 'en']:
            ten = load(entry['run_ids'][lang], lang, cp10, 10)
            twenty = load(run20[lang], lang, cp20, 20)
            key = 'target' if lang == 'zh' else 'English'
            assert twenty[1]['single_correct'] == old['paired_development100'][key]['adapter20_correct']
            assert base[lang][1]['single_correct'] == old['paired_development100'][key]['base_correct']
            comparisons = {}
            for name, left, right in [('Base_to10', base[lang], ten),
                    ('Base_to20', base[lang], twenty), ('10_to20', ten, twenty)]:
                paired = pair((*left[:3], None), (*right[:3], None))
                comparisons[name] = {'from_correct': paired['base_correct'],
                    'to_correct': paired['adapter20_correct'],
                    **{k: paired[k] for k in ['gained', 'lost', 'delta_correct']}}
            # Same original questions/gold in both languages, same per-language protocol across steps.
            assert [r['id'] for r in ten[2]] == [r['id'] for r in base['zh'][2]]
            assert [str(r['answer']) for r in ten[2]] == [str(r['answer']) for r in base['zh'][2]]
            results[model][lang] = {'correct_out_of100': {'0': base[lang][1]['single_correct'],
                '10': ten[1]['single_correct'], '20': twenty[1]['single_correct']},
                'paired_comparisons': comparisons, 'metrics10': ten[1]}
            sources[model][lang] = {step: {'run_id': run,
                'raw_items_sha256': digest(folder / run / 'ITEMS.jsonl'),
                'config_sha256': digest(folder / run / 'CONFIG.json')}
                for step, run in [('0', RUNS[lang]), ('10', entry['run_ids'][lang]), ('20', run20[lang])]}
            fresh.append(ten[1])
        hours += (datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds() / 3600
    write_new(a.output, {'observed_at': datetime.datetime.now().isoformat(), 'protocol': frozen['protocol'],
        'prespecified_checkpoint_grid': frozen['fixed_checkpoint_grid'], 'results': results,
        'source_records': sources, 'parents_completed_exit0': True,
        'new_cost_excluding_cached_Base_and20': {'responses': 400,
            'generated_tokens': sum(m['generated_tokens'] for m in fresh),
            'generation_seconds': sum(m['generation_seconds'] for m in fresh),
            'GPU_hours_two_unique_parents': hours, 'new_training': False},
        'run_comparison_caveat': frozen['run_comparison_caveat'],
        'selection_policy': frozen['selection_policy'],
        'scope': 'Finite development-only within-run checkpoint diagnosis. Does not establish a common10-update benefit across languages, replace the20-update comparison, choose final languages or generate a forgetting matrix. Changing the subsequent common stage budget requires matched controls.'})
    print(json.dumps({m: {l: v['correct_out_of100'] for l, v in rr.items()} for m, rr in results.items()}))


if __name__ == '__main__':
    main()
