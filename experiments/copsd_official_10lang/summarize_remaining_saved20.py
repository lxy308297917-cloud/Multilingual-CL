"""Summarize one completed, frozen saved20 language screening pipeline."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from summarize_saved_candidates8192 import load_eval, pair, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--language', choices=['de', 'es', 'ru', 'th', 'te', 'ja'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    root, lang = a.root, a.language
    script = Path(__file__).parent
    if lang == 'ja':
        frozen = read(script / 'ja_saved20_screen8192_frozen_20261020.PUBLIC.json')
        cp = root / ('models/checkpoint/copsd_vllm_ja_data_repair_v4/'
                     'ja_17b_project2900_prefixrepair_nonefixed_seed42_20261020_v4/checkpoint-20')
        queue = frozen['queue']
        names = dict(frozen['run_ids'])
        names['base'] = names.pop('Base')
        caveat = 'New v4 reference with the previously authorized single-record prefix repair. Full100 training is separate and need not be finished to evaluate this saved checkpoint.'
        adapter_sha = frozen['adapter_sha256']
    else:
        frozen = read(script / 'remaining_saved20_screen8192_frozen_20261020.PUBLIC.json')
        spec = frozen['languages'][lang]
        cp = root / 'models/checkpoint' / spec['checkpoint_relative']
        queue, names, adapter_sha = spec['queue'], spec['run_ids'], spec['adapter_sha256']
        caveat = frozen['historical_training_caveats']['de_es_ru' if lang in ['de', 'es', 'ru'] else lang]
    assert hashlib.sha256((cp / 'adapter_model.safetensors').read_bytes()).hexdigest() == adapter_sha
    for name, expected in frozen['code_sha256'].items():
        assert hashlib.sha256((script / name).read_bytes()).hexdigest() == expected, name
    runs = {role: load_eval(root, queue, name, 'en' if role == 'English20' else lang,
                            None if role == 'base' else cp) for role, name in names.items()}
    english = load_eval(root, 'copsd-en-sw-base-dev8192-v1',
                        'en_sw_base_dev100_8192_20261020_v1', 'en', None)
    for records in zip(*[x[2] for x in runs.values()], english[2]):
        assert len({r['id'] for r in records}) == len({r['answer'] for r in records}) == 1
    assert len({x[3] for x in runs.values()}) == 1
    pairs = {'target': pair(runs['base'], runs['target20']),
             'English': pair(english, runs['English20'])}
    out = {
        'observed_at': datetime.datetime.now().isoformat(),
        'protocol': 'project_dev_single_vllm8192_v2',
        'training_language': lang, 'checkpoint_step': 20,
        'paired_development100': pairs,
        'metrics': {k: x[1] for k, x in runs.items()},
        'cached_English_Base_metrics': english[1],
        'frozen_sources_checkpoint_raw_counts_configs_prompts_seeds_completed_exit0_verified': True,
        'historical_training_caveat': caveat,
        'cost': {
            'new_responses': 300,
            'new_generated_tokens': sum(x[1]['generated_tokens'] for x in runs.values()),
            'new_generation_seconds': sum(x[1]['generation_seconds'] for x in runs.values()),
            'new_allocated_GPU_hours_unique_parent_once': runs['base'][3],
            'cached_English_Base_excluded': True, 'new_training': False,
            'scope': 'One parent covers fresh target Base, saved20 target and saved20 English. Count its allocated clock once, not once per evaluation.'},
        'scope': 'Exploratory fixed20 development screen at a common8192 budget. Not a complete checkpoint grid, officialpass12, final language selection or independent formal-test confirmation. Historical training differences are preserved.'}
    with a.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(pairs, ensure_ascii=False))


if __name__ == '__main__':
    main()
