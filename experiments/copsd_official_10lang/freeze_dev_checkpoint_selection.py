"""Freeze exploratory checkpoint selection using development results only."""
import argparse
import hashlib
import json
from pathlib import Path
import datetime


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private-output', type=Path, required=True)
    p.add_argument('--public-output', type=Path, required=True)
    a = p.parse_args()
    assert not a.private_output.exists() and not a.public_output.exists()
    code = Path(__file__).parent
    selections, jobs, locks = [], [], {}
    dev = a.root/'logs/copsd_official_10lang/development_vllm'
    for lang in ['sw', 'th', 'bn', 'zh', 'de', 'fr']:
        curve = code/(lang+'_dev_fixed_curve_20261020.PUBLIC.json')
        grid = json.loads(curve.read_text())
        assert grid['paired_inputs_prompts_seeds_configs_verified']
        assert [g['step'] for g in grid['grid']] == [0, 20, 50, 100]
        best = max(grid['grid'], key=lambda g: (g['metrics']['single_correct'], -g['step']))
        selected = {'language': lang, 'step': best['step'], 'development_correct': best['metrics']['single_correct'],
            'development_Base_correct': grid['grid'][0]['metrics']['single_correct']}
        selections.append(selected)
        locks[str(curve)] = digest(curve)
        if best['step'] == 0:
            continue
        matches = list(dev.glob(f'{lang}_17b_dev100_vllm_ckpt{best["step"]}_20261020_curve_v1'))
        assert len(matches) == 1
        config = matches[0]/'CONFIG.json'
        cp = Path(json.loads(config.read_text())['adapter'])
        assert json.loads((cp/'trainer_state.json').read_text())['global_step'] == best['step']
        for f in ['adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json']:
            locks[str(cp/f)] = digest(cp/f)
        locks[str(config)] = digest(config)
        for benchmark, evaluation_language in [('mgsm_rev2', lang), ('polymath', lang), ('gsm8k', 'en')]:
            run_id = f'{evaluation_language}_17b_{benchmark}_{lang}_devselected{best["step"]}_20261020_v1'
            assert not (a.root/'logs/copsd_official_10lang/evaluation_vllm'/run_id).exists()
            jobs.append({'benchmark': benchmark, 'language': evaluation_language,
                         'training_language': lang, 'run_id': run_id, 'checkpoint': str(cp)})
    assert [(s['language'], s['step']) for s in selections if s['step']] == [('sw', 50), ('zh', 20), ('fr', 20)]
    for name in ['run_dev_selected_formal.sh', 'run_vllm_math_eval.sh', 'evaluate_vllm.py', 'freeze_vllm_eval.py']:
        locks[str(code/name)] = digest(code/name)
    report = {'frozen_at': datetime.datetime.now().isoformat(), 'rule': 'Highest independent development correct on0/20/50/100; ties use earliest step, including Base.',
        'selections': selections, 'new_formal_runs': len(jobs), 'new_training': False,
        'scope': 'Exploratory secondary analysis after fixed100 regressions were known. Checkpoints selected only by development scores; no selected-checkpoint formal outputs existed at freezing. Not independent confirmatory replication, stable language selection or sequential training.'}
    a.private_output.write_text(json.dumps({**report, 'jobs': jobs, 'sha256': locks}, indent=2)+'\n')
    a.public_output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
