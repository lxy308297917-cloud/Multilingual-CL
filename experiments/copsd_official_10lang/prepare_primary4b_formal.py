"""Bind a primary single20 development checkpoint to full formal tests.

This prepares missing ten-language reporting without retraining or selecting
models from test scores. Existing selected-stage training/evaluation files are
never modified. Queue creation and GPU submission remain separate operations.
"""
import argparse
import copy
import json
from pathlib import Path

from formal4b_contract import LANGUAGES, REVISION, digest, prepare, read
from sequential4b_selection import development_snapshot, evidence_counts, local


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--language', choices=sorted(LANGUAGES), required=True)
    p.add_argument('--development-report', type=Path, required=True)
    p.add_argument('--development-config', type=Path, required=True)
    p.add_argument('--benchmarks', nargs='+', choices=['mgsm_rev2', 'polymath', 'gsm8k'],
                   default=['mgsm_rev2', 'polymath', 'gsm8k'])
    p.add_argument('--freeze', type=Path, required=True)
    a = p.parse_args()
    root, code = a.root.resolve(), Path(__file__).parent
    assert not a.freeze.exists() and len(a.benchmarks) == len(set(a.benchmarks))
    template = read(a.template)
    assert template['model_id'] == 'Base'
    selected = read(local(root, template['selection_record_relative']))
    snapshot, models = development_snapshot(local(root, selected['development_snapshot_relative']))
    report = read(a.development_report)
    assert report['TensorBoard_four_accuracies_two_costs_read_back']
    counts = evidence_counts(report, a.language, 42)
    expected = models[a.language, 42]
    assert counts == [expected[k] for k in ['target_Base_correct_of200', 'target_OPD_correct_of200',
                                           'English_Base_correct_of200', 'English_OPD_correct_of200']]
    trained, = [k for k in report['languages'][a.language]['conditions'] if k != 'Base']
    source = report['languages'][a.language]['conditions'][trained]['source']
    assert a.development_config.parent.name == source['run_id']
    assert digest(a.development_config) == source['config_sha256']
    assert digest(a.development_config.with_name('ITEMS.jsonl')) == source['items_sha256']
    config = read(a.development_config)
    assert config['model_revision'] == REVISION and config['language'] == a.language
    assert config['checkpoint_step'] == 20 and config['items'] == 200
    adapter = Path(config['adapter'])
    relative = str(adapter.relative_to(root))
    assert digest(adapter / 'adapter_model.safetensors') == config['adapter_sha256']
    f = copy.deepcopy(template)
    for key in ['cached_cases', 'Base_cache_audit_relative', 'Base_cache_audit_sha256', 'cache_accounting']:
        f.pop(key, None)
    model_id = 'Primary_' + a.language.upper() + '20'
    f.update(model_id=model_id, experiment_role='independent_primary_single20_formal_reporting',
             output_relative='logs/copsd_official_10lang/primary4b_formal_20261024_v1',
             model_provenance={
                 'development_report_relative': str(a.development_report.relative_to(root)),
                 'development_report_sha256': digest(a.development_report),
                 'development_config_relative': str(a.development_config.relative_to(root)),
                 'development_config_sha256': digest(a.development_config),
                 'primary_training_seed': 42,
                 'checkpoint_choice': 'Pre-existing primary20; no formal-test selection.',
                 'cohort_note': 'Original2700-pool independent single, not selected900-pool matched reference.'})
    f['runtime_code_sha256']['prepare_primary4b_formal.py'] = digest(Path(__file__))
    f['cases'] = {}
    for benchmark in a.benchmarks:
        lang = 'en' if benchmark == 'gsm8k' else a.language
        name = benchmark + '_' + lang
        f['cases'][name] = dict(benchmark=benchmark, language=lang,
                               run_id=model_id + '_' + name,
                               adapter_relative=relative, adapter_sha256=config['adapter_sha256'],
                               expected_step=20)
    # Validate before publishing the final filename, leaving no launchable plan
    # when model/data/protocol validation fails.
    temporary = a.freeze.with_name(a.freeze.name + '.preparing')
    with temporary.open('x') as stream:
        json.dump(f, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    _, cases = prepare(root, temporary)
    assert all(not case['output'].exists() for case in cases.values())
    temporary.rename(a.freeze)
    print(json.dumps({'model_id': model_id, 'CPU_preflight': True,
                      'cases': list(cases), 'GPU_started': False}))


if __name__ == '__main__':
    main()
