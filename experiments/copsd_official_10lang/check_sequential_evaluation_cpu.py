"""Read-only real-output checks for the new stage evaluation collector.

No candidate selection record, production plan or GPU is created. The three
real SW MGSM outputs are not three languages or a complete seven-case stage.
"""
import argparse
import datetime
import json
from pathlib import Path

from sequential_evaluation import audit_benchmark, digest, read, selection, validate_case_set, write_new


def rejects(function):
    try:
        function()
    except (AssertionError, KeyError, FileNotFoundError):
        return True
    raise AssertionError('Invalid evidence was accepted')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    code = Path(__file__).parent
    manifest = read(code / 'sw20_formal8192_frozen_20261021.PUBLIC.json')
    previous = read(code / 'sw20_mgsm_formal8192_stage_completed_20261021.PUBLIC.json')
    audited, cases, checkpoints = {}, {}, {}
    for stage in manifest['stages']:
        if stage['benchmark'] != 'mgsm_rev2':
            continue
        model = stage['model']
        folder = a.root / 'logs/copsd_official_10lang/evaluation_vllm8192' / stage['run_id']
        cfg = read(folder / 'CONFIG.json')
        case = {'benchmark': 'mgsm_rev2', 'language': 'sw', 'run_id': stage['run_id'],
                'input_sha256': cfg['input_sha256']}
        cp = a.root / stage['adapter_relative'] if stage['adapter_relative'] else None
        result = audit_benchmark(a.root, code, case, cp)
        assert result['metrics'] == previous['metrics'][model]
        assert result['raw_items_sha256'] == previous['sources'][model]['raw_items_sha256']
        cases[model], checkpoints[model], audited[model] = case, cp, result['correct']
    assert audited == {'base': 8, 'seed42': 13, 'seed43': 14}
    wrong_adapter = rejects(lambda: audit_benchmark(a.root, code, cases['seed42'], checkpoints['seed43']))
    changed_input = rejects(lambda: audit_benchmark(a.root, code,
        dict(cases['seed42'], input_sha256='0' * 64), checkpoints['seed42']))
    wrong_language = rejects(lambda: audit_benchmark(a.root, code,
        dict(cases['seed42'], language='ja'), checkpoints['seed42']))
    mixed_protocol = rejects(lambda: audit_benchmark(a.root, code,
        dict(cases['seed42'], protocol='project_formal_single_vllm8192_c64_v2'), checkpoints['seed42']))
    # Only checks the completeness gate, never constructs a language selection.
    incomplete = rejects(lambda: validate_case_set(['sw', 'zh', 'ja'], list(cases.values())))
    formal = code / 'ten_language_formal100_completed_20261021.PUBLIC.json'
    formal_selection = rejects(lambda: selection(formal, digest(formal)))
    write_new(a.output, {'observed_at': datetime.datetime.now().isoformat(),
        'real_saved_SW_MGSM_outputs_audited': 3, 'real_saved_responses': 750,
        'correct_counts_match_independently_saved_report': audited,
        'rejected': {'other_checkpoint_adapter': wrong_adapter,
            'changed_input_identity': changed_input, 'wrong_language': wrong_language,
            'old16_outputs_relabelled_as64_protocol': mixed_protocol,
            'incomplete_seven_case_stage': incomplete, 'formal_report_as_development_selection': formal_selection},
        'collector_sha256': digest(code / 'sequential_evaluation.py'),
        'check_script_sha256': digest(Path(__file__)),
        'GPU_used': False, 'new_language_model_generation': False, 'production_selection_created': False,
        'scope': 'Read-only CPU provenance/completeness validation on real saved MGSM outputs. Not a seven-benchmark stage run, checkpoint handoff, three-language selection, full sequential evaluation or forgetting result. Does not rerun mathematical grading.'})
    print(json.dumps({'counts': audited, 'all_negative_evidence_rejected': True, 'GPU_used': False}))


if __name__ == '__main__':
    main()
