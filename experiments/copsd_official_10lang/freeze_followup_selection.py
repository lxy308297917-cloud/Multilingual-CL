"""Freeze a real supported three-language choice after math/output evidence exists.

The assistant performs the evidence review; no additional user approval is needed.
Rejects incomplete or ineligible evidence before creating a production record.
"""
import argparse
import datetime
import json
from pathlib import Path
import re
import tempfile
from followup_selection_guard import digest, read, validate_math_evidence, validate_pool_binding
from sequential_evaluation import PRESETS, selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--languages', nargs=3, required=True)
    p.add_argument('--math-report', type=Path, required=True)
    p.add_argument('--output-review', type=Path, required=True)
    p.add_argument('--private-output', type=Path, required=True)
    p.add_argument('--public-output', type=Path, required=True)
    p.add_argument('--experiment-id', default='copsd_followup200_seq3_seed42_v1')
    a = p.parse_args()
    assert not a.private_output.exists() and not a.public_output.exists()
    assert len(set(a.languages)) == 3 and re.fullmatch(r'[A-Za-z0-9_-]+', a.experiment_id)
    code = Path(__file__).parent
    freeze = code / 'followup200_frozen_20261021.PUBLIC.json'
    pool = a.root / 'data/copsd_official_20261015/sequential_pools_followup200_20261021_v1/POOLS.PRIVATE.json'
    spec = {'frozen_at': datetime.datetime.now().isoformat(), 'experiment_id': a.experiment_id,
        'source_protocol': 'independent_development_only', 'selected_languages': a.languages,
        'pool_by_language': {lang: i for i, lang in enumerate(a.languages)},
        'stage_updates': 20, 'training_seed': 42,
        'stage_evaluation_protocol': 'project_formal_single_vllm8192_c64_v2',
        'followup_data_freeze_sha256': digest(freeze), 'pool_manifest': str(pool.resolve()),
        'pool_manifest_sha256': digest(pool), 'followup_guard_sha256': digest(code / 'followup_selection_guard.py'),
        'selection_validator_sha256': digest(code / 'sequential_evaluation.py'),
        'evidence_reports': [{'path': str(a.math_report.resolve()), 'sha256': digest(a.math_report)}]}
    # Math gate precedes temporary/production output creation and review-file loading.
    validate_pool_binding(spec, code)
    validate_math_evidence(spec, read(a.math_report), code)
    spec['output_review_report'] = {'path': str(a.output_review.resolve()), 'sha256': digest(a.output_review)}
    assert spec['stage_evaluation_protocol'] in PRESETS
    with tempfile.TemporaryDirectory(prefix='copsd-supported-selection-') as temp:
        test = Path(temp) / 'validated_selection.json'
        test.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
        selection(test, digest(test))
        # Write only the already validated exact bytes; old records are never overwritten.
        a.private_output.parent.mkdir(parents=True, exist_ok=True)
        with a.private_output.open('x') as f: f.write(test.read_text())
    public = {k: spec[k] for k in ['frozen_at', 'experiment_id', 'source_protocol', 'selected_languages',
        'pool_by_language', 'stage_updates', 'training_seed', 'stage_evaluation_protocol',
        'followup_data_freeze_sha256', 'pool_manifest_sha256', 'followup_guard_sha256', 'selection_validator_sha256']}
    public.update(private_selection_sha256=digest(a.private_output), math_report_sha256=digest(a.math_report),
        output_review_sha256=digest(a.output_review), forward_order=a.languages, reverse_order=list(reversed(a.languages)),
        scope='Real evidence-qualified research selection; training and seven-suite matrices still require separate completed runs. Stage budgets remain20 updates,100-step scheduler horizon,effectivebatch32,2048 training cap,default clip0.05 and frozen original teacher.')
    a.public_output.parent.mkdir(parents=True, exist_ok=True)
    with a.public_output.open('x') as f: json.dump(public, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(public, ensure_ascii=False))


if __name__ == '__main__': main()
