"""Evidence/data checks for research selection after the fixed followup200.

These checks are performed by the assistant; they add no user approval step.
No models, queues, training or final language records are created here.
"""
import hashlib
import json
from pathlib import Path

PROTOCOL = 'project_followup_dev200_single8192_c64_v1'


def read(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_pool_binding(selected, code):
    code = Path(code)
    freeze_path = code / 'followup200_frozen_20261021.PUBLIC.json'
    frozen = read(freeze_path)
    assert selected['followup_data_freeze_sha256'] == digest(freeze_path)
    path = Path(selected['pool_manifest'])
    expected = read(code / 'sequential_pools_followup200_20261021.PUBLIC.json')['private_manifest_sha256']
    assert selected['pool_manifest_sha256'] == digest(path) == expected, 'Pool must be the frozen followup-safe version'
    pools = read(path)
    assert pools['followup_data_freeze_sha256'] == selected['followup_data_freeze_sha256']
    assert pools['followup_exclusion_sha256'] == frozen['future_training_exclusion']['sha256']
    dev = {str(i) for i in pools['development_ids']}
    assert len(dev) >= 300 and set(frozen['selected_IDs']) <= dev
    seen = set()
    for pool in pools['pool_ids']:
        ids = {str(i) for i in pool}
        assert len(pool) == len(ids) >= 640 and not ids & dev and not ids & seen
        seen.update(ids)
    assert len(pools['pool_ids']) == 3 and len(seen) == 2700
    return pools


def validate_math_evidence(selected, report, code):
    assert report.get('protocol') == PROTOCOL, 'Formal tests and teacher diagnostics cannot select research languages'
    assert report.get('all_requested_parents_completed_exit0') is True, 'Completed paired candidates required, not Base-only output'
    assert report.get('confirmation_stage') in ['primary_default_only', 'all_cases']
    assert report['data_freeze_sha256'] == selected['followup_data_freeze_sha256']
    assert report['analysis_source_sha256'] == digest(Path(code) / 'collect_followup200.py')
    chosen = selected['selected_languages']
    assert set(chosen) <= set(report['math_supported_default_candidates'])
    for lang in chosen:
        cell = report['paired_confirmation'][lang]
        assert cell['group'] == 'default_component_clip0.05'
        assert cell['both_target_deltas_positive'] and cell['prespecified_math_confirmation_passes']
        assert cell['target_fixed_two_model_mean']['CI_lower_strictly_positive']
        assert cell['target_fixed_two_model_mean']['paired_question_bootstrap95_CI_percentage_points'][0] > 0


def validate_output_review(selected, paired_reports):
    evidence = selected['output_review_report']
    path = Path(evidence['path'])
    assert digest(path) == evidence['sha256']
    review = read(path)
    assert review['protocol'] == 'project_followup200_output_review_v1'
    assert review['data_freeze_sha256'] == selected['followup_data_freeze_sha256']
    assert review['language_and_termination_reviewed'] is True and review['English_tradeoff_recorded'] is True
    assert review['math_report_sha256'] in paired_reports
    math_report = paired_reports[review['math_report_sha256']]
    assert set(selected['selected_languages']) <= set(review['languages_reviewed'])
    for lang in selected['selected_languages']:
        assessment = review['by_language'][lang]
        assert assessment['eligible_for_sequential'] is True
        assert isinstance(assessment['rationale'], str) and assessment['rationale'].strip()
        for case in ['base', lang + '42', lang + '43']:
            for target in [lang, 'en']:
                assert review['source_records'][case][target] == math_report['source_records'][case][target]
