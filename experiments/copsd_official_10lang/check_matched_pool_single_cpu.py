"""Read-only real-pool validation, with temporary corrupt-input rejection cases."""
import argparse
import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from matched_pool_single_entry import load_pool
from prepare_sequential_pools import materialize
from sequential_evaluation import digest, evaluation_identity, read, required_cases, write_new


def rejects(function):
    try:
        function()
    except AssertionError:
        return True
    raise AssertionError('Invalid matched-pool evidence accepted')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    code = Path(__file__).parent
    # Synthetic identities only; this does not create a development selection.
    identity_selection = {'experiment_id': 'interface_only', 'selected_languages': ['sw', 'ja', 'zh']}
    for language in identity_selection['selected_languages']:
        plan = {'experiment_role': 'matched_single_reference', 'reference_language': language,
            'direction': 'matched', 'stage_index': 1, 'order_id': 'interface_only_matched_' + language}
        assert evaluation_identity(identity_selection, plan) == ('matched_single_reference', language)
    assert len(required_cases(identity_selection['selected_languages'])) == 7
    bad = dict(plan, direction='forward')
    rejected_role = rejects(lambda: evaluation_identity(identity_selection, bad))
    sequential = {'direction': 'reverse', 'stage_index': 2, 'order_id': 'interface_only_reverse'}
    assert evaluation_identity(identity_selection, sequential) == ('sequential_stage', 'ja')
    rejected_relabel = rejects(lambda: evaluation_identity(identity_selection,
        dict(sequential, experiment_role='matched_single_reference', reference_language='ja')))
    folder = a.root / 'data/copsd_official_20261015/sequential_pools_20261019_v1'
    manifest = folder / 'POOLS.PRIVATE.json'
    sha = digest(manifest)
    sw_data = folder / 'sw_pool0_interface_check.json'
    sw_ids = load_pool(a.root, code, 'sw', 0, manifest, sha, sw_data, digest(sw_data), 20)
    repair = a.root / 'data/copsd_official_20261015/adapted_v3_ja_prefix_repair'
    with tempfile.TemporaryDirectory(prefix='copsd-matched-pool-') as temp:
        temp = Path(temp)
        ja_data = temp / 'ja_pool1.json'
        materialize(SimpleNamespace(manifest=manifest, language='ja', pool=1, output=ja_data,
            ja_repaired_input=repair / 'project_train_2900_ja.json',
            ja_repair_manifest=repair / 'project_train_2900_ja.PRIVATE.json'))
        ja_ids = load_pool(a.root, code, 'ja', 1, manifest, sha, ja_data, digest(ja_data), 20)
        assert len(sw_ids) == len(ja_ids) == 967 and not set(sw_ids) & set(ja_ids)
        wrong_pool = rejects(lambda: load_pool(a.root, code, 'ja', 0,
            manifest, sha, ja_data, digest(ja_data), 20))
        pools = read(manifest)
        original = {r['problem_id']: r for r in read(pools['inputs']['ja']['path'])}
        unrepaired = temp / 'ja_unrepaired.json'
        unrepaired.write_text(json.dumps([original[i] for i in ja_ids], ensure_ascii=False))
        bad_prefix = rejects(lambda: load_pool(a.root, code, 'ja', 1,
            manifest, sha, unrepaired, digest(unrepaired), 20))
        changed = read(ja_data)
        changed[0]['solution'] += '\nCORRUPTED_INTERFACE_TEST'
        wrong_reference = temp / 'changed_English_reference.json'
        wrong_reference.write_text(json.dumps(changed, ensure_ascii=False))
        changed_reference = rejects(lambda: load_pool(a.root, code, 'ja', 1,
            manifest, sha, wrong_reference, digest(wrong_reference), 20))
    write_new(a.output, {'observed_at': datetime.datetime.now().isoformat(),
        'real_pool_bindings_validated_for_interface_only': {'sw_pool0': len(sw_ids), 'ja_pool1_repaired': len(ja_ids)},
        'bound_pool_ID_sets_disjoint': True,
        'rejected': {'wrong_pool_index': wrong_pool, 'known_unrepaired_JA_prefix': bad_prefix,
                     'changed_English_teacher_reference': changed_reference},
        'entry_sha256': digest(code / 'matched_pool_single_entry.py'),
        'all_three_matched_language_identities_and_seven_suite_case_set_checked': True,
        'role_relabeling_rejected': rejected_role and rejected_relabel,
        'sequential_evaluator_sha256': digest(code / 'sequential_evaluation.py'),
        'preserved_GPU_validated_executor_sha256': digest(code / 'sequential_stage_entry.py'),
        'GPU_used': False, 'new_generation': False, 'production_selection_created': False,
        'scope': 'Real967-row pool checks and negative corrupt-input cases in temporary files only. SW0/JA1 are interface test bindings, not final selected languages. No full production config, new matched-single GPU run, seven-suite evaluation or effectiveness claim. Original datasets and source pools unchanged; temporary files removed.'})
    print(json.dumps({'actual_pool_rows': [len(sw_ids), len(ja_ids)], 'negative_cases_rejected': 3, 'GPU_used': False}))


if __name__ == '__main__':
    main()
