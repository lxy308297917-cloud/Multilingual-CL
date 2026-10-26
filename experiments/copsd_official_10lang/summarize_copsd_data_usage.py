"""Export IDs-only data provenance for primary singles and selected sequences.

Reads previously recorded prompt-fingerprint reconstructions; never synthesizes
answers, trains, changes splits or uses test scores to select data.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

LANGUAGES = ['sw', 'te', 'th', 'bn', 'zh', 'ja', 'ru', 'de', 'fr', 'es']


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root, primary_dir):
    code = Path(__file__).resolve().parent
    logs = root / 'logs/copsd_official_10lang'
    evidence = {}
    def bind(label, path):
        evidence[label] = digest(path)
    selected_path = code / 'sequential4b_selection_frozen_20261024_v1.LOCAL.json'
    selected = read(selected_path)
    pools_path = root / selected['pool_manifest_relative']
    assert digest(pools_path) == selected['pool_manifest_sha256']
    pools = read(pools_path)
    split_path = code / 'sw_lr20_frozen_20261022.PUBLIC.json'
    split = read(split_path)
    manifest_path = root / split['training_manifest_relative']
    assert digest(manifest_path) == split['training_manifest_sha256']
    manifest = read(manifest_path)
    original_ids = manifest['training_IDs']
    assert len(original_ids) == len(set(original_ids)) == 2700
    assert set(original_ids) == set().union(*map(set, pools['pool_ids']))
    index = {identifier: i for i, identifier in enumerate(original_ids)}
    followup_path = code / 'followup200_frozen_20261021.PUBLIC.json'
    followup = read(followup_path)
    development_ids = followup['selected_IDs']
    original100 = sorted(set(pools['development_ids']) - set(development_ids))
    assert len(development_ids) == 200 and len(set(development_ids)) == 200 and len(original100) == 100
    assert not set(original_ids) & set(pools['development_ids'])
    bind('selection', selected_path)
    bind('three_pool_manifest', pools_path)
    bind('primary2700_split_freeze', split_path)
    bind('primary2700_manifest', manifest_path)
    bind('independent200_ID_freeze', followup_path)
    primary_summary_path = primary_dir / 'COMPLETED.PRIVATE.json'
    primary_summary = read(primary_summary_path)
    assert primary_summary['all_ten_actual_ID_orders_equal']
    assert primary_summary['normalized_English_unique_training_pool'] == 2700
    assert primary_summary['normalized_English_pool_vs_reserved_overlap'] == 0
    bind('primary_actual_exposure_summary', primary_summary_path)
    primary, common_ids = {}, None
    for language in LANGUAGES:
        path = primary_dir / (language + '.PRIVATE.json')
        assert digest(path) == primary_summary['reports_sha256'][language]
        value = read(path)
        assert value['student_and_teacher_prompt_pairs_all_exact_and_unique']
        assert value['exposures'] == value['unique_original_ids'] == 640
        for filename, expected in value['source_sha256'].items():
            assert digest(root / filename) == expected
        ids = value['actual_exposure_original_ids_in_order']
        if common_ids is None:
            common_ids = ids
        assert ids == common_ids and set(ids) <= set(original_ids)
        primary[language] = {
            'model_id': value['model_id'], 'adapter_sha256': value['adapter_sha256'],
            'accepted_exposures': 640, 'actual_input_fingerprints_matched_uniquely': True,
            'retained_segments': value['retained_segments'], 'existing_JA_repair': value['existing_JA_repair'],
            'receipt_sha256': digest(path), 'reconstruction_source_sha256': value['reconstruction_source_sha256'],
        }
    mapping = {
        'SWmatched': 'matched_sw20', 'TEmatched': 'matched_te20', 'ZHmatched': 'matched_zh20',
        'forward_s2': 'forward_s2', 'reverse_s2': 'reverse_s2',
        'forward_s3': 'forward_s3', 'reverse_s3': 'reverse_s3',
    }
    actual_selected, by_language = {}, {}
    for model, label in mapping.items():
        marker_path = logs / f'selected4b_{model}_FORMAL_COMPLETED_20261024.PRIVATE.json'
        marker = read(marker_path)
        release_path = root / marker['model_provenance']['release_relative']
        assert digest(release_path) == marker['model_provenance']['release_sha256']
        release = read(release_path)
        freeze_path = root / release['freeze_relative']
        assert digest(freeze_path) == release['freeze_sha256']
        freeze = read(freeze_path)
        exposure_path = logs / f'selected4b_{label}_EXPOSURES_20261024.PRIVATE.json'
        exposure = read(exposure_path)
        assert exposure['release_sha256'] == digest(release_path)
        assert exposure['freeze_sha256'] == digest(freeze_path)
        assert exposure['student_and_teacher_prompt_pairs_all_exact']
        assert exposure['exposures'] == exposure['unique_original_ids'] == 640
        assert exposure['development_overlap'] == 0
        assert digest(code / 'recover_sequential4b_exposure_ids.py') == exposure['source_sha256']
        assert digest(root / freeze['checkpoint_relative'] / 'EXPOSURE_PROMPTS.PRIVATE.json') == exposure['recorded_prompt_sha256']
        assert digest(root / freeze['training_data_relative']) == freeze['training_data_sha256'] == exposure['training_data_sha256']
        ids = exposure['actual_exposure_original_ids_in_order']
        language = freeze['target_language'].lower()
        assert language == freeze['stage_context']['language']
        assert set(ids) <= set(pools['pool_ids'][freeze['stage_context']['pool_index']])
        if language in by_language:
            assert ids == by_language[language]
        else:
            by_language[language] = ids
        actual_selected[model] = {
            'language': language, 'pool_index': freeze['stage_context']['pool_index'],
            'actual_exposures': 640, 'actual_ID_order_equals_same_language_single': True,
            'exposure_manifest_sha256': digest(exposure_path), 'actual_input_prompt_sha256': exposure['recorded_prompt_sha256'],
            'release_sha256': digest(release_path), 'training_data_sha256': exposure['training_data_sha256'],
        }
        bind(model + ':accepted_model', marker_path)
        bind(model + ':actual_exposures', exposure_path)
    assert set(by_language) == {'sw', 'te', 'zh'}
    assert len(set().union(*map(set, by_language.values()))) == 1920
    assert not set().union(*map(set, by_language.values())) & set(pools['development_ids'])
    for direction in ['forward', 'reverse']:
        for stage in [1, 2, 3]:
            path = logs / f'selected4b_{direction}_s{stage}_MATCHED_COMPARISON_20261024.PRIVATE.json'
            control = read(path)
            assert control['actual_exposure_ID_order_exactly_equal'] and control['current_stage_exposures_each'] == 640
            bind(direction + ':matched_' + str(stage), path)
    audit_path = root / 'data/copsd_official_20261015/adapted_v2_ordered/DATA_AUDIT.json'
    audit = read(audit_path)
    assert audit['revision'] == '519ea2be68fec00c8a033cb2040059ff595b06a0'
    assert all(value['count'] == 0 for value in audit['exact_english_benchmark_overlap'].values())
    bind('official_data_adaptation_audit', audit_path)
    return {
        'observed_at': datetime.datetime.now().astimezone().isoformat(),
        'official_dataset': 'yihongLiu/COPSD-PolyMath-TrainDataset',
        'official_dataset_revision': audit['revision'],
        'ID_definition': 'SHA256 of original English problem.strip() UTF-8, as in the existing dataset adapter; no question/solution/answer texts exported.',
        'primary2700_original_IDs_in_input_order': original_ids,
        'index_encoding': 'All index arrays below are zero-based positions in primary2700_original_IDs_in_input_order; array order is preserved.',
        'reserved_original100_IDs': original100, 'independent_development200_IDs': development_ids,
        'primary_actual_common640_ID_indices_in_order': [index[value] for value in common_ids],
        'primary_ten_languages': primary,
        'three_ordered900_pools_ID_indices': [[index[value] for value in pool] for pool in pools['pool_ids']],
        'selected_language_pool_binding': selected['pool_by_language'],
        'selected_actual640_ID_indices_by_language': {lang: [index[value] for value in ids] for lang, ids in by_language.items()},
        'seven_selected_training_segments': actual_selected,
        'data_audit_summary': {
            'primary_pool_rows': 2700, 'primary_accepted_exposures_per_model': 640,
            'all_ten_primary_actual_ID_orders_equal': True,
            'normalized_English_unique_training_questions': 2700,
            'normalization': 'Unicode NFKC plus whitespace normalization; case preserved.',
            'train_vs_reserved_original100_and_dev200_ID_and_normalized_English_overlap': 0,
            'three_pool_sizes': [900, 900, 900], 'pool_pairwise_ID_and_normalized_English_overlap': 0,
            'actual_question_IDs_per_sequential_order': 1920,
            'stratification': pools['report']['stratification'],
            'stratification_limit': pools['report']['difficulty_limit'],
            'recorded_prior_exact_English_benchmark_overlap': audit['exact_english_benchmark_overlap'],
            'benchmark_manifest_sha256_in_prior_audit': audit['benchmark_manifest_sha256'],
            'English_and_reference_fields_shared_across_languages': True,
            'JA': 'Existing single-row literal-prefix repair remains a separately recorded project adaptation; no new edits.',
            'TE_recovery': 'Accepted640 = retained checkpoint5 first160 + resumed480. Original failed attempt logged256 prompts;96 beyond retained checkpoint5 are discarded from accepted-training exposure budget, not erased or zero-charged computation.',
        },
        'source_evidence_sha256': evidence, 'collector_sha256': digest(Path(__file__)),
        'no_model_forward_generation_or_dataset_modification': True, 'full_goal_complete': False,
        'scope': 'IDs-only data delivery. Primary2700 single models and matched900/sequential models remain separate cohorts. Teacher sees privileged English problem/reference, student target question; no target-language SFT labels are required. These provenance checks do not establish semantic translation/reference correctness, absence of all near-duplicates, general model efficacy or completion of pending formal tests. Original exact-English benchmark overlap audit is preserved; it is not relabeled an exhaustive semantic or new normalized-test audit.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--primary-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    value = summarize(args.root.resolve(), args.primary_dir)
    with args.output.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'primary_languages': 10, 'common_actual_questions': 640,
                      'selected_unique_actual_questions_per_order': 1920,
                      'reserved_original100_plus_dev200': 300, 'full_goal_complete': False}))


if __name__ == '__main__':
    main()
