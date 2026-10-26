"""Recover actual primary4B exposure IDs from saved input-token fingerprints.

CPU-only provenance reconstruction. No model forward, generation or dataset
rewrite. TE joins the retained checkpoint5 prefix with its actual resumed tail.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

LANGUAGES = ['sw', 'te', 'th', 'bn', 'zh', 'ja', 'ru', 'de', 'fr', 'es']


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--languages', nargs='+', choices=LANGUAGES, default=LANGUAGES)
    args = parser.parse_args()
    root, code = args.root.resolve(), Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = code / 'sequential4b_selection_frozen_20261024_v1.LOCAL.json'
    selection = read(selection_path)
    pool_path = root / selection['pool_manifest_relative']
    assert digest(pool_path) == selection['pool_manifest_sha256']
    pools = read(pool_path)
    pool_ids = set().union(*map(set, pools['pool_ids']))
    reserved = set(pools['development_ids'])
    assert len(pool_ids) == 2700 and len(reserved) == 300 and not pool_ids & reserved
    reference_path = Path(pools['inputs']['sw']['path'])
    assert digest(reference_path) == pools['inputs']['sw']['sha256']
    reference = {row['problem_id']: row for row in read(reference_path)}
    english_keys = ['source', 'problem', 'solution', 'Answer']
    normalized_pool = {normalized(reference[key]['problem']) for key in pool_ids}
    assert len(normalized_pool) == 2700
    old_dev_path = root / 'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json'
    old_development = read(old_dev_path)
    heldout = [row for key, row in reference.items() if key in reserved] + old_development
    assert len(heldout) == 300 and not normalized_pool & {normalized(row['problem']) for row in heldout}
    runtime = root / 'COPSD-ddp-teacher-unwrapped-20261018'
    core_manifest = read(root / 'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json')
    for name in ['multilingual_data_collator.py', 'language_config.py']:
        assert digest(runtime / name) == core_manifest['core_source_sha256'][name]
    sys.path.insert(0, str(runtime))
    import torch
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    from multilingual_data_collator import MultilingualSelfDistillationDataCollator
    tokenizer = AutoTokenizer.from_pretrained(root / 'models/Qwen3-4B', padding_side='left', local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = MultilingualSelfDistillationDataCollator(tokenizer, max_length=20000,
        reason_first=False, student_enable_thinking=True,
        include_problem_en=True, include_reference_solution_en=True)
    results = {}
    for language in args.languages:
        output = args.output_dir / (language + '.PRIVATE.json')
        if output.exists():
            prior = read(output)
            for name, expected in prior['source_sha256'].items():
                assert digest(root / name) == expected, name
            assert prior['exposures'] == prior['unique_original_ids'] == 640
            results[language] = prior
            print(json.dumps({'language': language, 'reused_existing_bound_reconstruction': True}), flush=True)
            continue
        formal_path = code / f'primary4b_{language.upper()}20_formal_frozen_20261024_v1.LOCAL.json'
        formal = read(formal_path)
        assert formal['model_revision'] == '1cfa9a7208912126459214e8b04321603b3df60c'
        spec = next(v for v in formal['cases'].values() if v['adapter_relative'])
        checkpoint = root / spec['adapter_relative']
        assert digest(checkpoint / 'adapter_model.safetensors') == spec['adapter_sha256']
        assert read(checkpoint / 'trainer_state.json')['global_step'] == 20
        folder = checkpoint.parent
        record_path = folder / 'EXPOSURE_PROMPTS.PRIVATE.json'
        recorded = read(record_path)
        sources = [selection_path, pool_path, reference_path, old_dev_path, formal_path,
                   checkpoint / 'adapter_model.safetensors', checkpoint / 'trainer_state.json',
                   record_path, runtime / 'multilingual_data_collator.py', runtime / 'language_config.py']
        segments = [{'role': 'completed_primary20', 'accepted_records': len(recorded)}]
        if language == 'te':
            resume_path = code / 'te4b_resume5_frozen_20261024.PUBLIC.json'
            resume = read(resume_path)
            assert resume['run_id'] == folder.name
            old_checkpoint = root / resume['resume_checkpoint_relative']
            assert read(old_checkpoint / 'trainer_state.json')['global_step'] == 5
            old_record_path = root / resume['old_exposure_prompts_relative']
            old_recorded = read(old_record_path)
            proof_path = folder / 'RESUME_DATA_ORDER_MATCH.json'
            load_path = folder / 'RESUME_OPTIMIZER_MODEL_LOADED.json'
            proof, loaded = read(proof_path), read(load_path)
            assert loaded['global_step'] == 5 and loaded['adapter_matches_saved_step5'] and loaded['scheduler_and_optimizer_step5_loaded']
            assert proof['next32_actual_prompt_tensor_hashes_match_original'] and proof['skipped_original_exposures'] == 160
            assert len(recorded) == 480 and len(old_recorded) == 256
            assert recorded[:32] == old_recorded[160:192]
            recorded = old_recorded[:160] + recorded
            segments = [{'role': 'retained_checkpoint5', 'accepted_records': 160,
                         'original_recorded_prompts': 256, 'discarded_after_retained_checkpoint': 96},
                        {'role': 'resumed_step6_to20', 'accepted_records': 480}]
            sources += [resume_path, old_checkpoint / 'trainer_state.json', old_record_path, proof_path, load_path]
        assert len(recorded) == 640
        input_path = Path(pools['inputs'][language]['path'])
        assert digest(input_path) == pools['inputs'][language]['sha256']
        rows = read(input_path)
        assert len(rows) == 2900 and len({row['problem_id'] for row in rows}) == 2900
        sources.append(input_path)
        repair = None
        if language == 'ja':
            repair_path = root / 'data/copsd_official_20261015/adapted_v3_ja_prefix_repair/project_train_2900_ja.json'
            repair_config = code / 'ja_repair_runtime_20261020.CONFIG.json'
            assert digest(repair_path) == read(repair_config)['data_sha256']
            fixed = read(repair_path)
            assert [row['problem_id'] for row in fixed] == [row['problem_id'] for row in rows]
            changed = [i for i, (a, b) in enumerate(zip(rows, fixed)) if a != b]
            assert len(changed) == 1
            index = changed[0]
            assert {k: v for k, v in rows[index].items() if k != 'problem_ja'} == {k: v for k, v in fixed[index].items() if k != 'problem_ja'}
            rows = fixed
            sources += [repair_path, repair_config]
            repair = 'Existing audited single-row literal Japanese prefix repair; no new modification.'
        lookup = {}
        for index, row in enumerate(rows, 1):
            assert all(row[key] == reference[row['problem_id']][key] for key in english_keys)
            batch = collator([{**row, 'target_lang': language, 'problem_en': row['problem']}])
            key = tuple(hashlib.sha256(batch[name].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        for name in ['student_prompts', 'teacher_prompts'])
            lookup.setdefault(key, []).append(row['problem_id'])
            if index % 1000 == 0:
                print(json.dumps({'language': language, 'CPU_candidate_rows': index}), flush=True)
        ids = []
        for record in recorded:
            matches = lookup.get((record['student_prompts_sha256'], record['teacher_prompts_sha256']), [])
            assert len(matches) == 1, (language, 'missing_or_ambiguous_prompt_match', len(matches))
            ids.append(matches[0])
        assert len(ids) == len(set(ids)) == 640
        assert set(ids) <= pool_ids and not set(ids) & reserved
        value = {'observed_at': datetime.datetime.now().astimezone().isoformat(),
                 'model_id': formal['model_id'], 'language': language,
                 'adapter_sha256': spec['adapter_sha256'],
                 'actual_exposure_original_ids_in_order': ids,
                 'exposures': 640, 'unique_original_ids': 640,
                 'reserved_original100_and_dev200_overlap': 0,
                 'student_and_teacher_prompt_pairs_all_exact_and_unique': True,
                 'retained_segments': segments, 'existing_JA_repair': repair,
                 'source_sha256': {str(p.relative_to(root)): digest(p) for p in sources},
                 'reconstruction_source_sha256': digest(Path(__file__)),
                 'GPU_initialized': torch.cuda.is_initialized(),
                 'scope': 'Actual accepted checkpoint20 training inputs, not a seed-based guess or inference from token counts. TE includes only retained checkpoint5 prefix and resumed15updates; discarded original-tail fingerprints remain separate overhead. No model forward, generation, loss update or data rewriting. IDs/hashes only; no prompt or answer text in this receipt.'}
        assert value['GPU_initialized'] is False
        with output.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        results[language] = value
        print(json.dumps({'language': language, 'exposures': 640, 'all_actual_prompt_pairs_matched': True}), flush=True)
    if set(results) == set(LANGUAGES):
        common = results['sw']['actual_exposure_original_ids_in_order']
        summary = {'observed_at': datetime.datetime.now().astimezone().isoformat(),
                   'languages': LANGUAGES, 'actual_accepted_exposures_each': 640,
                   'all_ten_actual_ID_orders_equal': all(v['actual_exposure_original_ids_in_order'] == common for v in results.values()),
                   'reports_sha256': {lang: digest(args.output_dir / (lang + '.PRIVATE.json')) for lang in LANGUAGES},
                   'normalized_English_unique_training_pool': 2700,
                   'normalized_English_pool_vs_reserved_overlap': 0,
                   'reserved_questions': 300, 'full_goal_complete': False,
                   'scope': 'Actual primary single data provenance only; original2700 cohort, not matched900 and not completion of formal evaluations.'}
        path = args.output_dir / 'COMPLETED.PRIVATE.json'
        assert not path.exists()
        with path.open('x') as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
