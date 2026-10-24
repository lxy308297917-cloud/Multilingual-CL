"""Map recorded student/teacher prompt fingerprints to actual original IDs.

CPU-only reconstruction uses the frozen official collator and tokenizer.
This recovers data provenance, not model weights or a failed training job.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from prepare_sequential_pools import read, sha


def recover(root, release_path, output):
    root, release_path, output = Path(root), Path(release_path), Path(output)
    assert not output.exists()
    release = read(release_path)
    assert release['global_step'] == 20 and release['engineering_only'] is False
    freeze = root / release['freeze_relative']
    assert sha(freeze) == release['freeze_sha256']
    f = read(freeze)
    assert f['micro_batch'] == 1 and f['effective_batch'] == 32 and f['updates'] == 20
    assert f['train_arguments'][f['train_arguments'].index('--max_length') + 1] == '20000'
    for flag in ['--reason_first', '--no_include_problem_en', '--no_include_reference_solution_en']:
        assert flag not in f['train_arguments']
    assert '--student_enable_thinking' in f['train_arguments']
    source = root / 'COPSD-ddp-teacher-unwrapped-20261018'
    manifest = read(root / 'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json')
    for name in ['multilingual_data_collator.py', 'language_config.py', 'multilingual_opsd_train.py']:
        assert sha(source / name) == manifest['core_source_sha256'][name]
    data = root / f['training_data_relative']
    assert sha(data) == f['training_data_sha256']
    rows = read(data)
    assert len(rows) == 900 and len({row['problem_id'] for row in rows}) == 900
    sys.path.insert(0, str(source))
    from multilingual_data_collator import MultilingualSelfDistillationDataCollator
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / 'models/Qwen3-4B', padding_side='left', local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = MultilingualSelfDistillationDataCollator(tokenizer, max_length=20000,
        reason_first=False, student_enable_thinking=True,
        include_problem_en=True, include_reference_solution_en=True)
    def key(batch):
        return tuple(hashlib.sha256(batch[name].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                     for name in ['student_prompts', 'teacher_prompts'])
    lookup = {}
    for row in rows:
        batch = collator([{**row, 'target_lang': f['target_language'], 'problem_en': row['problem']}])
        lookup.setdefault(key(batch), []).append(row['problem_id'])
    folder = root / f['checkpoint_relative']
    recorded_path = folder / 'EXPOSURE_PROMPTS.PRIVATE.json'
    recorded = read(recorded_path)
    assert len(recorded) == 640
    ids = []
    for record in recorded:
        matches = lookup[(record['student_prompts_sha256'], record['teacher_prompts_sha256'])]
        assert len(matches) == 1, 'Ambiguous prompt-to-ID mapping; report duplicates rather than invent IDs'
        ids.append(matches[0])
    assert len(set(ids)) == 640
    pools = read(root / read(root / f['selection_record_relative'])['pool_manifest_relative'])
    assert set(ids) <= set(pools['pool_ids'][f['stage_context']['pool_index']])
    assert not set(ids) & set(pools['development_ids'])
    result = {'run_id': f['run_id'], 'stage_context': f['stage_context'],
              'freeze_sha256': sha(freeze), 'release_sha256': sha(release_path),
              'recorded_prompt_sha256': sha(recorded_path), 'training_data_sha256': sha(data),
              'actual_exposure_original_ids_in_order': ids, 'exposures': 640,
              'unique_original_ids': 640, 'development_overlap': 0,
              'student_and_teacher_prompt_pairs_all_exact': True,
              'source_sha256': sha(Path(__file__)), 'GPU_started': False,
              'scope': 'Exact CPU mapping of recorded per-microbatch student/privileged-teacher token fingerprints to immutable900-row training inputs. Does not infer semantic correctness of references or mathematical efficacy; no training or model generation.'}
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.root, args.release, args.output)
    print(json.dumps({key: result[key] for key in ['run_id', 'exposures', 'unique_original_ids', 'development_overlap', 'student_and_teacher_prompt_pairs_all_exact', 'GPU_started']}))


if __name__ == '__main__':
    main()
