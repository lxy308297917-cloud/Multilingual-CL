"""Index completed primary and sequential model files without loading models.

Checks saved state presence and bound adapter hashes. This is not a new resume
or optimizer deserialization test. Physical storage paths remain local.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

LANGUAGES = ['sw', 'te', 'th', 'bn', 'zh', 'ja', 'ru', 'de', 'fr', 'es']
REQUIRED = ['adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json',
    'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin',
    'tokenizer.json', 'tokenizer_config.json']


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def inventory(root):
    code = Path(__file__).resolve().parent
    evidence, specifications = {}, {}
    for language in LANGUAGES:
        path = code / f'primary4b_{language.upper()}20_formal_frozen_20261024_v1.LOCAL.json'
        value = read(path)
        evidence[str(path.relative_to(root))] = digest(path)
        cases = list(value['cases'].values())
        assert len({case['adapter_relative'] for case in cases}) == 1
        specifications['primary_' + language] = {'path': cases[0]['adapter_relative'],
            'expected_adapter_sha256': cases[0]['adapter_sha256'], 'cohort': 'original2700_primary'}
    path = code / 'SELECTED4B_COMPLETE_SEQUENTIAL_RESULTS_20261026.PUBLIC.json'
    selected = read(path)
    evidence[str(path.relative_to(root))] = digest(path)
    assert selected['all_56_evaluations_accepted']
    for name, value in selected['unique_training_records'].items():
        specifications[name] = {'path': value['checkpoint_relative'],
            'expected_adapter_sha256': value['adapter_sha256'],
            'cohort': 'selected900_matched_or_sequential', 'stage_context': value['stage_context']}
    assert len(specifications) == 17
    public, physical, seen = {}, {}, set()
    for name, spec in specifications.items():
        folder = root / spec['path']
        resolved = folder.resolve(strict=True)
        assert resolved not in seen
        seen.add(resolved)
        state = read(folder / 'trainer_state.json')
        assert state['global_step'] == 20
        adapter = read(folder / 'adapter_config.json')
        assert adapter['r'] == 64 and adapter['lora_alpha'] == 128
        assert adapter['lora_dropout'] == .05
        actual = digest(folder / 'adapter_model.safetensors')
        assert actual == spec['expected_adapter_sha256'], name
        files = {}
        for filename in REQUIRED:
            path = folder / filename
            assert path.is_file() and path.stat().st_size > 0, (name, filename)
            files[filename] = {'bytes': path.stat().st_size}
            if filename != 'optimizer.pt':
                files[filename]['sha256'] = digest(path) if filename != 'adapter_model.safetensors' else actual
        public[name] = {**spec, 'global_step': 20, 'files': files,
            'saved_resume_state_files_present': True,
            'optimizer_deserialization_or_fresh_resume_test_performed': False}
        physical[name] = str(resolved)
    return ({'observed_at': datetime.datetime.now().astimezone().isoformat(),
        'models': public, 'unique_checkpoint_directories': 17,
        'source_evidence_sha256': evidence, 'collector_sha256': digest(Path(__file__)),
        'scope': 'Ten original2700 primary and seven selected900/sequential completed checkpoint20 directories. Stage1 reuse indexed once. Required files are present/nonempty; all adapter identities match formal releases. Optimizer files are not rehashed or deserialized, and no new resume is attempted. Paths are logical unified checkpoint entries; actual resolved storage paths stay in local sidecar. Does not imply model efficacy or completion of pending evaluations.',
        'GPU_work_started_or_changed': False, 'full_goal_complete': False}, physical)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private-storage-output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.private_storage_output.exists()
    report, physical = inventory(args.root.resolve())
    for path, value in [(args.output, report), (args.private_storage_output, physical)]:
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({'checkpoints': 17, 'bound_adapters_match': True,
        'required_resume_files_present': True, 'new_resume_test': False}))


if __name__ == '__main__':
    main()
