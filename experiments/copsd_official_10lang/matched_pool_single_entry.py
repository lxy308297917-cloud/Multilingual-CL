"""Fresh-Base single references for the exact language-bound sequential pools.

Reuses the GPU-validated official-core stage executor without changing its
trainer, optimizer, loss, teacher or rollout implementation. Only experiment
context/data validation and output namespace differ. No final-language
selection is created here; a frozen development-only selection is required.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

from sequential_evaluation import REVISION, digest, read, selection


def load_pool(root, code, language, index, manifest_path, manifest_sha, data_path, data_sha, updates):
    assert digest(manifest_path) == manifest_sha
    pools = read(manifest_path)
    assert type(index) is int and 0 <= index <= 2
    original = pools['inputs'][language]
    assert digest(original['path']) == original['sha256']
    assert digest(data_path) == data_sha
    rows = read(data_path)
    ids = pools['pool_ids'][index]
    assert [r['problem_id'] for r in rows] == ids
    assert len(ids) == len(set(ids)) and not set(ids) & set(pools['development_ids'])
    assert len(rows) >= 32 * updates
    reference = Path(original['path'])
    if language == 'ja':
        reference = root / 'data/copsd_official_20261015/adapted_v3_ja_prefix_repair/project_train_2900_ja.json'
        assert digest(reference) == read(code / 'ja_repair_runtime_20261020.CONFIG.json')['data_sha256']
    lookup = {r['problem_id']: r for r in read(reference)}
    assert rows == [lookup[identifier] for identifier in ids]
    return ids


def validate_assets(root, code):
    for name, sha in read(code / 'seeded_cost_runtime_20261020.CONFIG.json')['code_sha256'].items():
        assert digest(code / name) == sha, name
    compatibility = read(code / 'ja_nonefixed_runtime_20261020.CONFIG.json')
    for name, sha in compatibility['code_sha256'].items():
        assert digest(code / name) == sha, name
    binary = root / 'envs/copsd-vllm011-20261018/lib/python3.10/site-packages/vllm/cumem_allocator.abi3.so'
    assert digest(binary) == compatibility['binary_sha256']
    frozen = read(root / 'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json')
    for name, sha in frozen['core_source_sha256'].items():
        assert digest(root / 'COPSD-ddp-teacher-unwrapped-20261018' / name) == sha, name
    assert digest(frozen['model_lock']['path']) == frozen['model_lock']['sha256']


def prepare(root, config_path):
    code = Path(__file__).parent
    spec = read(config_path)
    assert shutil.disk_usage(root).free >= 20 * 2**30
    assert spec['entry_sha256'] == digest(Path(__file__))
    assert spec['wrapper_sha256'] == digest(code / 'run_matched_pool_single.sh')
    assert spec['executor_sha256'] == digest(code / 'sequential_stage_entry.py')
    assert spec['experiment_role'] == 'matched_single_reference'
    assert not spec.get('engineering_only', False) and not spec.get('parent_checkpoint')
    assert spec['model_revision'] == REVISION
    s = selection(spec['selection_record'], spec['selection_record_sha256'])
    language = spec['language']
    assert language in s['selected_languages']
    assert spec['pool_index'] == s['pool_by_language'][language]
    assert spec['updates'] == s['stage_updates'] == 20
    assert spec['seed'] == s['training_seed'] and type(spec['seed']) is int and spec['seed'] >= 0
    assert spec['order_id'] == s['experiment_id'] + '_matched_' + language
    assert spec['stage_index'] == 1
    assert re.fullmatch(r'[A-Za-z0-9_-]+', spec['run_id'])
    output = root / 'models/checkpoint/copsd_matched_singles_v1' / spec['run_id']
    assert not output.exists(), 'Use a new segment directory, preserving previous results'
    ids = load_pool(root, code, language, spec['pool_index'], spec['pool_manifest'],
        spec['pool_manifest_sha256'], spec['training_data'], spec['training_data_sha256'], spec['updates'])
    validate_assets(root, code)
    context = {key: spec[key] for key in ['order_id', 'stage_index', 'language', 'pool_index',
               'model_revision', 'updates', 'seed', 'training_data_sha256', 'pool_manifest_sha256']}
    context.update(engineering_only=False, experiment_role='matched_single_reference',
        selection_record_sha256=spec['selection_record_sha256'], data_rows=len(ids), parent_checkpoint=None)
    resume = Path(spec['resume_checkpoint']) if spec.get('resume_checkpoint') else None
    start = 0
    if resume:
        assert resume.resolve().is_relative_to((root / 'models/checkpoint/copsd_matched_singles_v1').resolve())
        assert read(resume.parent / 'STAGE_CONTEXT.PRIVATE.json')['context'] == context
        start = read(resume / 'trainer_state.json')['global_step']
        assert 0 < start < 20
        for name in ['adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
            assert (resume / name).stat().st_size > 0
    return spec, context, output, None, resume, start, ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if a.prepare_only:
        spec, _, _, _, _, start, ids = prepare(a.root, a.config)
        print(json.dumps({'CPU_preflight': True, 'GPU_started': False, 'language': spec['language'],
            'role': 'matched_single_reference', 'pool_index': spec['pool_index'],
            'rows': len(ids), 'start_step': start, 'stop_step': 20}))
        return
    # Process-local composition only; the preserved stage source is unchanged.
    import sequential_stage_entry as executor
    original_prepare = executor.prepare
    executor.prepare = prepare
    try:
        executor.execute(a.root, a.config)
    finally:
        executor.prepare = original_prepare


if __name__ == '__main__':
    main()
