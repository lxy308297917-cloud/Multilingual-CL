"""Engineering-only 4B adapter handoff on the existing audited training pool."""
from pathlib import Path

from single4b_compact_entry import prepare as prepare_single, read, sha


def prepare(root, freeze):
    f, rows, data, output = prepare_single(root, freeze)
    assert f['engineering_only'] is True
    assert f['training_role'] == 'sequential4b_engineering_handoff'
    assert f['stop_after_updates'] == 2 and f['parent_checkpoint_step'] == 20
    gate_path = root / f['adapter_handoff_CPU_report_relative']
    assert sha(gate_path) == f['adapter_handoff_CPU_report_sha256']
    gate = read(gate_path)
    assert gate['actual_adapter_tensors_loaded'] == 504
    assert gate['all_loaded_adapter_values_exact'] and gate['original_Base_bitwise_unchanged']
    parent = root / f['parent_checkpoint_relative']
    release_path = root / f['parent_release_relative']
    assert sha(release_path) == f['parent_release_sha256']
    release = read(release_path)
    assert release['global_step'] == 20
    assert release['adapter_sha256'] == f['parent_adapter_sha256']
    assert sha(parent / 'adapter_model.safetensors') == f['parent_adapter_sha256']
    assert read(parent / 'trainer_state.json')['global_step'] == 20
    assert Path(read(parent / 'adapter_config.json')['base_model_name_or_path']).resolve() == (root / 'models/Qwen3-4B').resolve()
    previous = read(parent.parent / 'SERVER_PILOT20_UPDATES_DONE.json')
    assert previous['global_step'] == 20 and previous['base_unchanged_bitwise']
    assert read(parent.parent / 'SERVER_PARENT_DONE.json')['two_role_job_completed']
    assert f['model_revision'] == '1cfa9a7208912126459214e8b04321603b3df60c'
    assert f['target_language'] == 'RU', 'This engineering fixture reuses accepted RU20 only'
    assert previous['micro_steps'] == 640 and len(previous['syncs']) == 21
    assert f['train_arguments'][f['train_arguments'].index('--max_steps') + 1] == '100'
    return f, rows, data, output
