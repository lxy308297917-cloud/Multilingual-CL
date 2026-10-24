"""Research-stage preparation for selected 4B languages and matched controls."""
from pathlib import Path
import re
import shutil

from prepare_sequential_pools import read, sha
from sequential4b_data import pool_rows
from sequential4b_selection import local, selection, stage_context, REVISION


def prepare(root, freeze):
    root = Path(root)
    code = Path(__file__).parent
    f = read(freeze)
    assert f['engineering_only'] is False
    assert f['model_revision'] == REVISION
    required_code = {'sequential4b_context.py', 'sequential4b_stage_entry.py',
                     'sequential4b_stage_job.py', 'run_sequential4b_stage.sh',
                     'sequential4b_selection.py', 'sequential4b_data.py',
                     'sequential_adapter.py', 'sequential_stage_entry.py',
                     'server_rollout_compat.py', 'compact_response_logits.py',
                     'lora_sync_weights.py', 'cost_instrumentation.py',
                     'mask_reduction_memory.py', 'run_trl_audited_server32.py'}
    assert required_code <= set(f['code_sha256'])
    for name, digest in f['code_sha256'].items():
        assert sha(code / name) == digest, name
    for name, digest in f['installed_source_sha256'].items():
        assert sha(root / name) == digest, name
    selected = selection(root, local(root, f['selection_record_relative']),
                         f['selection_record_sha256'])
    context = stage_context(selected, f['direction'], f['stage_index'],
                            f['target_language'].lower(), f['training_role'])
    assert context == f['stage_context']
    assert f['target_language'] == context['language'].upper()
    assert f['updates'] == 20 and f['scheduler_horizon'] == 100
    assert f['training_seed'] == f['data_seed'] == context['training_seed']
    # Use the accepted cohort's entire argument list; only language and run
    # grouping change. This prevents an unnoticed loss/optimizer/budget fork.
    reference = read(code / 'ru4b_single20_frozen_20261024.PUBLIC.json')
    expected_args = list(reference['train_arguments'])
    expected_args[expected_args.index('--train_language') + 1] = f['target_language']
    expected_args[expected_args.index('--wandb_project') + 1] = f['wandb_project']
    assert f['train_arguments'] == expected_args
    pools = local(root, selected['pool_manifest_relative'])
    rows = pool_rows(root, pools, selected['pool_manifest_sha256'],
                     context['language'], context['pool_index'])
    data = local(root, f['training_data_relative'])
    assert sha(data) == f['training_data_sha256'] and read(data) == rows
    binding = read(data.with_name(data.name + '.BINDING.PRIVATE.json'))
    assert binding['context'] == context
    assert binding['training_data_sha256'] == sha(data)
    assert binding['selection_record_sha256'] == f['selection_record_sha256']
    lock = local(root, f['model_lock_relative'])
    assert sha(lock) == f['model_lock_sha256']
    model = read(lock)
    assert model['revision'] == REVISION
    for name, value in model['files'].items():
        assert sha(root / model['model_relative'] / name) == value['sha256']
    core = read(root / 'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json')
    for name, digest in core['core_source_sha256'].items():
        assert sha(root / 'COPSD-ddp-teacher-unwrapped-20261018' / name) == digest
    assert read(code / 'compact_response_logits_CPU_20261024.PUBLIC.json')['all_passed']
    gate = read(code / 'te4b_actual_Trainer_OOM_case_passed_20261024.PUBLIC.json')
    assert gate['actual_Accelerate_Trainer_backward_and_CUDA_sync_passed']
    assert gate['repair_patch_sha256'] == sha(code / 'compact_response_logits.py')
    assert re.fullmatch(r'[A-Za-z0-9_-]+', f['run_id'])
    output = local(root, f['checkpoint_relative'])
    assert output.is_relative_to(root / 'models/checkpoint')
    assert output.name == f['run_id'] and not output.exists()
    container = output.parent.resolve()
    assert container == Path(f['checkpoint_actual_container'])
    assert container.is_relative_to((root / 'models/checkpoint').resolve()) or container.is_relative_to(Path('/data1') / root.name)
    assert shutil.disk_usage(root).free >= 2 * 2**30
    assert shutil.disk_usage(container).free >= f['disk_required_bytes'] >= 4 * 2**30
    parent_relative = f.get('parent_checkpoint_relative')
    needs_parent = context['experiment_role'] == 'sequential_stage' and context['stage_index'] > 1
    assert bool(parent_relative) == needs_parent
    if needs_parent:
        parent = local(root, parent_relative)
        assert parent.is_relative_to(root / 'models/checkpoint')
        assert parent.resolve().is_relative_to((root / 'models/checkpoint').resolve()) or parent.resolve().is_relative_to(Path('/data1') / root.name)
        assert sha(parent / 'adapter_model.safetensors') == f['parent_adapter_sha256']
        assert read(parent / 'trainer_state.json')['global_step'] == 20
        previous = read(parent.parent / 'SEQUENTIAL4B_STAGE_CONTEXT.PRIVATE.json')
        wanted = stage_context(selected, f['direction'], f['stage_index'] - 1)
        # Stage one may be an explicitly reused, exactly matched single.
        if previous['context']['experiment_role'] == 'matched_single_reference':
            assert f['stage_index'] == 2
            assert selected['first_stage_reuse_policy'] == 'reuse_exact_matched_single'
            wanted = stage_context(selected, 'matched', 1, wanted['language'], 'matched_single_reference')
        assert previous['context'] == wanted
        assert previous['selection_record_sha256'] == f['selection_record_sha256']
        done = read(parent.parent / 'SEQUENTIAL4B_RESEARCH_UPDATES_DONE.json')
        assert done['global_step'] == 20 and done['base_unchanged_bitwise']
        assert done['stage_context'] == wanted and done['engineering_only'] is False
        assert read(parent.parent / 'SERVER_PARENT_DONE.json')['two_role_job_completed']
    else:
        assert f.get('parent_adapter_sha256') is None
    return f, rows, data, output
