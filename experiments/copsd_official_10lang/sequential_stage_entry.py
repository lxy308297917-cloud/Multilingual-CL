"""Official-core stage launcher: one adapter, frozen Base, fresh stage optimizer.

An explicitly engineering-only two-update handoff may reuse an existing single.
Research stages require a frozen independent-development selection record.
Within-stage recovery keeps its optimizer and uses a fresh output directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
LANGUAGES = 'sw te th bn zh ja ru de fr es'.split()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize_stage_model(trainer, checkpoint=None, expected_step=None):
    """Called after unique PEFT creation, before optimizer/scheduler creation."""
    from peft import PeftModel
    from sequential_adapter import frozen_fingerprint, load_stage_adapter
    assert trainer.optimizer is None and trainer.lr_scheduler is None
    model = trainer.model
    assert isinstance(model, PeftModel) and set(model.peft_config) == {'default'}
    assert model.active_adapter == 'default'
    assert all('lora_' in name for name, p in model.named_parameters() if p.requires_grad)
    if checkpoint:
        return load_stage_adapter(model, checkpoint, expected_step)
    return {'adapter_count': 1, 'fresh_original_Base': True,
            'frozen_Base_sha256_PRIVATE': frozen_fingerprint(model)}


def verify_optimization_start(trainer, state, expected_start):
    """Observe real Trainer-created optimization state, before any new update."""
    assert state.global_step == expected_start
    optimizer = trainer.optimizer.state_dict()['state']
    scheduler_step = trainer.lr_scheduler.state_dict()['last_epoch']
    assert scheduler_step == expected_start
    if expected_start == 0:
        assert not optimizer, 'A new stage must not inherit optimizer moments'
    else:
        assert optimizer and {int(v['step']) for v in optimizer.values()} == {expected_start}
    return {'global_step': expected_start, 'optimizer_state_empty': not optimizer,
            'scheduler_last_epoch': scheduler_step,
            'mode': 'stage_reset' if expected_start == 0 else 'within_stage_resume'}


def prepare(root, config_path):
    code = Path(__file__).parent
    spec = read(config_path)
    assert shutil.disk_usage(root).free >= 20 * 2**30, 'Storage reserve required'
    assert spec['entry_sha256'] == digest(Path(__file__))
    assert spec['wrapper_sha256'] == digest(code / 'run_sequential_stage.sh')
    run_id = spec['run_id']
    assert re.fullmatch(r'[A-Za-z0-9_-]+', run_id)
    output = root / 'models/checkpoint/copsd_sequential_stages_v1' / run_id
    assert not output.exists(), 'Use a fresh segment output; never overwrite weights'
    assert spec['model_revision'] == REVISION
    language = spec['language']
    assert language in LANGUAGES
    engineering = spec.get('engineering_only', False)
    selection_sha = None
    if engineering:
        assert spec['updates'] == 2 and spec['stage_index'] == 2
        assert language == 'sw' and spec.get('parent_checkpoint')
        assert spec['order_id'].startswith('engineering_')
    else:
        selection_path = Path(spec['selection_record'])
        selection_sha = digest(selection_path)
        assert selection_sha == spec['selection_record_sha256']
        from sequential_evaluation import selection as validated_selection
        selection = validated_selection(selection_path, selection_sha)
        chosen = selection['selected_languages']
        assert selection['source_protocol'] == 'independent_development_only'
        assert len(chosen) == len(set(chosen)) == 3 and set(chosen) <= set(LANGUAGES)
        assert selection['pool_by_language'] == {lang: i for i, lang in enumerate(chosen)}
        assert selection['stage_updates'] == spec['updates'] == 20
        assert selection['training_seed'] == spec['seed']
        assert selection['evidence_reports']
        for item in selection['evidence_reports']:
            assert digest(Path(item['path'])) == item['sha256']
        direction = spec['direction']
        assert direction in ['forward', 'reverse']
        order = chosen if direction == 'forward' else list(reversed(chosen))
        assert spec['order_id'] == selection['experiment_id'] + '_' + direction
        assert 1 <= spec['stage_index'] <= 3
        assert language == order[spec['stage_index'] - 1]
        assert spec['pool_index'] == selection['pool_by_language'][language]
    assert re.fullmatch(r'[A-Za-z0-9_-]+', spec['order_id'])
    assert isinstance(spec['seed'], int) and spec['seed'] >= 0
    pools = read(spec['pool_manifest'])
    assert digest(spec['pool_manifest']) == spec['pool_manifest_sha256']
    if not engineering:
        assert spec['pool_manifest_sha256'] == selection['pool_manifest_sha256']
        assert Path(spec['pool_manifest']).resolve() == Path(selection['pool_manifest']).resolve()
    item = pools['inputs'][language]
    assert digest(item['path']) == item['sha256']
    data = Path(spec['training_data'])
    assert digest(data) == spec['training_data_sha256']
    rows = read(data)
    ids = pools['pool_ids'][spec['pool_index']]
    assert [row['problem_id'] for row in rows] == ids
    assert len(set(ids)) == len(ids) and not set(ids) & set(pools['development_ids'])
    reference = Path(item['path'])
    if language == 'ja':
        reference = root / 'data/copsd_official_20261015/adapted_v3_ja_prefix_repair/project_train_2900_ja.json'
        assert digest(reference) == read(code / 'ja_repair_runtime_20261020.CONFIG.json')['data_sha256']
    lookup = {row['problem_id']: row for row in read(reference)}
    assert rows == [lookup[identifier] for identifier in ids]
    assert len(rows) >= 32 * spec['updates']
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
    context = {key: spec[key] for key in ['order_id', 'stage_index', 'language', 'pool_index',
               'model_revision', 'updates', 'seed', 'training_data_sha256', 'pool_manifest_sha256']}
    context.update(engineering_only=engineering, selection_record_sha256=selection_sha,
                   data_rows=len(rows), parent_checkpoint=spec.get('parent_checkpoint'))
    parent = Path(spec['parent_checkpoint']) if spec.get('parent_checkpoint') else None
    resume = Path(spec['resume_checkpoint']) if spec.get('resume_checkpoint') else None
    for checkpoint in [p for p in [parent, resume] if p]:
        assert checkpoint.resolve().is_relative_to((root / 'models/checkpoint').resolve())
        assert (checkpoint / 'adapter_model.safetensors').stat().st_size > 0
    if parent:
        previous = read(parent.parent / 'COLLOCATE_TRAIN_UPDATES_DONE.json')
        assert previous['global_step'] == read(parent / 'trainer_state.json')['global_step']
        assert previous['base_unchanged_bitwise']
        assert read(parent.parent / 'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
        if not engineering:
            prior = previous['sequential_context']
            assert prior['order_id'] == context['order_id']
            assert prior['stage_index'] == context['stage_index'] - 1
            assert prior['language'] == order[spec['stage_index'] - 2]
            assert prior['selection_record_sha256'] == selection_sha
            assert prior['model_revision'] == REVISION
            assert previous['global_step'] == spec['updates']
            evaluation_path = parent.parent / 'STAGE_EVALUATIONS_DONE.json'
            assert digest(evaluation_path) == spec['parent_evaluation_sha256']
            evaluation = read(evaluation_path)
            required = {(benchmark, lang) for lang in chosen for benchmark in ['mgsm_rev2', 'polymath']}
            required.add(('gsm8k', 'en'))
            assert evaluation['completed'] and evaluation['selection_record_sha256'] == selection_sha
            assert evaluation['protocol'] == selection['stage_evaluation_protocol']
            assert evaluation['checkpoint_adapter_sha256'] == digest(parent / 'adapter_model.safetensors')
            observed = [(x['benchmark'], x['language']) for x in evaluation['completed_benchmarks']]
            assert len(observed) == 7 and set(observed) == required
    else:
        assert not engineering and spec['stage_index'] == 1
    start = 0
    if resume:
        assert read(resume.parent / 'STAGE_CONTEXT.PRIVATE.json')['context'] == context
        start = read(resume / 'trainer_state.json')['global_step']
        assert 0 < start < spec['updates']
        for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
            assert (resume / name).stat().st_size > 0
    return spec, context, output, parent, resume, start, ids


def execute(root, config_path):
    spec, context, output, parent, resume, start, ids = prepare(root, config_path)
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    os.environ.pop('COPSD_EXPECT_INITIAL_SYNC_JSON', None)
    if parent and not resume:
        from build_stage_initial_reference import build
        expectation = root / 'logs/copsd_official_10lang/stage_references' / (spec['run_id'] + '.PRIVATE.json')
        build(parent, expectation, spec['order_id'], spec['stage_index'], REVISION, spec.get('engineering_only', False))
        os.environ['COPSD_EXPECT_INITIAL_SYNC_JSON'] = str(expectation)
    from cumem_none_ref_compat import install
    repair = install(read(Path(__file__).with_name('ja_nonefixed_runtime_20261020.CONFIG.json'))['binary_sha256'])
    import prompt_budget_guard
    prompt_budget_guard.install()
    import torch
    import seeded_cost_batch_entry as seeded
    runtime, probe = seeded.batch.runtime, seeded.batch.probe
    original_init = seeded.original_init

    def init(trainer, *args, **kwargs):
        original_init(trainer, *args, **kwargs)
        assert trainer.fixed_teacher and trainer.beta == 0 and trainer.jsd_token_clip == .05
        assert trainer.model.peft_config['default'].r == 64
        assert trainer.model.peft_config['default'].lora_alpha == 128
        assert list(trainer.train_dataset['problem_id']) == ids
        checkpoint = resume or parent
        expected = read(checkpoint / 'trainer_state.json')['global_step'] if checkpoint else None
        handoff = initialize_stage_model(trainer, checkpoint, expected)
        if resume or (parent and not spec.get('engineering_only', False)):
            previous_context = read(checkpoint.parent / 'STAGE_CONTEXT.PRIVATE.json')
            assert handoff['frozen_Base_sha256_PRIVATE'] == previous_context['initialization']['frozen_Base_sha256_PRIVATE']
        trainer._stage_loaded_hash = probe.fingerprint((n, p) for n, p in trainer.model.named_parameters() if p.requires_grad)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'STAGE_CONTEXT.PRIVATE.json').write_text(json.dumps({'context': context,
            'stage_spec_sha256': digest(config_path), 'initialization': handoff,
            'start_step': start, 'online_sampling_bitwise_recovery_promised': False}, indent=2) + '\n')
        (output / 'DATA_ORDER.PRIVATE.json').write_text(json.dumps({'original_ID_order': ids,
            'seed': spec['seed'], 'data_seed': spec['seed'], 'dataset_rows': len(ids),
            'recovery': 'HF checkpoint epoch/global_step and RNG restore data skipping; online engine stream is not promised bitwise identical.'}) + '\n')
        (output / 'ALLOCATOR_COMPAT_ACTUAL.json').write_text(json.dumps(repair, indent=2) + '\n')

    class StageCallback(runtime.ColocateCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            assert args.max_steps == 100 and args.seed == args.data_seed == spec['seed']
            self.optimization_start = verify_optimization_start(self.trainer, state, start)
            super().on_train_begin(args, state, control, **kwargs)
            assert self.trainer._probe_syncs[0]['adapter_hash'] == self.trainer._stage_loaded_hash
            self.expected_microsteps = (runtime.cumulative_microsteps(spec['updates'], len(ids), 32)
                - runtime.cumulative_microsteps(start, len(ids), 32))
            print('STAGE_OPTIMIZATION_START_VERIFIED', json.dumps(self.optimization_start), flush=True)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= spec['updates']:
                assert state.global_step == spec['updates']
                control.should_training_stop = control.should_save = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            trainer = self.trainer
            assert state.global_step == spec['updates']
            assert trainer._probe_micro_steps == self.expected_microsteps
            assert len(trainer._probe_syncs) == state.global_step - start + 1
            assert probe.fingerprint((n, p) for n, p in trainer.model.named_parameters() if not p.requires_grad) == trainer._probe_base_hash
            proof = {'start_step': start, 'global_step': state.global_step,
                'completed_updates': state.global_step - start, 'scheduler_horizon': 100,
                'micro_steps': trainer._probe_micro_steps, 'effective_batch': 32,
                'max_completion_length': 2048, 'base_unchanged_bitwise': True,
                'syncs': trainer._probe_syncs, 'sequential_context': context,
                'optimization_start': self.optimization_start,
                'initial_student_adapter_matched_loaded_checkpoint': bool(parent or resume),
                'train_seconds_including_diagnostic_hashing': time.monotonic() - trainer._probe_started,
                'scope': 'Stage training only; requires checkpoint, parentexit0 and all selected-language/English evaluations. Engineering handoff is not a research sequence.'}
            (output / 'COLLOCATE_TRAIN_UPDATES_DONE.json').write_text(json.dumps(proof, indent=2) + '\n')

    seeded.original_init = init
    runtime.ColocateCallback = StageCallback
    args = ['--train_language', spec['language'].upper(), '--translated_data_path', spec['training_data'],
        '--model_name_or_path', str(root / 'models/Qwen3-1.7B'), '--learning_rate', '5e-6',
        '--max_grad_norm', '0.1', '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '32',
        '--gradient_checkpointing', '--gradient_checkpointing_kwargs', '{"use_reentrant": false}',
        '--max_steps', '100', '--save_steps', '1' if spec['updates'] == 2 else '5', '--logging_steps', '1',
        '--output_dir', str(output), '--run_config', spec['run_id'], '--max_completion_length', '2048',
        '--max_length', '20000', '--attn_implementation', 'sdpa', '--torch_dtype', 'bfloat16',
        '--beta', '0', '--lmbda', '1', '--use_peft', '--lora_r', '64', '--lora_alpha', '128',
        '--lora_target_modules', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
        '--temperature', '1.1', '--top_p', '0.95', '--top_k', '20', '--fixed_teacher',
        '--student_enable_thinking', '--jsd_token_clip', '0.05', '--use_vllm', '--vllm_mode', 'colocate',
        '--vllm_tensor_parallel_size', '1', '--vllm_gpu_memory_utilization', '0.4',
        '--vllm_enable_sleep_mode', 'true', '--vllm_sync_frequency', '1', '--report_to', 'tensorboard',
        '--logging_dir', str(root / 'logs/copsd_official_10lang/tensorboard' / spec['run_id']),
        '--seed', str(spec['seed']), '--data_seed', str(spec['seed']), '--wandb_project', 'COPSD-sequential']
    if resume:
        args += ['--resume_from_checkpoint', str(resume)]
    sys.argv = [str(Path(__file__))] + args
    seeded.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        spec, _, _, _, _, start, _ = prepare(args.root, args.config)
        print(json.dumps({'CPU_preflight': True, 'GPU_initialized': False,
            'engineering_only': spec.get('engineering_only', False), 'start_step': start,
            'stop_after': spec['updates'], 'language': spec['language']}))
    else:
        execute(args.root, args.config)


if __name__ == '__main__':
    main()
