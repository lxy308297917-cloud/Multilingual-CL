"""Official-derived single-GPU OPD with validated rollout adaptations.

Uses the immutable two-update probe helpers. Supports a fresh run or a real
Trainer checkpoint continuation with a new output directory. The official
trainer, collator, AdamW and clipped KD remain unchanged.
"""
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

import torch
from transformers import TrainerCallback
import vllm_colocate_probe_entry as probe
from resume_torch28_entry import enable_numpy_rng_compat
from vllm_ordered_shutdown import shutdown


def cumulative_microsteps(step, rows, accumulation):
    epochs, tail = divmod(step, math.ceil(rows / accumulation))
    return epochs * rows + min(tail * accumulation, rows)


class ColocateCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        t = self.trainer
        assert t.use_vllm and t.vllm_mode == 'colocate'
        assert t.accelerator.num_processes == 1
        assert args.per_device_train_batch_size == 1
        assert args.gradient_accumulation_steps == 32
        assert args.max_completion_length == 2048
        assert args.max_steps in (2, 100)
        self.start_step = state.global_step
        assert 0 <= self.start_step < args.max_steps
        self.expected_microsteps = (
            cumulative_microsteps(args.max_steps, len(t.train_dataset), 32)
            - cumulative_microsteps(self.start_step, len(t.train_dataset), 32))
        t._probe_base_hash = probe.fingerprint(
            (n, p) for n, p in t.model.named_parameters() if not p.requires_grad)
        t._probe_started = time.monotonic()
        restore = {'start_step': state.global_step}
        if state.global_step:
            steps = {int(v['step'].item()) for v in t.optimizer.state_dict()['state'].values()}
            assert steps == {state.global_step}, steps
            assert t.lr_scheduler.state_dict()['last_epoch'] == state.global_step
            restore.update(optimizer_steps=sorted(steps), scheduler_step=state.global_step)
        t._move_model_to_vllm()
        t._last_vllm_sync_step = state.global_step
        reference = os.environ.get('COPSD_EXPECT_INITIAL_SYNC_JSON')
        if reference:
            old = json.loads(Path(reference).read_text())
            prior = next(x for x in old['syncs'] if x['global_step'] == state.global_step)
            assert t._probe_syncs[0]['adapter_hash'] == prior['adapter_hash']
            assert t._probe_syncs[0]['engine_hash'] == prior['engine_hash']
            restore['loaded_adapter_and_inference_weights_match_parent'] = True
        print('COLLOCATE_RESTORE_VERIFIED', json.dumps(restore), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        t = self.trainer
        assert state.global_step == args.max_steps
        assert t._probe_micro_steps == self.expected_microsteps
        assert len(t._probe_syncs) == state.global_step - self.start_step + 1
        assert probe.fingerprint(
            (n, p) for n, p in t.model.named_parameters() if not p.requires_grad) == t._probe_base_hash
        result = {
            'start_step': self.start_step, 'global_step': state.global_step,
            'completed_updates': state.global_step - self.start_step,
            'micro_steps': t._probe_micro_steps, 'effective_batch': 32,
            'max_completion_length': 2048, 'base_unchanged_bitwise': True,
            'syncs': t._probe_syncs,
            'train_seconds_including_diagnostic_hashing': time.monotonic() - t._probe_started,
            'torch_peak_allocator_accounting_mib': torch.cuda.max_memory_allocated() / 2**20,
            'memory_caveat': 'Allocator accounting with sleeping pools is not physical board peak; use external telemetry.',
            'scope': 'Update completion only; requires checkpoint files, normal exit and independent downstream evaluation.'}
        (Path(args.output_dir) / 'COLLOCATE_TRAIN_UPDATES_DONE.json').write_text(
            json.dumps(result, indent=2) + '\n')
        print('COLLOCATE_TRAIN_UPDATES_DONE', json.dumps(result), flush=True)


def main():
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
    assert ',' not in os.environ['CUDA_VISIBLE_DEVICES']
    enable_numpy_rng_compat()
    # instrumented_init resolves this callback through its module globals.
    probe.ProbeCallback = ColocateCallback
    core = probe.core
    core.OPSDTrainer.__init__ = probe.instrumented_init
    core.OPSDTrainer._move_model_to_vllm = probe.nonmutating_sync
    core.OPSDTrainer._generate_on_policy_outputs_vllm = probe.instrumented_generation
    core.OPSDTrainer.training_step = probe.instrumented_training_step
    core.OPSDTrainer._wake_vllm_if_needed = probe.preserving_wake
    namespace = runpy.run_path(str(probe.SOURCE / 'multilingual_opsd_train.py'), run_name='__main__')
    trainer = namespace['trainer']
    for callback in trainer.callback_handler.callbacks:
        writer = getattr(callback, 'tb_writer', None)
        if writer is not None:
            writer.flush()
            writer.close()
    import wandb
    wandb.finish()
    shutdown(trainer, probe.wake_sleeping_allocations)
    (Path(trainer.args.output_dir) / 'RUNTIME_SHUTDOWN_DONE.json').write_text(
        json.dumps({'ordered_pool_release_completed': True,
                    'scope': 'Parent queue must separately confirm exit code0.'}) + '\n')
    print('ORDERED_RUNTIME_SHUTDOWN_DONE', flush=True)


if __name__ == '__main__':
    main()
