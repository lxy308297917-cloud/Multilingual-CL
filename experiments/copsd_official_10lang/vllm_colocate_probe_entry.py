"""Instrument the frozen COPSD trainer for a two-update, full-length GPU probe.

Project adaptations: non-mutating LoRA export and explicit initial policy sync.
No change to rollout prompts, teacher context, optimizer or clipped KD loss.
This entry only supports one GPU, ordinary LoRA and the frozen local source.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

import torch
from transformers import TrainerCallback
from lora_sync_weights import iter_merged_weights

SOURCE = Path(os.environ['COPSD_OFFICIAL_SOURCE'])
sys.path.insert(0, str(SOURCE))
import multilingual_opsd_trainer as core


def fingerprint(parameters):
    digest = hashlib.sha256()
    for name, tensor in parameters:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ProbeCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.trainer
        assert trainer.use_vllm and trainer.vllm_mode == 'colocate'
        assert trainer.accelerator.num_processes == 1
        assert args.max_steps == 2 and args.gradient_accumulation_steps == 32
        assert args.per_device_train_batch_size == 1
        trainer._probe_base_hash = fingerprint((n, p) for n, p in trainer.model.named_parameters() if not p.requires_grad)
        trainer._probe_started = time.monotonic()
        # Released callback first syncs at step end. An explicit initial sync is
        # required when loading a trained adapter; also verify it for fresh LoRA.
        trainer._move_model_to_vllm()
        trainer._last_vllm_sync_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        assert state.global_step == 2
        assert fingerprint((n, p) for n, p in trainer.model.named_parameters() if not p.requires_grad) == trainer._probe_base_hash
        assert len(trainer._probe_syncs) == 3
        assert trainer._probe_syncs[0]['engine_hash'] != trainer._probe_syncs[1]['engine_hash']
        assert trainer._probe_syncs[1]['engine_hash'] != trainer._probe_syncs[2]['engine_hash']
        result = {'global_step': state.global_step, 'micro_steps': trainer._probe_micro_steps,
                  'effective_batch': 32, 'max_completion_length': 2048,
                  'base_unchanged_bitwise': True, 'syncs': trainer._probe_syncs,
                  'train_seconds_including_diagnostic_hashing': time.monotonic() - trainer._probe_started,
                  'torch_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'scope': 'Engineering probe only; no downstream effectiveness claim. GPU UUIDs and raw commands remain local.'}
        assert result['micro_steps'] == 64
        path = Path(args.output_dir) / 'COLLOCATE_PROBE_UPDATES_DONE.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        print('COLLOCATE_PROBE_UPDATES_DONE', json.dumps(result), flush=True)


def install_preserving_sleep(engine):
    # After the initial full sync, consecutive accumulation microsteps must
    # reuse the same policy. vLLM level2 discards weights; level1 backs them up.
    original_sleep = engine.sleep

    def preserving_sleep(level=1):
        return original_sleep(level=1)

    engine.sleep = preserving_sleep


def wake_sleeping_allocations(engine):
    executor = engine.llm_engine.model_executor
    if executor.is_sleeping:
        # At an optimizer boundary sync already woke weights; only KV remains.
        # Between microsteps both tags sleep. Do not ask to remap awake tags.
        engine.wake_up(tags=sorted(executor.sleeping_tags))


def preserving_wake(self):
    if self.vllm_mode == 'colocate' and self.vllm_enable_sleep_mode:
        core.empty_cache()
        wake_sleeping_allocations(self.vllm_engine)


original_init = core.OPSDTrainer.__init__
original_generation = core.OPSDTrainer._generate_on_policy_outputs_vllm
original_training_step = core.OPSDTrainer.training_step


def instrumented_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._probe_syncs = []
    self._probe_micro_steps = 0
    if self.vllm_enable_sleep_mode:
        install_preserving_sleep(self.vllm_engine)
    self.add_callback(ProbeCallback(self))


def nonmutating_sync(self):
    assert self.vllm_mode == 'colocate' and self.accelerator.num_processes == 1
    assert not self.is_fsdp_enabled and self.accelerator.state.deepspeed_plugin is None
    started = time.monotonic()
    if self.vllm_enable_sleep_mode:
        core.empty_cache()
        self.vllm_engine.wake_up(tags=['weights'])
    engine_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
    engine_model.load_weights(iter_merged_weights(self.model))
    self.vllm_engine.reset_prefix_cache()
    assert fingerprint((n, p) for n, p in self.model.named_parameters() if not p.requires_grad) == self._probe_base_hash
    adapter_hash = fingerprint((n, p) for n, p in self.model.named_parameters() if p.requires_grad)
    row = {'global_step': self.state.global_step, 'adapter_hash': adapter_hash,
           'engine_hash': fingerprint(engine_model.named_parameters()),
           'seconds_including_diagnostic_hashing': time.monotonic() - started}
    self._probe_syncs.append(row)
    print('COLLOCATE_SYNC_VERIFIED', json.dumps(row), flush=True)


def instrumented_generation(self, *args, **kwargs):
    assert self._last_vllm_sync_step == self.state.global_step
    current = fingerprint((n, p) for n, p in self.model.named_parameters() if p.requires_grad)
    assert current == self._probe_syncs[-1]['adapter_hash'], 'Stale rollout policy'
    if self._probe_micro_steps in (1, 33):
        engine_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
        assert fingerprint(engine_model.named_parameters()) == self._probe_syncs[-1]['engine_hash'], 'Restored inference weights differ'
        print('COLLOCATE_SLEEP_WEIGHT_RESTORE_VERIFIED', self._probe_micro_steps, flush=True)
    return original_generation(self, *args, **kwargs)


def instrumented_training_step(self, *args, **kwargs):
    started = time.monotonic()
    result = original_training_step(self, *args, **kwargs)
    torch.cuda.synchronize()
    self._probe_micro_steps += 1
    print('COLLOCATE_MICROSTEP', json.dumps({'global_step': self.state.global_step,
          'micro_step': self._probe_micro_steps, 'seconds': time.monotonic() - started,
          'loss_finite': bool(torch.isfinite(result).all())}), flush=True)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Non-finite KD loss')
    return result


if __name__ == '__main__':
    if '--help' not in sys.argv:
        assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
        assert os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
        assert ',' not in os.environ['CUDA_VISIBLE_DEVICES']
    core.OPSDTrainer.__init__ = instrumented_init
    core.OPSDTrainer._move_model_to_vllm = nonmutating_sync
    core.OPSDTrainer._generate_on_policy_outputs_vllm = instrumented_generation
    core.OPSDTrainer.training_step = instrumented_training_step
    core.OPSDTrainer._wake_vllm_if_needed = preserving_wake
    runpy.run_path(str(SOURCE / 'multilingual_opsd_train.py'), run_name='__main__')
