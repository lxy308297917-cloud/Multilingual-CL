"""Independent batched-rollout adaptation of the validated v1 entry.

Does not modify the frozen trainer or existing TE/JA runs. Keeps all official
sampling, packing, teacher, loss, optimizer and accumulation settings.
"""
import json
from pathlib import Path
from transformers import TrainerCallback
import vllm_colocate_entry as runtime
from vllm_update_batch import UpdateBatch

probe = runtime.probe
old_init = probe.instrumented_init
old_sync = probe.nonmutating_sync


class BatchEnd(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        t = self.trainer
        controller = t._update_batch
        controller.assert_drained()
        assert sum(r['rollouts'] for r in controller.records) == t._probe_micro_steps
        rows = list(controller.records)
        controller.detach()
        t._update_batch = None
        (Path(args.output_dir) / 'UPDATE_BATCH_GENERATION.json').write_text(
            json.dumps({'groups': rows, 'stochastic_replay_equivalence_claimed': False}, indent=2) + '\n')


def instrumented_init(self, *args, **kwargs):
    old_init(self, *args, **kwargs)
    assert not self.reason_first and self.vllm_tensor_parallel_size == 1
    self._update_batch = UpdateBatch(self)
    self._update_batch.install()
    callback = BatchEnd()
    callback.trainer = self
    self.add_callback(callback)


def sync(self):
    controller = getattr(self, '_update_batch', None)
    if controller is not None:
        controller.assert_drained()
    return old_sync(self)


def generation(self, *args, **kwargs):
    assert self._last_vllm_sync_step == self.state.global_step
    current = probe.fingerprint((n, p) for n, p in self.model.named_parameters() if p.requires_grad)
    assert current == self._probe_syncs[-1]['adapter_hash'], 'Rollout policy changed within accumulation'
    # Official packing and mask/label construction still execute per microbatch.
    # Cached outputs stay on CPU; do not hash sleeping/unmapped inference tensors.
    return probe.original_generation(self, *args, **kwargs)


def main():
    probe.instrumented_init = instrumented_init
    probe.nonmutating_sync = sync
    probe.instrumented_generation = generation
    runtime.main()


if __name__ == '__main__':
    main()
