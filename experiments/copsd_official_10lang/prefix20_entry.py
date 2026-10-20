"""Project budget extension: stop at update2/20 of the unchanged100-step schedule.

No loss, optimizer, teacher, prompt, packing or rollout modification. The two
update mode is engineering validation;20 is the prespecified development repeat.
"""
import json
import os
from pathlib import Path
import time

import torch
import seeded_cost_batch_entry as seeded

runtime = seeded.batch.runtime
probe = runtime.probe


class PrefixCallback(runtime.ColocateCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.stop_after = int(os.environ['COPSD_STOP_AFTER_UPDATES'])
        assert self.stop_after in [2, 20] and args.max_steps == 100
        assert state.global_step == 0, 'Development repeats must start fresh'
        super().on_train_begin(args, state, control, **kwargs)
        self.expected_microsteps = runtime.cumulative_microsteps(
            self.stop_after, len(self.trainer.train_dataset), 32)
        print('PREFIX_BUDGET', json.dumps({'scheduler_horizon': args.max_steps,
            'stop_after_updates': self.stop_after, 'fresh_Base': True,
            'seed': args.seed, 'data_seed': args.data_seed}), flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after:
            assert state.global_step == self.stop_after
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        t = self.trainer
        assert args.max_steps == 100 and state.global_step == self.stop_after
        assert t._probe_micro_steps == self.expected_microsteps
        assert len(t._probe_syncs) == state.global_step + 1
        assert probe.fingerprint((n, p) for n, p in t.model.named_parameters()
                                 if not p.requires_grad) == t._probe_base_hash
        result = {'start_step': 0, 'global_step': state.global_step,
            'completed_updates': state.global_step,
            'scheduler_horizon': args.max_steps, 'stop_after_updates': self.stop_after,
            'micro_steps': t._probe_micro_steps, 'effective_batch': 32,
            'max_completion_length': 2048, 'base_unchanged_bitwise': True,
            'syncs': t._probe_syncs,
            'train_seconds_including_diagnostic_hashing': time.monotonic() - t._probe_started,
            'torch_peak_allocator_accounting_mib': torch.cuda.max_memory_allocated() / 2**20,
            'memory_caveat': 'Use external telemetry for physical peak.',
            'scope': 'Project prefix-budget repeat, not a completed100-update reproduction. Parent exit0 and development pairing remain required.'}
        (Path(args.output_dir) / 'COLLOCATE_TRAIN_UPDATES_DONE.json').write_text(
            json.dumps(result, indent=2) + '\n')
        print('COLLOCATE_TRAIN_UPDATES_DONE', json.dumps(result), flush=True)


def main():
    # runtime.main resolves this module global when it installs the callback.
    runtime.ColocateCallback = PrefixCallback
    seeded.main()


if __name__ == '__main__':
    main()
