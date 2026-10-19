"""Seed before PEFT initialization; measure exact masks and forward cost.

Official core loss and optimization stay unchanged. Earlier unseeded-initializer
runs remain separately labelled. Stack watchdog only dumps; never kills/retries.
"""
import faulthandler
import json
from pathlib import Path
from transformers import TrainerCallback,set_seed
import vllm_update_batch_entry as batch
from cost_instrumentation import CostTracker

tracker=CostTracker()
original_init=batch.old_init


def arm_watchdog():
    faulthandler.dump_traceback_later(600,repeat=False)


class CostReport(TrainerCallback):
    def on_train_begin(self,args,state,control,**kwargs):
        self.start_step=state.global_step
        arm_watchdog()

    def on_step_begin(self,args,state,control,**kwargs):arm_watchdog()
    def on_step_end(self,args,state,control,**kwargs):arm_watchdog()

    def write(self,args,state,name):
        data=tracker.snapshot()
        data.update(start_step=self.start_step,global_step=state.global_step,
                    seed_applied_before_model_and_PEFT=args.seed,
                    loss_temperature=tracker.trainer.temperature,
                    scope='This process segment only; parent continuation ledgers remain separate.',
                    forward_timer_scope='Outer model forward CUDA events; excludes rollout, KL, backward, optimizer and saving.',
                    method='Original official loss returned unchanged, exact shifted-label masks counted.')
        path=Path(args.output_dir)/name
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)
        return data

    def on_save(self,args,state,control,**kwargs):
        data=self.write(args,state,'COST_SEGMENT_PROGRESS.json')
        checkpoint=Path(args.output_dir)/f'checkpoint-{state.global_step}'
        assert checkpoint.is_dir()
        (checkpoint/'COST_SEGMENT_PROGRESS.json').write_text(json.dumps(data,indent=2)+'\n')
        print('COST_PROGRESS',json.dumps(data),flush=True)

    def on_train_end(self,args,state,control,**kwargs):
        faulthandler.cancel_dump_traceback_later()
        expected=(batch.runtime.cumulative_microsteps(state.global_step,len(tracker.trainer.train_dataset),32)
                 -batch.runtime.cumulative_microsteps(self.start_step,len(tracker.trainer.train_dataset),32))
        assert len(tracker.records)==expected
        data=self.write(args,state,'COST_SEGMENT_DONE.json')
        (Path(args.output_dir)/'COST_CALLS.PRIVATE.json').write_text(json.dumps(tracker.records,indent=2)+'\n')
        print('COST_SEGMENT_DONE',json.dumps(data),flush=True)
        tracker.detach()


def init(self,*args,**kwargs):
    training_args=kwargs['args']
    set_seed(training_args.seed)
    original_init(self,*args,**kwargs)
    assert self.fixed_teacher and not self.use_thinking_machines_loss
    tracker.attach(self)
    self.add_callback(CostReport())


def main():
    original=batch.probe.core.OPSDTrainer.generalized_jsd_loss
    batch.probe.core.OPSDTrainer.generalized_jsd_loss=staticmethod(tracker.observe_loss(original))
    batch.old_init=init
    try:batch.main()
    finally:faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
