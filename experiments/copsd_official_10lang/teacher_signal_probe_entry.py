"""Two-update instrumentation of real on-policy privileged-teacher signals.

All original sampling, prompts, masks, fixed-teacher and loss outputs retained.
Additional detached statistics add overhead: not a throughput benchmark.
"""
import json
from pathlib import Path
import sys
import torch
from transformers import TrainerCallback
import vllm_update_batch_entry as batch
from observe_teacher_clip import make_observer

records=[]
original_init=batch.old_init


class SignalReport(TrainerCallback):
    def __init__(self,trainer):
        self.trainer=trainer

    def on_train_end(self,args,state,control,**kwargs):
        assert state.global_step==2 and len(records) in (32,64)
        totals={k:sum(r[k] for r in records) for k in records[0]}
        valid=totals['valid_response_positions'];assert valid>0
        summary={'calls':len(records),'totals':totals,
          'loss_temperature':self.trainer.temperature,'component_clip':self.trainer.jsd_token_clip,'beta':self.trainer.beta,
          'teacher_argmax_component_clipped_fraction':totals['teacher_argmax_component_clipped_positions']/valid,
          'argmax_disagreement_fraction':totals['argmax_disagreement_positions']/valid,
          'clipped_teacher_mass_fraction_estimate':totals['clipped_teacher_probability_mass_sum']/totals['teacher_probability_mass_sum'],
          'mean_unclipped_KL_component_sum_float32':totals['unclipped_component_sum']/valid,
          'mean_clipped_component_sum_float32':totals['clipped_component_sum']/valid,
          'mean_teacher_minus_student_sampled_logprob':totals['teacher_minus_student_sampled_logprob_sum']/valid,
          'returned_loss_uses_unmodified_original_function':True,
          'scope':'Real on-policy two-update training signals, not full100-update downstream efficacy. Float32 diagnostic reductions of original-dtype components. Instrumentation overhead included.'}
        t=self.trainer
        torch.cuda.synchronize()
        for handle in t._signal_handles:handle.remove()
        events=t._signal_events
        summary['teacher_forward_calls']=sum(kind=='teacher' for kind,_,_ in events)
        summary['student_forward_calls']=sum(kind=='student' for kind,_,_ in events)
        assert summary['teacher_forward_calls']==summary['student_forward_calls']==len(records)
        summary['teacher_forward_cuda_seconds']=sum(begin.elapsed_time(end) for kind,begin,end in events if kind=='teacher')/1000
        summary['student_forward_cuda_seconds']=sum(begin.elapsed_time(end) for kind,begin,end in events if kind=='student')/1000
        summary['forward_timer_scope']='Outer PEFT forward CUDA events only; excludes rollout, KL, backward, optimizer and saving. Instrumented probe, not formal-run throughput.'
        t._signal_events=[]
        t._signal_handles=[]
        (Path(args.output_dir)/'TEACHER_SIGNAL_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
        (Path(args.output_dir)/'TEACHER_SIGNAL_CALLS.PRIVATE.json').write_text(json.dumps(records,indent=2)+'\n')
        print('TEACHER_SIGNAL_SUMMARY',json.dumps(summary),flush=True)


def init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    self._signal_events=[]
    stack=[]
    def begin(module,inputs):
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
        first.record();stack.append(('student' if torch.is_grad_enabled() else 'teacher',first,last))
    def end(module,inputs,output):
        kind,first,last=stack.pop();last.record();self._signal_events.append((kind,first,last))
    self._signal_handles=[self.model.register_forward_pre_hook(begin),self.model.register_forward_hook(end)]
    self.add_callback(SignalReport(self))


def main():
    if '--help' not in sys.argv:
        assert '--max_steps' in sys.argv and sys.argv[sys.argv.index('--max_steps')+1]=='2'
    original=batch.probe.core.OPSDTrainer.generalized_jsd_loss
    batch.probe.core.OPSDTrainer.generalized_jsd_loss=staticmethod(make_observer(original,records))
    batch.old_init=init
    batch.main()


if __name__=='__main__':main()
