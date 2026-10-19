"""Read-only loss-mask counts and forward timers; never replaces the loss."""
import inspect
import torch


class CostTracker:
    def __init__(self):
        self.records=[]
        self.events=[]
        self.handles=[]
        self.trainer=None

    def observe_loss(self,original):
        signature=inspect.signature(original)
        def observed(*args,**kwargs):
            values=signature.bind(*args,**kwargs);values.apply_defaults()
            labels=values.arguments['labels'];assert labels is not None
            valid=int((labels!=-100).sum())
            self.records.append({'step_before_update':self.trainer.state.global_step,
                                 'valid_response_positions':valid,
                                 'masked_response_positions':labels.numel()-valid})
            return original(*args,**kwargs)
        return observed

    def attach(self,trainer):
        assert self.trainer is None
        self.trainer=trainer
        stack=[]
        def begin(module,args,kwargs):
            ids=kwargs['input_ids'];mask=kwargs.get('attention_mask')
            count=ids.numel();attended=int(mask.sum()) if mask is not None else count
            first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
            first.record()
            stack.append(('student' if torch.is_grad_enabled() else 'teacher',first,last,count,attended))
        def end(module,args,kwargs,output):
            row=stack.pop();row[2].record();self.events.append(row)
        self.handles=[trainer.model.register_forward_pre_hook(begin,with_kwargs=True),
                      trainer.model.register_forward_hook(end,with_kwargs=True)]

    def snapshot(self):
        torch.cuda.synchronize()
        result={'loss_calls':len(self.records),
                'valid_response_loss_positions':sum(r['valid_response_positions'] for r in self.records),
                'masked_response_positions':sum(r['masked_response_positions'] for r in self.records)}
        for kind in ['student','teacher']:
            rows=[r for r in self.events if r[0]==kind]
            assert len(rows)==len(self.records),(kind,len(rows),len(self.records))
            result[kind+'_forward_calls']=len(rows)
            result[kind+'_forward_cuda_seconds']=sum(a.elapsed_time(b) for _,a,b,_,_ in rows)/1000
            result[kind+'_processed_input_positions']=sum(r[3] for r in rows)
            result[kind+'_attended_input_positions']=sum(r[4] for r in rows)
        return result

    def detach(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.events=[];self.trainer=None
