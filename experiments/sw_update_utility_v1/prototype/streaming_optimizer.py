"""Streaming reference variant. On nonfinite state, abort and restore checkpoint."""
import torch
from optimizer import BlockAdamW
from block_ops import utility_scores,block_sum,select_equal_blocks,apply_delta

class StreamingBlockAdamW(BlockAdamW):
    def needs_refresh(self):
        return self.steps%self.settings['refresh']==0 or getattr(self,'deferred_refresh',False)
    def select_masks(self,scores):
        c=self.settings
        return dict(zip(self.params,select_equal_blocks(scores,c['fraction'],positive_only=c['mode']=='utility')))
    def candidate(self,name):
        p=self.params[name];s=self.state[name];c=self.settings;t=self.steps+1
        # Same FP32 definition as the reference. No all-model candidate allocation.
        return -c['lr']*(s['m']/(1-.9**t))/((s['v']/(1-.999**t)).sqrt()+1e-8)-c['lr']*c['weight_decay']*p.float()
    @torch.no_grad()
    def step(self,source_grads=None,target_score_grads=None):
        c=self.settings;refresh=self.needs_refresh()
        if refresh and c['mode']=='utility' and (source_grads is None or set(source_grads)!=set(self.params)):
            raise ValueError('Refresh needs separate source gradients')
        for name,p in self.params.items():
            if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Missing/nonfinite target gradient')
            if refresh and c['mode']=='utility':
                s=source_grads[name]
                if s.shape!=p.shape or not torch.isfinite(s).all():raise ValueError('Invalid source gradient')
        if target_score_grads is not None:
            if set(target_score_grads)!=set(self.params):raise ValueError('Target scoring gradient identity mismatch')
            for name,p in self.params.items():
                g=target_score_grads[name]
                if g.shape!=p.shape or not torch.isfinite(g).all():raise ValueError('Invalid target scoring gradient')
        scores=[];diagnostic_blocks={}
        for name,p in self.params.items():
            if name not in self.state:self.state[name]={'m':torch.zeros_like(p,dtype=torch.float32),'v':torch.zeros_like(p,dtype=torch.float32)}
            s=self.state[name];g=p.grad.float()
            # Keep arithmetic equivalent to reference; temporary allocation is per matrix.
            s['m'].copy_(s['m']*.9+g*.1)
            s['v'].copy_(s['v']*.999+g.square()*.001)
            d=self.candidate(name)
            if not torch.isfinite(d).all():raise FloatingPointError('Nonfinite candidate; restore checkpoint before retry')
            if refresh:
                if c['mode']=='utility':
                    scoring_grad=g if target_score_grads is None else target_score_grads[name].to(p.device)
                    score,utility,risk=utility_scores(scoring_grad,source_grads[name].to(p.device),d,c['risk_lambda'],c['block_size'])
                    diagnostic_blocks[name]={'utility':utility.cpu(),'risk':risk.cpu()}
                else:score=block_sum(s['m'].abs(),c['block_size'])
                diagnostic_blocks.setdefault(name,{})['score']=score.cpu()
                scores.append(score)
            del d,g
        if refresh:
            self.masks=self.select_masks(scores);self.last_diagnostics={}
        for name,p in self.params.items():
            d=self.candidate(name)
            if refresh:
                blocks=self.masks[name];selected=blocks.cpu();values=diagnostic_blocks[name]
                info={'candidate_blocks':blocks.numel(),'selected_blocks':int(blocks.sum())}
                for key,value in values.items():
                    info[key+'_sum']=float(value.sum());info[key+'_selected_sum']=float(value[selected].sum())
                active=blocks.repeat_interleave(c['block_size'],0).repeat_interleave(c['block_size'],1)
                previous=p[active].float();proposal=d[active].float()
                actual=(previous+proposal).to(p.dtype).float()-previous
                info.update(proposed_squared_norm=float(proposal.square().sum()),applied_squared_norm=float(actual.square().sum()),rounding_squared_error=float((actual-proposal).square().sum()),changed_coordinates=int((actual!=0).sum()))
                self.last_diagnostics[name]=info
                del active,previous,proposal,actual
            apply_delta(p,d,self.masks[name],c['block_size']);del d
        self.deferred_refresh=c['lr']==0
        self.steps+=1
        return {'step':self.steps,'refreshed':refresh,'selected_blocks':sum(int(m.sum()) for m in self.masks.values())}
