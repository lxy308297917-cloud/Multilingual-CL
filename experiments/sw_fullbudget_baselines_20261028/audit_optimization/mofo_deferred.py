"""MoFO engineering candidate: exact threshold/gathers, device unions, one readback.

The frozen baseline remains unchanged. This class is not used by formal training
until CPU/CUDA, real-model recovery and isolated throughput acceptance pass.
"""
from pathlib import Path
import sys,math,torch,numpy as np
FROZEN=Path(__file__).resolve().parents[2]/'code';sys.path.insert(0,str(FROZEN))
from optimizer import BaselineAdamW

class DeferredMoFO(BaselineAdamW):
 def __init__(self,named,method='MoFO',static=None,shapes=None,quota=None):
  assert method=='MoFO'
  super().__init__(named,method,static=static,shapes=shapes,quota=quota)
  self.unions={n:v.to(self.named[n].device) for n,v in self.unions.items()}

 @torch.no_grad()
 def step(self,lr):
  items=list(self.named.items());assert all(p.grad is not None for _,p in items)
  # One synchronized check before updates, rather than196 separate grad checks.
  assert bool(torch.stack([torch.isfinite(p.grad).all() for _,p in items]).all())
  t=self.steps+1;device_stats=[];selected_counts=[]
  for n,p in items:
   m,v=self.state[n]['m'],self.state[n]['v'];g=p.grad.float()
   m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
   score=m.abs();k=math.floor(p.numel()*.15)
   threshold=torch.topk(score.flatten(),k,sorted=False).values.min()
   idx=torch.where(score.flatten()>=threshold)[0] # all ties retained, every step
   self.indices[n]=idx;selected_counts.append(idx.numel())
   # Exact same pointwise FP32 expression, evaluated only at selected coordinates.
   selected_m=m.flatten().index_select(0,idx);selected_v=v.flatten().index_select(0,idx)
   delta=-lr*(selected_m/(1-.9**t))/((selected_v/(1-.999**t)).sqrt()+1e-8)
   self.unions[n][idx]=True
   wd=0. if ('norm' in n or n.endswith('.bias')) else .01
   old_full=p.detach().float() # BF16->FP32 already copies; no second clone needed
   if p.dtype==torch.float32:old_full=old_full.clone()
   p.mul_(1-lr*wd)
   old=p.flatten().index_select(0,idx);new=(old.float()+delta).to(p.dtype)
   p.flatten().index_copy_(0,idx,new)
   actual=p.float()-old_full
   d2=(delta-lr*wd*old_full.flatten().index_select(0,idx)).square().sum()
   a2=actual.square().sum();nz=torch.count_nonzero(actual);union=self.unions[n].sum()
   # Each original per-matrix reduction stays FP32/int64; only its scalar is cast.
   device_stats.append(torch.stack((a2.double(),d2.double(),nz.double(),union.double(),torch.isfinite(new).all().double())))
   del g,score,selected_m,selected_v,delta,old_full,old,new,actual
  host=torch.stack(device_stats).cpu().tolist()
  stats=dict(step=t,lr=lr,selected=0,nonzero_write=0,actual_delta_sq=0.,candidate_delta_sq=0.,mask_union=0,per_module={})
  for (n,p),count,row in zip(items,selected_counts,host):
   a2,d2,nz,union,finite=row;assert finite==1
   self.union_counts[n]=int(union)
   bucket='Attention' if '.self_attn.' in n else 'MLP' if '.mlp.' in n else 'Other'
   sub=stats['per_module'].setdefault(bucket,dict(selected=0,nonzero_write=0,actual_delta_sq=0.))
   for target in (stats,sub):
    target['selected']+=count;target['nonzero_write']+=int(nz);target['actual_delta_sq']+=a2
   stats['candidate_delta_sq']+=d2;stats['mask_union']+=int(union)
  self.steps=t;stats['actual_norm']=math.sqrt(stats['actual_delta_sq']);return stats

 def state_dict(self):
  return dict(method=self.method,names=list(self.named),quota=self.quota,steps=self.steps,calibrations=self.calibrations,
   state={n:{k:v.cpu() for k,v in s.items()} for n,s in self.state.items()},english=self.english,
   indices={n:v.cpu() for n,v in self.indices.items()},
   unions={n:torch.from_numpy(np.packbits(v.cpu().numpy(),bitorder='little')) for n,v in self.unions.items()})

 def load_state_dict(self,state):
  super().load_state_dict(state)
  self.unions={n:v.to(self.named[n].device) for n,v in self.unions.items()}
