"""Dense FP32-moment control matching sparse candidate arithmetic."""
import copy,math
import torch
from block_ops import adamw_candidate

class DenseFP32AdamW(torch.optim.Optimizer):
 def __init__(self,named_parameters,lr=5e-5,weight_decay=.01):
  self.named=dict(named_parameters)
  if not self.named:raise ValueError('Empty parameters')
  decay=[p for n,p in self.named.items() if not (n.endswith('.bias') or 'norm' in n.lower())]
  no_decay=[p for n,p in self.named.items() if n.endswith('.bias') or 'norm' in n.lower()]
  groups=[{'params':decay,'weight_decay':weight_decay},{'params':no_decay,'weight_decay':0.}]
  super().__init__(groups,dict(lr=lr,weight_decay=weight_decay))
 @torch.no_grad()
 def step(self,closure=None):
  loss=None
  if closure is not None:
   with torch.enable_grad():loss=closure()
  for group in self.param_groups:
   if not math.isfinite(group['lr']) or group['lr']<0:raise ValueError('Invalid learning rate')
   for p in group['params']:
    if p.grad is None or not torch.isfinite(p.grad).all():raise FloatingPointError('Missing/nonfinite gradient')
  for group in self.param_groups:
   for p in group['params']:
    state=self.state[p]
    if not state:state.update(step=0,m=torch.zeros_like(p,dtype=torch.float32),v=torch.zeros_like(p,dtype=torch.float32))
    delta,m,v=adamw_candidate(p,p.grad,state['m'],state['v'],state['step']+1,group['lr'],weight_decay=group['weight_decay'])
    p.copy_((p.float()+delta).to(p.dtype));state.update(step=state['step']+1,m=m,v=v)
  return loss
 def state_dict(self):
  d=super().state_dict();d['parameter_names']=list(self.named);return d
 def load_state_dict(self,state_dict):
  if state_dict['parameter_names']!=list(self.named):raise ValueError('Parameter identity mismatch')
  if len(state_dict['param_groups'])!=len(self.param_groups):raise ValueError('Group mismatch')
  pairs=[]
  for saved,current in zip(state_dict['param_groups'],self.param_groups):
   if len(saved['params'])!=len(current['params']):raise ValueError('Group size mismatch')
   for idx,p in zip(saved['params'],current['params']):
    s=state_dict['state'].get(idx)
    if s is not None:
     if set(s)!={'step','m','v'} or type(s['step'])!=int or s['step']<0:raise ValueError('Invalid state')
     if any(s[k].shape!=p.shape or s[k].dtype!=torch.float32 or not torch.isfinite(s[k]).all() for k in ['m','v']):raise ValueError('Invalid moment')
     if (s['v']<0).any():raise ValueError('Negative second moment')
     pairs.append((p,s))
  # Generic loader handles scheduler/group metadata; replace its dtype-cast moments.
  super().load_state_dict({k:state_dict[k] for k in ['state','param_groups']})
  for p,s in pairs:self.state[p]={'step':s['step'],'m':s['m'].to(p.device).clone(),'v':s['v'].to(p.device).clone()}
