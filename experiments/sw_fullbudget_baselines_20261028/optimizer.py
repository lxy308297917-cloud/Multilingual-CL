"""Baseline identities remain explicit: official MoFO decay is not mask-restricted."""
import math,torch,numpy as np

def exact_indices(score,k):
 flat=score.flatten();assert 0<k<=flat.numel()
 if k==flat.numel():return torch.arange(k,device=flat.device)
 threshold=torch.topk(flat,k,sorted=False).values.min()
 greater=torch.where(flat>threshold)[0];ties=torch.where(flat==threshold)[0]
 return torch.cat((greater,ties[:k-greater.numel()])).sort().values

class BaselineAdamW:
 def __init__(self,named,method,static=None,shapes=None,quota=None):
  self.named=dict(named);self.method=method;self.steps=0;self.english={};self.calibrations=0
  self.quota=dict(quota) if quota is not None else None
  if self.quota is not None:assert self.quota.keys()==self.named.keys()
  self.state={n:{k:torch.zeros_like(p,dtype=torch.float32) for k in ('m','v')} for n,p in self.named.items()}
  self.indices={} if static is None else {n:i.to(p.device) for n,p in self.named.items() for i in [static[n]]}
  self.unions={n:torch.zeros(p.numel(),dtype=torch.bool) for n,p in self.named.items()}
  self.union_counts={n:0 for n in self.named}
 @torch.no_grad()
 def calibrate(self,grads):
  assert self.method=='Protected' and self.steps%200==0
  for n,g in grads.items():
   x=g.float().cpu().square();assert torch.isfinite(x).all()
   if n not in self.english:self.english[n]=x
   else:self.english[n].mul_(.9).add_(x,alpha=.1)
  self.calibrations+=1
 @torch.no_grad()
 def step(self,lr):
  t=self.steps+1;stats=dict(step=t,lr=lr,selected=0,nonzero_write=0,actual_delta_sq=0.,candidate_delta_sq=0.,mask_union=0,per_module={})
  for n,p in self.named.items():
   assert p.grad is not None and torch.isfinite(p.grad).all()
   if self.method=='Dense':
    # Chunk the tied233M embedding: full arange/gather would exceed3090 memory.
    m,v=self.state[n]['m'].flatten(),self.state[n]['v'].flatten();pf=p.flatten();gf=p.grad.flatten()
    wd=0. if ('norm' in n or n.endswith('.bias')) else .01
    a2=d2=0.;nz=0
    for start in range(0,p.numel(),1<<20):
     stop=min(start+(1<<20),p.numel());g=gf[start:stop].float();mm=m[start:stop];vv=v[start:stop]
     mm.mul_(.9).add_(g,alpha=.1);vv.mul_(.999).addcmul_(g,g,value=.001)
     old=pf[start:stop].float();d=-lr*(mm/(1-.9**t))/((vv/(1-.999**t)).sqrt()+1e-8)-lr*wd*old
     new=(old+d).to(p.dtype);actual=new.float()-old
     assert torch.isfinite(new).all();a2+=float(actual.square().sum());d2+=float(d.square().sum());nz+=int(torch.count_nonzero(actual));pf[start:stop].copy_(new)
    if t==1:self.unions[n].fill_(True);self.union_counts[n]=p.numel()
    sub=stats['per_module'].setdefault('All',dict(selected=0,nonzero_write=0,actual_delta_sq=0.))
    for target in (stats,sub):target['selected']+=p.numel();target['nonzero_write']+=nz;target['actual_delta_sq']+=a2
    stats['candidate_delta_sq']+=d2;stats['mask_union']+=p.numel()
    continue
   m,v=self.state[n]['m'],self.state[n]['v'];g=p.grad.float()
   m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
   delta=-lr*(m/(1-.9**t))/((v/(1-.999**t)).sqrt()+1e-8)
   wd=0. if ('norm' in n or n.endswith('.bias')) else .01
   refresh=self.method=='MoFO' or (self.method in ('Element','Protected') and self.steps%20==0)
   if refresh:
    score=m.abs()
    if self.method=='Protected':
     e=self.english[n].to(p.device);score=(score/(score.mean()+1e-12))/(1+e/(e.mean()+1e-12));del e
    k=self.quota[n] if self.quota is not None else math.floor(p.numel()*.15)
    if self.method=='MoFO':
     # Official threshold retains all ties (may exceed nominal k).
     threshold=torch.topk(score.flatten(),k,sorted=False).values.min()
     idx=torch.where(score.flatten()>=threshold)[0]
    else:idx=exact_indices(score,k)
    self.indices[n]=idx
   idx=self.indices.get(n)
   if idx is None:idx=torch.arange(p.numel(),device=p.device)
   if refresh or t==1:
    self.unions[n][idx.cpu()]=True;self.union_counts[n]=int(self.unions[n].sum())
   # Official MoFO performs a separate full-parameter decay before sparse addcdiv.
   if self.method=='MoFO':
    old_full=p.detach().float().clone();p.mul_(1-lr*wd)
    old=p.flatten()[idx];new=(old.float()+delta.flatten()[idx]).to(p.dtype)
    p.flatten().index_copy_(0,idx,new)
    actual=p.float()-old_full
    d2=float((delta.flatten()[idx]-lr*wd*old_full.flatten()[idx]).square().sum())
    a2=float(actual.square().sum());nz=int(torch.count_nonzero(actual));del old_full,actual
   else:
    old=p.flatten()[idx];d=delta.flatten()[idx]-lr*wd*old.float();new=(old.float()+d).to(p.dtype)
    actual=new.float()-old.float();a2=float(actual.square().sum());nz=int(torch.count_nonzero(actual));d2=float(d.square().sum())
    p.flatten().index_copy_(0,idx,new);del d,actual
   assert torch.isfinite(new).all()
   bucket='Attention' if '.self_attn.' in n else 'MLP' if '.mlp.' in n else 'Other'
   sub=stats['per_module'].setdefault(bucket,dict(selected=0,nonzero_write=0,actual_delta_sq=0.))
   for target in (stats,sub):
    target['selected']+=idx.numel();target['nonzero_write']+=nz;target['actual_delta_sq']+=a2
   stats['candidate_delta_sq']+=d2;stats['mask_union']+=self.union_counts[n]
   del g,delta,old,new
  self.steps=t;stats['actual_norm']=math.sqrt(stats['actual_delta_sq']);return stats
 def state_dict(self):
  return dict(method=self.method,names=list(self.named),quota=self.quota,steps=self.steps,calibrations=self.calibrations,
   state={n:{k:v.cpu() for k,v in s.items()} for n,s in self.state.items()},english=self.english,
   indices={n:v.cpu() for n,v in self.indices.items()},
   unions={n:torch.from_numpy(np.packbits(v.numpy(),bitorder='little')) for n,v in self.unions.items()})
 def load_state_dict(self,s):
  assert s['method']==self.method and s['names']==list(self.named)
  assert s.get('quota')==self.quota
  self.steps=s['steps'];self.calibrations=s['calibrations'];self.english=s['english']
  for n,p in self.named.items():
   self.state[n]={k:v.to(p.device,dtype=torch.float32) for k,v in s['state'][n].items()}
   if n in s['indices']:self.indices[n]=s['indices'][n].to(p.device)
   self.unions[n]=torch.from_numpy(np.unpackbits(s['unions'][n].numpy(),bitorder='little',count=p.numel()).astype(bool))
   self.union_counts[n]=int(self.unions[n].sum())
