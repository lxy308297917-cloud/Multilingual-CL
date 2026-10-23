"""Original unprotected FP32 AdamW/momentum masks with exact static per-matrix quotas."""
import torch,math,numpy as np

class BudgetSparseAdamW:
 def __init__(self,named,quota,mask_interval=20,wd=.01):
  self.named={n:p for n,p in named.items() if n in quota};self.quota=dict(quota);assert set(self.named)==set(quota)
  self.mask_interval=mask_interval;self.wd=wd;self.steps=0;self.masks={};self.unions={};self.indices={};self.union_counts={};self.state={n:{'m':torch.zeros_like(p,dtype=torch.float32),'v':torch.zeros_like(p,dtype=torch.float32)} for n,p in self.named.items()};self.K=sum(quota.values())
  assert all(p.is_contiguous() and 0<quota[n]<=p.numel() for n,p in self.named.items())
 @torch.no_grad()
 def step(self,lr):
  assert math.isfinite(lr) and lr>=0;t=self.steps+1;refresh=self.steps%self.mask_interval==0;candidate_sq=actual_sq=0.;nonzero=0;per_module={};changed_masks=0
  for n,p in self.named.items():
   assert p.grad is not None
   m,v=self.state[n]['m'],self.state[n]['v'];g=p.grad.float();m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
   # Exact original _candidate equation and operation order.
   delta=-lr*(m/(1-.9**t))/((v/(1-.999**t)).sqrt()+1e-8)-lr*self.wd*p.float()
   assert bool(torch.isfinite(delta).all()),'nonfinite candidate '+n
   if refresh:
    target=m.abs();score=target/(target.mean()+1e-12);order=torch.argsort(score.flatten(),descending=True,stable=True);idx=order[:self.quota[n]].sort().values
    mask=torch.zeros(p.numel(),dtype=torch.bool);mask[idx.cpu()]=True;mask=mask.view_as(p);changed_masks+=int(torch.count_nonzero(mask!=self.masks[n])) if n in self.masks else self.quota[n];self.masks[n]=mask;self.indices[n]=idx;self.unions[n]=self.unions.get(n,torch.zeros_like(mask))|mask;self.union_counts[n]=int(torch.count_nonzero(self.unions[n]));del order,target,score,mask
   idx=self.indices[n];old=p.flatten().index_select(0,idx);d=delta.flatten().index_select(0,idx);new=(old.float()+d).to(p.dtype);actual=new.float()-old.float();assert bool(torch.isfinite(new).all())
   a2=float(actual.square().sum());d2=float(d.square().sum());nz=int(torch.count_nonzero(new!=old));actual_sq+=a2;candidate_sq+=d2;nonzero+=nz;bucket='Attention' if '.self_attn.' in n else 'MLP';r=per_module.setdefault(bucket,{'actual_delta_sq':0.,'candidate_delta_sq':0.,'nonzero_write':0,'selected':0});r['actual_delta_sq']+=a2;r['candidate_delta_sq']+=d2;r['nonzero_write']+=nz;r['selected']+=self.quota[n]
   # Only these selected coordinates can be assigned, including all decay.
   p.view(-1).index_copy_(0,idx,new);del g,delta,old,d,new,actual
  self.steps=t
  assert sum(self.quota.values())==self.K
  return {'step':t,'lr':lr,'selected':self.K,'mask_refresh':refresh,'mask_changed_coordinates':changed_masks if refresh else 0,'mask_union':sum(self.union_counts.values()),'nonzero_write':nonzero,'candidate_norm':math.sqrt(candidate_sq),'actual_norm':math.sqrt(actual_sq),'per_module':per_module}
 def metadata(self):return {'names':list(self.named),'shapes':{n:tuple(p.shape) for n,p in self.named.items()},'quota':self.quota,'mask_interval':self.mask_interval,'wd':self.wd,'K':self.K}
 def state_dict(self):
  def packed(mapping):return {n:torch.from_numpy(np.packbits(v.numpy().reshape(-1),bitorder='little')) for n,v in mapping.items()}
  return {'metadata':self.metadata(),'steps':self.steps,'state':{n:{k:v.cpu() for k,v in s.items()} for n,s in self.state.items()},'masks_packed':packed(self.masks),'unions_packed':packed(self.unions),'mask_encoding':'np.packbits little; exactnumel frommetadata; index reconstructed in ascending flat order'}
 @torch.no_grad()
 def load_state_dict(self,data):
  assert data['metadata']==self.metadata();assert set(data['state'])==set(self.named);self.steps=data['steps']
  for n,ss in data['state'].items():
   for k in ('m','v'):
    v=ss[k];assert v.shape==self.named[n].shape and v.dtype==torch.float32;self.state[n][k].copy_(v)
  for key,target in [('masks_packed',self.masks),('unions_packed',self.unions)]:
   target.clear()
   for n,v in data[key].items():target[n]=torch.from_numpy(np.unpackbits(v.numpy(),bitorder='little',count=self.named[n].numel()).copy()).bool().view_as(self.named[n])
  self.indices={n:m.flatten().nonzero().flatten().to(self.named[n].device) for n,m in self.masks.items()};self.union_counts={n:int(torch.count_nonzero(m)) for n,m in self.unions.items()}
  if self.steps:
   assert set(self.masks)==set(self.named) and all(int(m.sum())==self.quota[n] for n,m in self.masks.items())
