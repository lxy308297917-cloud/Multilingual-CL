"""FP32-moment AdamW updates for the two preregistered SW D2 mechanisms."""
import math
import torch


def _candidate(p, g, m, v, step, lr, wd):
    m.mul_(.9).add_(g.float(), alpha=.1)
    v.mul_(.999).addcmul_(g.float(), g.float(), value=.001)
    return -lr * (m / (1-.9**step)) / ((v / (1-.999**step)).sqrt() + 1e-8) - lr*wd*p.float()


def _top_mask(scores, fraction):
    k=math.floor(scores.numel()*fraction)
    if k == 0: return torch.zeros_like(scores,dtype=torch.bool)
    # Stable sort fixes ties by flattened matrix position.
    order=torch.argsort(scores.flatten(),descending=True,stable=True)
    mask=torch.zeros(scores.numel(),dtype=torch.bool,device=scores.device)
    mask[order[:k]]=True
    return mask.reshape(scores.shape)


class ElementwiseProtectedAdamW(torch.optim.Optimizer):
    """Every matrix gets a 15% element budget; moments accrue for all candidates."""
    def __init__(self,named,source_provider=None,protected=True,lr=5e-5,wd=.01,
                 fraction=.15,mask_interval=20,source_interval=200):
        self.named=dict(named)
        if not self.named or not 0 < fraction <= 1: raise ValueError('Invalid candidates/budget')
        super().__init__(list(self.named.values()),dict(lr=lr,weight_decay=wd))
        self.source_provider=source_provider;self.protected=protected
        self.fraction=fraction;self.mask_interval=mask_interval;self.source_interval=source_interval
        self.steps=0;self.masks={};self.sensitivity={};self.calibration_stats=[]
    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None: raise ValueError('Closure not supported')
        lr=self.param_groups[0]['lr'];wd=self.param_groups[0]['weight_decay']
        if not math.isfinite(lr) or lr<0: raise ValueError('Invalid learning rate')
        for name,p in self.named.items():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()): raise FloatingPointError('Missing/nonfinite target gradient: '+name)
        refresh_source=self.protected and (self.steps==0 or self.steps%self.source_interval==0) 
        # Provider computes source gradients without altering target .grad buffers.
        if refresh_source:
            if self.source_provider is None: raise ValueError('Missing source calibration')
            with torch.enable_grad(): source,stats=self.source_provider(self.steps)
            if set(source)!=set(self.named): raise ValueError('Source gradient identity mismatch')
            for name,p in self.named.items():
                g=source[name].float().cpu()
                if g.shape!=p.shape or not bool(torch.isfinite(g).all()): raise FloatingPointError('Invalid source gradient')
                old=self.sensitivity.get(name)
                self.sensitivity[name]=g.square() if old is None else old.mul_(.9).addcmul_(g,g,value=.1)
            self.calibration_stats.append(stats)
        step=self.steps+1
        for name,p in self.named.items():
            state=self.state[p]
            if not state:
                state['m']=torch.zeros_like(p,dtype=torch.float32)
                state['v']=torch.zeros_like(p,dtype=torch.float32)
            m=state['m'];v=state['v']
            delta=_candidate(p,p.grad,m,v,step,lr,wd)
            if self.steps==0 or self.steps%self.mask_interval==0:
                target=m.abs();target=target/(target.mean()+1e-12)
                if self.protected:
                    sens=self.sensitivity[name].to(target.device)
                    sens=sens/(sens.mean()+1e-12)
                    score=target/(1+sens)
                else: score=target
                self.masks[name]=_top_mask(score,self.fraction)
            mask=self.masks[name]
            p[mask]=(p[mask].float()+delta[mask]).to(p.dtype)
        self.steps=step
    def state_dict(self):
        data=super().state_dict()
        data['mechanism']={'names':list(self.named),'steps':self.steps,'protected':self.protected,
            'fraction':self.fraction,'mask_interval':self.mask_interval,'source_interval':self.source_interval,
            'masks':self.masks,'sensitivity':self.sensitivity,'calibration_stats':self.calibration_stats}
        return data
    def load_state_dict(self,data):
        x=data['mechanism']
        for key,value in [('names',list(self.named)),('protected',self.protected),('fraction',self.fraction),
                          ('mask_interval',self.mask_interval),('source_interval',self.source_interval)]:
            if x[key]!=value:raise ValueError('Optimizer identity mismatch: '+key)
        params=list(self.named.values());groups=data['param_groups']
        if len(groups)!=1 or len(groups[0]['params'])!=len(params):raise ValueError('Parameter group mismatch')
        saved={}
        for idx,p in zip(groups[0]['params'],params):
            s=data['state'].get(idx,{})
            if s and (set(s)!={'m','v'} or any(t.shape!=p.shape or t.dtype!=torch.float32 for t in s.values())):raise ValueError('Invalid moment')
            saved[p]={k:t.to(p.device).clone() for k,t in s.items()}
        super().load_state_dict({k:data[k] for k in ('state','param_groups')})
        for p,s in saved.items():self.state[p]=s
        self.steps=x['steps'];self.masks={n:x['masks'][n].to(self.named[n].device) for n in x['masks']}
        self.sensitivity={n:t.cpu() for n,t in x['sensitivity'].items()}
        self.calibration_stats=x['calibration_stats']


class ProjectedRowAdamW(torch.optim.Optimizer):
    """Project the *full* AdamW delta before selecting whole output rows."""
    def __init__(self,named,bases,lr=5e-5,wd=.01,alpha=.5,row_fraction=.5,mask_interval=20):
        self.named=dict(named);self.bases={n:u.float().cpu() for n,u in bases.items()}
        if set(self.named)!=set(self.bases):raise ValueError('Basis/candidate mismatch')
        for n,p in self.named.items():
            u=self.bases[n]
            if p.ndim!=2 or u.ndim!=2 or u.shape[0]!=p.shape[1]:raise ValueError('Invalid basis: '+n)
        super().__init__(list(self.named.values()),dict(lr=lr,weight_decay=wd))
        self.alpha=alpha;self.row_fraction=row_fraction;self.mask_interval=mask_interval
        self.steps=0;self.masks={}
    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('Closure not supported')
        lr=self.param_groups[0]['lr'];wd=self.param_groups[0]['weight_decay']
        for n,p in self.named.items():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('Missing/nonfinite gradient: '+n)
        step=self.steps+1
        for n,p in self.named.items():
            state=self.state[p]
            if not state:state.update(m=torch.zeros_like(p,dtype=torch.float32),v=torch.zeros_like(p,dtype=torch.float32))
            delta=_candidate(p,p.grad,state['m'],state['v'],step,lr,wd)
            if self.alpha:
                u=self.bases[n].to(p.device)
                delta=delta-self.alpha*((delta@u)@u.T)
            if self.steps==0 or self.steps%self.mask_interval==0:
                self.masks[n]=_top_mask(delta.square().mean(1),self.row_fraction)
            mask=self.masks[n]
            p[mask,:]=(p[mask,:].float()+delta[mask,:]).to(p.dtype)
        self.steps=step
    def state_dict(self):
        data=super().state_dict();data['mechanism']={'names':list(self.named),'steps':self.steps,
            'alpha':self.alpha,'row_fraction':self.row_fraction,'mask_interval':self.mask_interval,
            'masks':self.masks,'bases':self.bases}
        return data
    def load_state_dict(self,data):
        x=data['mechanism']
        for key,value in [('names',list(self.named)),('alpha',self.alpha),('row_fraction',self.row_fraction),('mask_interval',self.mask_interval)]:
            if x[key]!=value:raise ValueError('Optimizer identity mismatch: '+key)
        for n,u in self.bases.items():
            if not torch.equal(u,x['bases'][n].to(device='cpu',dtype=torch.float32)):raise ValueError('Basis changed: '+n)
        groups=data['param_groups'];params=list(self.named.values())
        if len(groups)!=1 or len(groups[0]['params'])!=len(params):raise ValueError('Parameter group mismatch')
        saved={}
        for idx,p in zip(groups[0]['params'],params):
            s=data['state'].get(idx,{})
            if s and (set(s)!={'m','v'} or any(t.shape!=p.shape or t.dtype!=torch.float32 for t in s.values())):raise ValueError('Invalid moment')
            saved[p]={k:t.to(p.device).clone() for k,t in s.items()}
        super().load_state_dict({k:data[k] for k in ('state','param_groups')})
        for p,s in saved.items():self.state[p]=s
        self.steps=x['steps'];self.masks={n:x['masks'][n].to(self.named[n].device) for n in x['masks']}
