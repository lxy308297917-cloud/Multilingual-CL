"""Single-device torch optimizer adapter; full Trainer/GPU release pending."""
import copy
import torch
from grouped_optimizer import GroupedBlockAdamW

class TorchBlockAdamW(torch.optim.Optimizer):
    def __init__(self,named_parameters,groups,source_provider=None,**settings):
        named=dict(named_parameters)
        super().__init__(list(named.values()),dict(lr=settings.get('lr',5e-5)))
        self.engine=GroupedBlockAdamW(named,groups,**settings)
        self.source_provider=source_provider
        self.last_step=None
        self.target_score_grads=None
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        if len(self.param_groups)!=1:raise ValueError('Single parameter group required')
        self.engine.settings['lr']=self.param_groups[0]['lr']
        c=self.engine.settings
        refresh=self.engine.needs_refresh()
        source=None
        if refresh and c['mode']=='utility':
            if self.source_provider is None:raise ValueError('Utility refresh requires source provider')
            source=self.source_provider()
        self.last_step=self.engine.step(source,self.target_score_grads)
        self.target_score_grads=None
        return loss
    def state_dict(self):
        # PyTorch's state_dict returns tensor references; avoid a full FP32 moment clone.
        outer=super().state_dict()
        outer['block_engine']={'settings':copy.deepcopy(self.engine.settings),'steps':self.engine.steps,
                               'names':list(self.engine.params),'state':self.engine.state,
                               'masks':self.engine.masks,'last_selection':copy.deepcopy(self.engine.last_selection),
                               'deferred_refresh':getattr(self.engine,'deferred_refresh',False)}
        return outer
    def load_state_dict(self,state_dict):
        e=state_dict['block_engine'];c=self.engine.settings
        expected={k:v for k,v in c.items() if k!='lr'}
        actual={k:v for k,v in e['settings'].items() if k!='lr'}
        if e['names']!=list(self.engine.params) or expected!=actual:raise ValueError('Optimizer identity mismatch')
        if e['steps']<0:raise ValueError('Invalid saved step')
        if e['steps'] and (set(e['state'])!=set(self.engine.params) or set(e['masks'])!=set(self.engine.params)):
            raise ValueError('Incomplete optimizer state')
        # Keep moments FP32 even when parameter dtype is BF16 (generic torch loader casts).
        restored={};masks={}
        for n,p in self.engine.params.items():
            if n not in e['state']:continue
            s=e['state'][n];mask=e['masks'][n];size=c['block_size']
            if set(s)!={'m','v'} or any(x.shape!=p.shape or x.dtype!=torch.float32 or not torch.isfinite(x).all() for x in s.values()):raise ValueError('Invalid moments')
            if bool((s['v']<0).any()):raise ValueError('Negative second moment')
            if mask.dtype!=torch.bool or mask.shape!=torch.Size([p.shape[0]//size,p.shape[1]//size]):raise ValueError('Invalid mask')
            restored[n]={k:v.to(p.device).clone() for k,v in s.items()}
            masks[n]=mask.to(p.device).clone()
        # The generic state is empty by design; prevent unnoticed extra state.
        if state_dict['state']:raise ValueError('Unexpected generic optimizer state')
        super().load_state_dict({k:state_dict[k] for k in ['state','param_groups']})
        self.engine.settings['lr']=self.param_groups[0]['lr']
        self.engine.steps=e['steps'];self.engine.state=restored;self.engine.masks=masks
        self.engine.last_selection=copy.deepcopy(e['last_selection'])
        self.engine.deferred_refresh=e.get('deferred_refresh',False)
