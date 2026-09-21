"""Layer-budget streaming prototype; not yet a torch.optim/Trainer adapter."""
from streaming_optimizer import StreamingBlockAdamW
from group_selection import select_grouped

class GroupedBlockAdamW(StreamingBlockAdamW):
    def __init__(self,named_parameters,groups,policy='global',layer_weights=None,**kwargs):
        super().__init__(named_parameters,**kwargs)
        if set(groups)!=set(self.params):raise ValueError('Parameter/layer identity mismatch')
        if policy not in ('global','fixed'):raise ValueError('Invalid policy')
        if policy=='global' and layer_weights is not None:raise ValueError('Adaptive allocation must not use positional weights')
        self.settings.update(groups=dict(groups),policy=policy,layer_weights=layer_weights)
        self.last_selection=None
    def select_masks(self,scores):
        c=self.settings
        masks,info=select_grouped(scores,[c['groups'][n] for n in self.params],c['fraction'],c['policy'],c['layer_weights'],positive_only=c['mode']=='utility')
        self.last_selection=info
        return dict(zip(self.params,masks))
    def state_dict(self):
        state=super().state_dict()
        # Only scalar/list accounting; no tensors in selection metadata.
        import copy
        state['last_selection']=copy.deepcopy(self.last_selection)
        state['deferred_refresh']=getattr(self,'deferred_refresh',False)
        return state
    def load_state_dict(self,state):
        super().load_state_dict(state)
        import copy
        self.last_selection=copy.deepcopy(state['last_selection'])
        self.deferred_refresh=state.get('deferred_refresh',False)
