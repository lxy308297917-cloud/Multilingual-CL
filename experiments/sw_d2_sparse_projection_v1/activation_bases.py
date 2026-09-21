"""Deterministic randomized covariance range finder for protected input subspaces."""
import hashlib,json,torch
from pathlib import Path

LAYERS={1,2,3,4,5,23,24,25,26,27}  # historical zero-based layer labels

def edge_matrices(model):
    return {n:m for n,m in model.named_modules() if any(f'.layers.{i}.' in n for i in LAYERS)
            and isinstance(m,torch.nn.Linear) and ('.self_attn.' in n or '.mlp.' in n)}

@torch.no_grad()
def estimate_bases(model,rows,collate,rank=32,seed=42,tokens_per_sample=8):
    modules=edge_matrices(model)
    if len(modules)!=70:raise ValueError(f'Expected 70 edge matrices, got {len(modules)}')
    cpu=torch.Generator(device='cpu').manual_seed(seed)
    random={n:torch.randn(m.in_features,rank,generator=cpu,dtype=torch.float32)/rank**.5 for n,m in modules.items()}
    accum={n:torch.zeros_like(v) for n,v in random.items()};counts={n:0 for n in modules}
    handles=[]
    for name,module in modules.items():
        def hook(mod,inputs,output,name=name):
            x=inputs[0].detach().reshape(-1,inputs[0].shape[-1])
            if x.shape[0]>tokens_per_sample:
                positions=torch.linspace(0,x.shape[0]-1,tokens_per_sample,device=x.device).long()
                x=x.index_select(0,positions)
            x=x.float();omega=random[name].to(x.device)
            accum[name].add_((x.T@(x@omega)).cpu())
            counts[name]+=x.shape[0]
        handles.append(module.register_forward_hook(hook))
    training=model.training;model.eval()
    try:
        for row in rows:
            batch=collate([row]);device=next(model.parameters()).device
            with torch.autocast('cuda',dtype=torch.bfloat16):
                model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device),use_cache=False)
    finally:
        for h in handles:h.remove()
        model.train(training)
    bases={}
    for n,y in accum.items():
        if counts[n]==0:raise ValueError('No activations: '+n)
        q,_=torch.linalg.qr(y,mode='reduced')
        if q.shape[1]!=rank or not torch.isfinite(q).all():raise ValueError('Invalid basis: '+n)
        bases[n]=q.contiguous()
    return bases,counts

def basis_identity(bases):
    return {n:hashlib.sha256(u.contiguous().numpy().tobytes()).hexdigest() for n,u in bases.items()}
