"""Independent, task-balanced source gradients; no optimizer state access."""
import contextlib
import time
import torch


def per_example_loss(logits, labels):
    target=labels[:,1:]
    counts=(target!=-100).sum(1)
    if not bool((counts>0).all()):raise ValueError('Every source example needs supervised tokens')
    loss=torch.nn.functional.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').view_as(target)
    values=loss.sum(1)/counts
    if not bool(torch.isfinite(values).all()):raise FloatingPointError('Nonfinite source loss')
    return values


def source_gradients(model, named_parameters, task_batches, device, autocast_dtype=None):
    """Equal weight per task, then example; return FP32 gradients on CPU.

    Calibration is evaluated at current weights in eval mode. autograd.grad avoids
    touching accumulated target .grad buffers. Module training flags and torch RNG
    are restored even on error. Input lists must contain complete labelled batches.
    """
    params=dict(named_parameters);device=torch.device(device)
    if not params or not task_batches or any(not batches for batches in task_batches.values()):
        raise ValueError('Empty calibration or parameter set')
    if any(not p.requires_grad for p in params.values()):raise ValueError('Selected parameter is frozen')
    counts={task:sum(int(b['labels'].shape[0]) for b in batches) for task,batches in task_batches.items()}
    if any(n==0 for n in counts.values()):raise ValueError('Empty task')
    modes={m:m.training for m in model.modules()}
    devices=sorted({p.device.index for p in model.parameters() if p.device.type=='cuda'})
    gradients={n:torch.zeros(p.shape,dtype=torch.float32,device='cpu') for n,p in params.items()}
    stats={'examples':counts,'input_tokens':0,'supervised_tokens':0,'task_mean_loss':{},'wall_seconds':None}
    start=time.perf_counter()
    try:
        with torch.random.fork_rng(devices=devices),torch.enable_grad():
            model.eval()
            for task,batches in task_batches.items():
                total=0.
                for raw in batches:
                    batch={k:v.to(device) for k,v in raw.items()}
                    labels=batch.pop('labels')
                    amp=torch.autocast(device.type,dtype=autocast_dtype) if autocast_dtype is not None else contextlib.nullcontext()
                    with amp:
                        output=model(**batch,use_cache=False)
                        values=per_example_loss(output.logits,labels)
                        loss=values.sum()/(counts[task]*len(task_batches))
                    grads=torch.autograd.grad(loss,tuple(params.values()),allow_unused=False)
                    for (name,_),grad in zip(params.items(),grads):
                        if not bool(torch.isfinite(grad).all()):raise FloatingPointError('Nonfinite source gradient')
                        gradients[name].add_(grad.detach().to(device='cpu',dtype=torch.float32))
                    total+=float(values.detach().sum())
                    stats['input_tokens']+=int(batch['attention_mask'].sum())
                    stats['supervised_tokens']+=int((labels[:,1:]!=-100).sum())
                    del grads,output,values,loss,batch,labels
                stats['task_mean_loss'][task]=total/counts[task]
    finally:
        for module,training in modes.items():module.training=training
    stats['wall_seconds']=time.perf_counter()-start
    return gradients,stats
