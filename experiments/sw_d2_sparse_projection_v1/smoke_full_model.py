import argparse,json,time,os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file
from method_optimizers import ElementwiseProtectedAdamW,ProjectedRowAdamW
from activation_bases import LAYERS
import sys
sys.path.insert(0,'/root/autodl-fs/ssu-project/experiments/sw_update_utility_v1/prototype')
from source_gradients import per_example_loss,source_gradients
p=argparse.ArgumentParser();p.add_argument('method',choices=['element_protected','element_unprotected','edge_projected_rows','edge_rows_no_projection']);a=p.parse_args()
started=time.time();model=AutoModelForCausalLM.from_pretrained('/root/models/Qwen2.5-1.5B-Instruct',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
all_named=dict(model.named_parameters());candidate={n:p for n,p in all_named.items() if '.layers.' in n and p.ndim==2}
assert len(candidate)==196
cal=[json.loads(s) for s in Path('/root/autodl-tmp/sw_update_utility_v1/calibration_candidates.jsonl').open() if json.loads(s)['split']=='scoring']
x=min((r for r in cal if len(r['input_ids'])<=128),key=lambda r:len(r['input_ids']))
if os.environ.get('SMOKE_FULL_LENGTH')=='1':
 x=json.loads(next(Path('/root/autodl-tmp/d2_bilingual_v1/sw/train.jsonl').open()))
 assert len(x['input_ids'])==2048
def batch(r):return {'input_ids':torch.tensor([r['input_ids']],device='cuda'), 'attention_mask':torch.ones(1,len(r['input_ids']),device='cuda',dtype=torch.long),'labels':torch.tensor([r['labels']],device='cuda')}
if a.method.startswith('element'):
 for n,p in all_named.items():p.requires_grad_(n in candidate)
 def provider(step):
  source_rows=([r for task,n in [('instruction',11),('reading',11),('math',10)] for r in [z for z in cal if z['task']==task][:n]] if os.environ.get('SMOKE_FULL_LENGTH')=='1' else [x])
  grads,stats=source_gradients(model,candidate,{'english':[batch(z) for z in source_rows]},torch.device('cuda'),torch.bfloat16)
  return grads,stats
 opt=ElementwiseProtectedAdamW(candidate,source_provider=provider if a.method=='element_protected' else None,protected=a.method=='element_protected')
else:
 edge={n:p for n,p in candidate.items() if any(f'.layers.{i}.' in n for i in LAYERS)}
 for n,p in all_named.items():p.requires_grad_(n in edge)
 bases={n+'.weight':u for n,u in load_file('/root/autodl-tmp/sw_d2_sparse_projection_v1/edge_bases_rank32.safetensors').items()}
 opt=ProjectedRowAdamW(edge,bases,alpha=.5 if a.method=='edge_projected_rows' else 0.)
model.train();b=batch(x);labels=b.pop('labels');out=model(**b,use_cache=False)
loss=per_example_loss(out.logits,labels).mean();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
first=next(iter(opt.named));before=opt.named[first].detach().cpu().clone();opt.step();after=opt.named[first].detach().cpu()
mask=opt.masks[first].cpu();mask=mask if mask.ndim==2 else mask[:,None].expand_as(before)
assert torch.equal(before[~mask],after[~mask]);assert bool((before[mask]!=after[mask]).any())
result={'method':a.method,'status':'passed','loss':float(loss.detach()),'seconds':time.time()-started,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'candidate_parameters':sum(t.numel() for t in opt.named.values()),
        'first_matrix_selected_fraction':float(mask.float().mean()),'first_matrix_inactive_exact':True}
path=Path(__file__).parent/('SMOKE_'+a.method+('_2048' if os.environ.get('SMOKE_FULL_LENGTH')=='1' else '')+'.json');path.write_text(json.dumps(result,indent=2));print(json.dumps(result))
