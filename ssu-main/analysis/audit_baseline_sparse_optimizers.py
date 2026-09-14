"""Numerical CPU audit of the actual project hooks; not a GPU/full-model release."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'training/src'))
from utils.model_utils import _freeze_structured_ssu_2d,_make_scale_hook
from utils.mofo import MoFOAdamW
c=json.loads((R/'configs/ig_baseline_train_v1.json').read_text());torch.set_num_threads(1)
records=[]
for dtype in [torch.bfloat16,torch.float32]:
 for method in ['ssu_column','top20_column']:
  p=torch.nn.Parameter(torch.linspace(.001,.02,64*32).reshape(64,32).to(dtype));before=p.detach().clone()
  if method=='ssu_column':
   count=_freeze_structured_ssu_2d(p,.5,p.shape,SimpleNamespace(input_activations=torch.arange(1,33).float()),axis_preference='column')
  else:
   scale=torch.zeros_like(p);scale[:,:6]=1;p.register_hook(_make_scale_hook(scale));count=64*26
  opt=torch.optim.AdamW([p],lr=c['learning_rate'],betas=(c['adam_beta1'],c['adam_beta2']),eps=c['adam_epsilon'],weight_decay=c['weight_decay'])
  for i in range(10):
   opt.zero_grad();p.sum().backward()
   if i==0:
    frozen=p.grad==0;assert int(frozen.sum())==count
   opt.step()
  changed=p.detach()!=before
  record={'method':method,'dtype':str(dtype),'steps':10,'frozen_elements':count,'frozen_changed_elements':int((changed&frozen).sum()),'eligible_changed_elements':int((changed&~frozen).sum()),'optimizer_state_elements':sum(x.numel() for x in opt.state[p].values() if isinstance(x,torch.Tensor))}
  assert record['eligible_changed_elements']>0
  if dtype==torch.bfloat16:assert record['frozen_changed_elements']==0
  records.append(record)
for tied in [False,True]:
 p=torch.nn.Parameter(torch.full((1000,),.001));before=p.detach().clone();opt=MoFOAdamW([p],lr=c['learning_rate'],weight_decay=c['weight_decay'],update_fraction=.15)
 p.grad=torch.ones_like(p) if tied else torch.arange(1,1001).float()/1000;opt.step();changed=int((p.detach()!=before).sum());assert changed==(1000 if tied else 150)
 records.append({'method':'mofo15','dtype':'float32','equal_momentum_ties':tied,'elements':1000,'changed_elements':changed})
out=R.parent/'experiments/ig_baseline_suite_v1/sparse_optimizer_cpu_audit.json'
r={'scope':'CPU small-tensor numerical audit of actual hooks and optimizer; full-model GPU smoke still required before method release','config_sha256':hashlib.sha256((R/'configs/ig_baseline_train_v1.json').read_bytes()).hexdigest(),'records':records,'conclusions':['BF16 tested fixed-mask elements stayed bitwise unchanged at the configured lr and weight decay.','FP32 gradient masks do not prevent AdamW decay changes; do not generalize BF16 observations to other precision/settings.','MoFO threshold ties may select more than the nominal fraction, including all elements in the equal-momentum case.']}
out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False,indent=2))
