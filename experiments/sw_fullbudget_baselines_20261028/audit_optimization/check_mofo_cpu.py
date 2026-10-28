"""Same-gradient identity against the frozen optimizer, including serialization."""
from pathlib import Path
import torch,copy,json,math,sys,tempfile,time
from mofo_deferred import DeferredMoFO,BaselineAdamW
ROOT=Path(__file__).resolve().parents[4]
REPORT=ROOT/'outputs/sw_fullbudget_baselines_20261028/audit_optimization_20261028'
torch.set_num_threads(2);torch.manual_seed(42);checks=[]
for dtype in (torch.bfloat16,torch.float32):
 names=['model.layers.0.self_attn.q_proj.weight','model.layers.1.mlp.up_proj.weight']
 a={n:torch.nn.Parameter(torch.randn(23,37,dtype=dtype)) for n in names};b={n:torch.nn.Parameter(p.detach().clone()) for n,p in a.items()}
 old=BaselineAdamW(a,'MoFO');new=DeferredMoFO(b)
 for step in range(40):
  for n in names:
   g=torch.ones_like(a[n]) if step==0 else torch.zeros_like(a[n]) if step==20 else torch.randn_like(a[n])
   a[n].grad=g.clone();b[n].grad=g.clone()
  lr=0. if step in (0,19,20) else 5e-5
  x,y=old.step(lr),new.step(lr)
  assert x==y,(dtype,step,'statistics differ',x,y)
  for n in names:
   assert torch.equal(a[n],b[n]),(dtype,step,n,'weights')
   assert torch.equal(old.indices[n],new.indices[n])
   for k in ('m','v'):assert torch.equal(old.state[n][k],new.state[n][k])
   assert torch.equal(old.unions[n],new.unions[n])
  if step==21:
   with tempfile.TemporaryDirectory(prefix='mofo-serialization-') as tmp:
    path=Path(tmp)/'state.pt';torch.save(new.state_dict(),path)
    restored=DeferredMoFO(b);restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=False));new=restored
 checks.append(dict(dtype=str(dtype),updates=40,weights_moments_masks_statistics_bitwise=True,zero_LR_steps=[1,20,21],all_equal_threshold_at_first_step=True,state_roundtrip_at_step=22))
(REPORT/'MOFO_CPU_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,checks=checks,scope='CPU only, small matrices, same-input-gradient identity against frozen optimizer; no CUDA or real-model equivalence claim',checked_unix=time.time()),indent=2)+'\n')
print('CPU:40 updates per dtype, masks/weights/moments/all statistics and state roundtrip exactly match.')
