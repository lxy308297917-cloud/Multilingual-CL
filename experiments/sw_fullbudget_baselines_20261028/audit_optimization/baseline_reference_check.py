"""Independent CPU reference audit; records differences rather than hiding them."""
from pathlib import Path
import sys,torch,math,json,importlib.util,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import REPORT as BASE_REPORT,write,sha
from optimizer import BaselineAdamW
REPORT=BASE_REPORT/'audit_optimization_20261028';torch.set_num_threads(2)
import torch.optim.optimizer as optimizer_module
optimizer_module._dispatch_sqrt=math.sqrt
spec=importlib.util.spec_from_file_location('saved_official_mofo',BASE_REPORT/'sources/mofo.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
report={}
for dtype in (torch.bfloat16,torch.float32):
 torch.manual_seed(1234);model=torch.nn.Linear(37,23,bias=False,dtype=dtype)
 p=torch.nn.Parameter(model.weight.detach().clone());current=BaselineAdamW({'weight':p},'MoFO')
 ref=official.AdamW_MoFO(model,weight_decay=.01,lr=5e-5,fraction=.15)
 max_weight=max_m=max_v=0.;mask_different=weight_different=0
 for i in range(40):
  g=torch.ones_like(p) if i==0 else torch.randn_like(p);p.grad=g.clone();model.weight.grad=g.clone()
  current.step(5e-5);ref.step();s=ref.state[model.weight]
  threshold=torch.topk(s['m'].abs().flatten(),math.floor(p.numel()*.15)).values.min()
  mask=s['m'].abs().flatten()>=threshold;current_mask=torch.zeros(p.numel(),dtype=torch.bool);current_mask[current.indices['weight']]=True
  mask_different+=int(not torch.equal(mask,current_mask));weight_different+=int(not torch.equal(p,model.weight))
  max_weight=max(max_weight,float((p.float()-model.weight.float()).abs().max()))
  max_m=max(max_m,float((current.state['weight']['m']-s['m']).abs().max()))
  max_v=max(max_v,float((current.state['weight']['v']-s['v']).abs().max()))
 report[str(dtype)]=dict(updates=40,mask_mismatch_steps=mask_different,weight_bit_mismatch_steps=weight_different,
  maximum_absolute_weight_difference=max_weight,maximum_absolute_m_difference=max_m,maximum_absolute_v_difference=max_v,
  reference='saved official MoFO, only removed private sqrt symbol shim',
  explicit_operation_differences=['current mul/add moment vs reference lerp','current FP32 update expression vs reference sqrt/addcdiv arithmetic'])
formula=[]
for method in ('Dense','HFT15'):
 for dtype in (torch.bfloat16,torch.float32):
  torch.manual_seed(1234);p=torch.nn.Parameter(torch.randn(23,37,dtype=dtype));ref=p.detach().clone()
  opt=BaselineAdamW({'model.layers.0.mlp.up_proj.weight':p},method);m=torch.zeros_like(ref,dtype=torch.float32);v=m.clone()
  for i in range(20):
   g=torch.randn_like(p);p.grad=g;lr=0. if i==0 else 5e-5
   opt.step(lr);g=g.float();m=.9*m+.1*g;v=.999*v+.001*g*g
   # Algebraic reference intentionally reassociates operations; test operation-order parity separately.
   ref=(ref.float()-lr*(m/(1-.9**(i+1)))/((v/(1-.999**(i+1))).sqrt()+1e-8)-lr*.01*ref.float()).to(dtype)
   # Algebraic split addition/subtraction may round differently in FP32.
  formula.append(dict(method=method,dtype=str(dtype),updates=20,maximum_absolute_weight_difference=float((p.float()-ref.float()).abs().max()),
   bitwise_equal=torch.equal(p,ref),scope='independent algebraic FP32 expression, not default torch.optim BF16 operator parity'))
ordered=[]
for method in ('Dense','HFT15'):
 for dtype in (torch.bfloat16,torch.float32):
  torch.manual_seed(1234);p=torch.nn.Parameter(torch.randn(23,37,dtype=dtype));ref=p.detach().clone()
  name='model.layers.0.mlp.up_proj.weight'
  opt=BaselineAdamW({name:p},method);m=torch.zeros_like(ref,dtype=torch.float32);v=m.clone()
  for i in range(20):
   g=torch.randn_like(p);p.grad=g.clone();lr=0. if i==0 else 5e-5
   opt.step(lr);gg=g.float()
   m=torch.add(torch.mul(m,.9),gg,alpha=.1)
   v=torch.addcmul(torch.mul(v,.999),gg,gg,value=.001)
   old=ref.float()
   d=-lr*(m/(1-.9**(i+1)))/((v/(1-.999**(i+1))).sqrt()+1e-8)
   d=d-lr*.01*old
   ref=(old+d).to(dtype)
   assert torch.equal(p,ref),(method,dtype,i,'weight')
   assert torch.equal(opt.state[name]['m'],m),(method,dtype,i,'m')
   assert torch.equal(opt.state[name]['v'],v),(method,dtype,i,'v')
  ordered.append(dict(method=method,dtype=str(dtype),updates=20,weights_and_moments_bitwise_equal=True,
   scope='independent functional recurrence preserving FP32 operation order; small CPU matrix, not full model'))
write(REPORT/'BASELINE_REFERENCE_AUDIT.json' ,dict(official_MoFO=report,Dense_HFT_explicit_formula=formula,Dense_HFT_operation_order=ordered,
 official_source_sha256=sha(BASE_REPORT/'sources/mofo.py'),frozen_optimizer_sha256=sha(BASE_REPORT/'code/optimizer.py'),
 no_real_model_or_CUDA_claim=True,checked_unix=time.time()))
print(json.dumps(dict(official_MoFO=report,Dense_HFT=formula),indent=2))
