from settings import *
import torch,copy,math,importlib.util
from optimizer import BaselineAdamW
torch.manual_seed(42);torch.set_num_threads(2)
checks=[]
for method in METHODS:
 w=torch.nn.Parameter(torch.randn(10,12));named={'model.layers.0.mlp.up_proj.weight':w};name=next(iter(named))
 static={name:torch.arange(20)} if method=='SSU' else None
 opt=BaselineAdamW(named,method,static)
 if method=='Protected':opt.calibrate({name:torch.randn_like(w)})
 before=w.detach().clone();w.grad=torch.randn_like(w);stats=opt.step(.003)
 if method in ('Element','Protected','SSU'):
  mask=torch.zeros(w.numel(),dtype=torch.bool);mask[opt.indices[name]]=True
  assert torch.equal(w.flatten()[~mask],before.flatten()[~mask])
 if method in ('Dense','HFT15'):
  reference=torch.nn.Parameter(before.clone());reference.grad=w.grad.clone();a=torch.optim.AdamW([reference],lr=.003,weight_decay=.01,foreach=False);a.step()
  torch.testing.assert_close(w,reference,atol=2e-7,rtol=1e-6)
 s=copy.deepcopy(opt.state_dict());w2=torch.nn.Parameter(w.detach().clone());o2=BaselineAdamW({name:w2},method,static);o2.load_state_dict(s)
 for _ in range(3):
  g=torch.randn_like(w);w.grad=g.clone();w2.grad=g.clone();opt.step(.003);o2.step(.003)
  assert torch.equal(w,w2)
  assert torch.equal(opt.state[name]['m'],o2.state[name]['m'])
 checks.append(method+':resume and update checks')
# Official MoFO reference, with only removed-private-symbol compatibility shim.
import torch.optim.optimizer as torch_optimizer
torch_optimizer._dispatch_sqrt=math.sqrt
spec=importlib.util.spec_from_file_location('official_mofo_reference',REPORT/'sources/mofo.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
net=torch.nn.Linear(12,10,bias=False);w=torch.nn.Parameter(net.weight.detach().clone())
opt=BaselineAdamW({'weight':w},'MoFO');ref=m.AdamW_MoFO(net,weight_decay=.01,lr=.003,fraction=.15)
for _ in range(4):
 g=torch.randn_like(w);net.weight.grad=g.clone();w.grad=g.clone();opt.step(.003);ref.step()
 torch.testing.assert_close(w,net.weight,rtol=1e-6,atol=2e-7)
# Zero LR advances moments exactly once and writes nothing.
w=torch.nn.Parameter(torch.randn(10,12,dtype=torch.bfloat16));o=BaselineAdamW({'weight':w},'Element');old=w.clone();w.grad=torch.ones_like(w);s=o.step(0.)
assert torch.equal(old,w) and s['nonzero_write']==0 and o.steps==1
write(REPORT/'OPTIMIZER_ACCEPTANCE.json',dict(passed=True,checks=checks,official_MoFO_FP32_reference_parity=True,masked_outside_unchanged=True,
 zero_lr=True,full_state_resume=True,scope='CPU numerical acceptance; real Qwen longest-batch/save-resume preflights separate'))
# The revised element budget must be honored, rather than silently using floor15%.
for method in ('Element','Protected'):
 w=torch.nn.Parameter(torch.randn(10,12));name='weight';o=BaselineAdamW({name:w},method,quota={name:24})
 if method=='Protected':o.calibrate({name:torch.randn_like(w)})
 w.grad=torch.randn_like(w);stats=o.step(.003);assert stats['selected']==24
 copy_opt=BaselineAdamW({name:torch.nn.Parameter(w.detach().clone())},method,quota={name:24});copy_opt.load_state_dict(copy.deepcopy(o.state_dict()))
b=read(REPORT/'COMMON_BUDGET_20261028.json');shapes=read(REPORT/'DATA_FROZEN.json')['shapes']
assert sum(b['element_quotas'].values())==sum(shapes[n][0]*v for n,v in b['ssu_columns'].items())==sum(math.prod(shapes[n]) for n in b['hft_selected_names'])==196608000
r=read(REPORT/'OPTIMIZER_ACCEPTANCE.json');r.update(exact_common_budget_CPU_checked=True,common_selected_elements=196608000,configuration_sha256=sha(REPORT/'TRAINING_CONFIG_FROZEN.json'));write(REPORT/'OPTIMIZER_ACCEPTANCE.json',r)
print('Baseline optimizer checks pass')
