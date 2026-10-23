"""Necessary numerical regression: match old unprotected AdamW, exact quota/support and lossless recovery."""
from settings import *
from optimizer import BudgetSparseAdamW
import torch,importlib.util

def main():
 torch.set_num_threads(4);torch.manual_seed(42);spec=importlib.util.spec_from_file_location('original_sparse_optim',OLD.parent/'sw_d2_sparse_projection_v1/method_optimizers.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);p=torch.nn.Parameter(torch.randn(4,8).bfloat16());q=torch.nn.Parameter(p.detach().clone());name='model.layers.0.self_attn.q_proj.weight';a=BudgetSparseAdamW({name:p},{name:16});b=mod.ElementwiseProtectedAdamW({name:q},protected=False,fraction=.5,mask_interval=20);refreshes=[]
 for step in range(43):
  g=(torch.ones_like(p) if step==0 else torch.randn_like(p));p.grad=g.clone();q.grad=g.clone();before=p.detach().clone();lr=5e-5*(step+1)/43;r=a.step(lr);b.param_groups[0]['lr']=lr;b.step();assert torch.equal(p,q),step;assert torch.equal(a.state[name]['m'],b.state[q]['m']) and torch.equal(a.state[name]['v'],b.state[q]['v']);assert torch.equal(a.masks[name],b.masks[name]);assert int(torch.count_nonzero(a.masks[name]))==16;assert not bool(((p!=before)&~a.masks[name]).any());assert r['selected']==16
  if r['mask_refresh']:refreshes.append(step+1)
 assert refreshes==[1,21,41]
 state=a.state_dict();rpar=torch.nn.Parameter(p.detach().clone());restored=BudgetSparseAdamW({name:rpar},{name:16});restored.load_state_dict(state);g=torch.randn_like(p);p.grad=g.clone();rpar.grad=g.clone();a.step(5e-5);restored.step(5e-5);assert torch.equal(p,rpar) and torch.equal(a.state[name]['m'],restored.state[name]['m']) and torch.equal(a.unions[name],restored.unions[name])
 budget=json.loads((REPORT/'BUDGET_FROZEN.json').read_text());assert all(sum(g['quota'].values())==budget['K'] for g in budget['groups'].values())
 # All first-step scores equal: stable flat tie picks first16.
 t=torch.nn.Parameter(torch.zeros(4,8,dtype=torch.bfloat16));o=BudgetSparseAdamW({name:t},{name:16});t.grad=torch.ones_like(t);o.step(0.);assert o.masks[name].flatten()[:16].all() and not o.masks[name].flatten()[16:].any()
 write(REPORT/'OPTIMIZER_TESTS.json',{'passed':True,'old_unprotected_BF16_weights_FP32_moments_bitwise_equal_updates':43,'mask_refreshes':refreshes,'offmask_write_zero_each_update':True,'packbits_resume_nextupdate_bitwise_equal':True,'stable_flat_tie_order':True,'all3_exact_K':budget['K'],'original_optimizer_sha256':sha(OLD.parent/'sw_d2_sparse_projection_v1/method_optimizers.py'),'new_optimizer_sha256':sha(pathlib.Path(__file__).parent/'optimizer.py')});print('OPTIMIZER NUMERICAL TESTS PASSED')
if __name__=='__main__':main()
