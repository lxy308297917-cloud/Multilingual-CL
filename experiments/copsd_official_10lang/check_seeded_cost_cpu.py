from pathlib import Path
import argparse,ast,json
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from peft import LoraConfig,get_peft_model
from transformers import Qwen3Config,Qwen3ForCausalLM,set_seed
from diagnose_official_component_clip import extract
from cost_instrumentation import CostTracker

parser=argparse.ArgumentParser();parser.add_argument('--upstream',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);cli=parser.parse_args()
torch.set_num_threads(2)
config=Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=32)
set_seed(7);base=Qwen3ForCausalLM(config).eval();base_state={k:v.detach().clone() for k,v in base.state_dict().items()};inputs=torch.tensor([[1,2,3]])
with torch.no_grad():expected=base(inputs,use_cache=False).logits

def trial(prior,early):
 set_seed(prior)
 if early:set_seed(42)
 model=Qwen3ForCausalLM(config);model.load_state_dict(base_state)
 model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'])).eval()
 params={k:v.detach().clone() for k,v in model.named_parameters() if v.requires_grad}
 set_seed(42)
 with torch.no_grad():assert torch.equal(model(inputs,use_cache=False).logits,expected)
 return params
same=lambda a,b:a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
assert not same(trial(101,False),trial(202,False))
assert same(trial(101,True),trial(202,True))
fn=extract(cli.upstream);fn.decorator_list=[];ns={'torch':torch,'F':F};exec(compile(ast.Module(body=[fn],type_ignores=[]),'pinned_loss','exec'),ns);loss_fn=ns['generalized_jsd_loss']
s=torch.tensor([[[.01,.4,.3,.29],[.25,.25,.25,.25]]],dtype=torch.float64).log().requires_grad_(True);t=torch.tensor([[[.9,.05,.03,.02],[.25,.25,.25,.25]]],dtype=torch.float64).log();labels=torch.tensor([[0,-100]])
kwargs=dict(student_logits=s,teacher_logits=t,labels=labels,beta=0,token_clip=.05)
tracker=CostTracker();tracker.trainer=SimpleNamespace(state=SimpleNamespace(global_step=3))
rng=torch.get_rng_state().clone();plain=loss_fn(**kwargs);g1,=torch.autograd.grad(plain,s)
observed=tracker.observe_loss(loss_fn)(**kwargs);g2,=torch.autograd.grad(observed,s)
assert torch.equal(plain,observed) and torch.equal(g1,g2) and torch.equal(rng,torch.get_rng_state())
assert tracker.records==[{'step_before_update':3,'valid_response_positions':1,'masked_response_positions':1}]
report={'scope':'Tiny synthetic CPU Qwen3/PEFT initializer and exact pinned loss; not full-model GPU replay','late_seed_does_not_control_initial_LoRA':True,'early_seed_controls_initial_LoRA_across_prior_rng_states':True,'initial_zero_B_adapter_policy_equals_same_Base':True,'cost_observer_loss_equal':True,'cost_observer_gradient_equal':True,'cost_observer_rng_unchanged':True,'exact_mask_counts_verified':True,'full_model_GPU_initialization_verified':False}
cli.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
