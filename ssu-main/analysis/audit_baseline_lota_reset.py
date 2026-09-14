"""Exercise actual LoTA calibration/reset/masking on a small Qwen model on CPU."""
import sys,json
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'training/src'))
from utils.model_utils import lota_calibrate_mask,lota_prepare_sparse_training
torch.set_num_threads(1);torch.manual_seed(42)
model=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True)).to(torch.bfloat16)
base={n:p.detach().clone() for n,p in model.named_parameters()}
ids=torch.randint(0,64,(2,16));batch={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':ids.clone()}
calib=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01)
state=lota_calibrate_mask(model,[batch]*100,calib,sparsity=.9,calibration_steps=100,device='cpu',skip_embeddings_and_head=True,grad_accum_steps=1,use_amp=False,enable_gradient_checkpointing=False,verbose=False)
calibration_changed=sum(int((p.detach()!=base[n]).sum()) for n,p in model.named_parameters());assert calibration_changed>0
lota_prepare_sparse_training(model,state,verbose=False)
assert all(torch.equal(p,base[n]) for n,p in model.named_parameters())
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=.01);assert not opt.state
model(**batch).loss.backward();opt.step()
frozen_changes=0;selected_changes=0
for n,p in model.named_parameters():
 mask=state.mask[n];changes=p.detach()!=base[n];frozen_changes+=int((changes&~mask).sum());selected_changes+=int((changes&mask).sum())
assert frozen_changes==0 and selected_changes>0
r={'scope':'CPU small Qwen model only; full-model GPU smoke required before LoTA release','calibration_steps':100,'calibration_changed_elements':calibration_changed,'reset_to_base_bitwise':True,'fresh_optimizer_initial_state_empty':True,'frozen_changed_elements_after_training_step':frozen_changes,'selected_changed_elements_after_training_step':selected_changes,'selected_mask_elements':state.trainable_params,'total_elements':state.total_params}
(R.parent/'experiments/ig_baseline_suite_v1/lota_reset_cpu_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
