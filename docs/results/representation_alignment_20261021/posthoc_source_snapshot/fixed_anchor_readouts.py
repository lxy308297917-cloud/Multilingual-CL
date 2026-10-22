"""CPU-only predefined fixed-English-anchor and English-drift diagnostics.
No extra forward pass, model selection, checkpoint selection, or training change.
"""
from pathlib import Path
import sys,json,csv,hashlib
import torch
import torch.nn.functional as F
ROOT=Path('/data0/users/lixiaoyang/Documents/Codex/2026-10-21/referenced-chatgpt-conversation-this-is-an');CODE=ROOT/'outputs/Multilingual-CL-alignment-recovery-v3/experiments/representation_alignment_v1';OUT=Path('/data0/users/lixiaoyang/models/checkpoint/representation_alignment_20261021');sys.path.insert(0,str(CODE));from common import retrieval

def main():
 torch.set_num_threads(4);allrows=[];models=['Base','ExistingFullDense']+['align-'+m for m in 'ABCD']
 for name in models:
  if not(OUT/'diagnostics'/name/'DONE.json').exists():raise RuntimeError('Require all frozen endpoint diagnostics, missing '+name)
 with torch.inference_mode():
  for split,n in [('dev',500),('diagnostic',1000)]:
   base=torch.load(OUT/'diagnostics/Base'/(split+'_representations.pt'),map_location='cpu',mmap=True,weights_only=True)
   for name in models:
    current=torch.load(OUT/'diagnostics'/name/(split+'_representations.pt'),map_location='cpu',mmap=True,weights_only=True)
    for method in ['mean','last','weighted']:
     assert base['en'][method].shape==current['en'][method].shape==current['sw'][method].shape==(29,n,1536)
     for layer in range(29):
      en0=base['en'][method][layer];en=current['en'][method][layer];sw=current['sw'][method][layer];fixed=retrieval(en0,sw);drift=float(((en-en0).norm(dim=1)/en0.norm(dim=1).clamp_min(1e-12)).mean());same_cos=float(F.cosine_similarity(en,en0,dim=1).mean())
      if name=='Base':assert drift==0
      allrows.append({'model':name,'split':split,'pooling':method,'HF_hidden_state_index':layer,'fixed_EN_to_current_SW_p1':fixed['en_to_sw_p1'],'current_SW_to_fixed_EN_p1':fixed['sw_to_en_p1'],'fixed_mexa_strict':fixed['mexa_strict'],'fixed_positive_negative_gap':fixed['positive_negative_gap'],'current_EN_to_Base_EN_mean_relative_L2':drift,'current_EN_Base_EN_mean_cosine':same_cos})
 dest=ROOT/'outputs/representation_alignment_report';p=dest/'FIXED_ANCHOR_AND_EN_DRIFT.csv'
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 (dest/'FIXED_ANCHOR_READOUT_DONE.json').write_text(json.dumps({'rows':len(allrows),'all_models':models,'all_layers_and_pooling_reported':True,'no_new_GPU_forward':True,'cuda_initialized':torch.cuda.is_initialized(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'diagnostic_only_not_capability_or_upgrade_gate':True},indent=2)+'\n')
if __name__=='__main__':main()
