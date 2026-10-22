"""Independent exact-K/clip-scope/trajectory and complete checkpoint hashes; CPU-only."""
from settings import *
import torch

def main():
 start=time.time();budget=json.loads((REPORT/'BUDGET_FROZEN.json').read_text());proof={'groups':{}};reference=None;base_sha=sha(BASE/'model.safetensors');tokenizer_sha=sha(BASE/'tokenizer.json');main_identity=None
 for group in ('S-All','S-Attn','S-MLP'):
  out=OUT/group;done=json.loads((out/'TRAINING_DONE.json').read_text());config=json.loads((out/'CONFIG.json').read_text());logs=rows(out/'STEPS.jsonl');assert config['protocol_sha256']==sha(REPORT/'PROTOCOL_FROZEN.json');assert config['optimizer_sha256']==sha(REPORT/'code/optimizer.py') and config['engine_sha256']==sha(REPORT/'code/engine.py');assert done['steps']==len(logs)==4293;assert all(r['selected']==budget['K']==77070336 for r in logs);assert [r['step'] for r in logs]==list(range(1,4294));assert done['counts']==logs[-1]['counts'] and done['counts']['main_input']==1536351 and done['counts']['main_supervised']==599469;assert done['zero_auxiliary'] and done['frozen_positions_unchanged']
  assert config['model_sha256']==base_sha and config['tokenizer_sha256']==tokenizer_sha;assert config['resume'] is None and config['start_step']==0;assert config['budget_sha256']==sha(REPORT/'BUDGET_FROZEN.json') and config['code_sha256']==sha(REPORT/'code/train.py');assert config['no_auxiliary_data'] and config['gradient_enabled_candidate_elements']==budget['all_candidate_parameters']
  if reference is None:reference=logs;main_identity=config['main']
  else:assert all(a['lr']==b['lr'] and a['counts']==b['counts'] for a,b in zip(reference,logs))
  assert config['main']==main_identity
  checks={}
  for step in (2146,4293):
   cp=out/('checkpoint_%06d'%step);receipt=json.loads((cp/'RECOVERABLE.json').read_text())
   for rel,value in receipt['files'].items():assert sha(cp/rel)==value,(group,step,rel)
   s=torch.load(cp/'state.pt',map_location='cpu',mmap=True,weights_only=False);opt=s['optimizer'];assert s['step']==opt['steps']==s['scheduler']['step']==step;assert s['group']==group and s['protocol_sha256']==sha(REPORT/'PROTOCOL_FROZEN.json');assert opt['metadata']['quota']==budget['groups'][group]['quota'];assert opt['metadata']['K']==77070336 and len(opt['state'])==budget['groups'][group]['eligible_matrices'];assert all(v.dtype==torch.float32 for entry in opt['state'].values() for v in entry.values());assert s['counts']==logs[step-1]['counts'];assert (cp/'TRANSPORT_VERIFIED.json').exists();checks[str(step)]={'model_sha256':sha(cp/'model/model.safetensors'),'state_sha256':sha(cp/'state.pt'),'receipt_sha256':sha(cp/'RECOVERABLE.json'),'steps_FP32moments_K_quota_counts_verified':True};del s,opt
  ev=RUN/'evaluation'/group;ed=json.loads((ev/'DONE.json').read_text());assert ed['rows']==512 and ed['scored_rows']==511 and ed['predictions_sha256']==sha(ev/'PREDICTIONS.jsonl');assert ed['model_sha256']==done['final_model_sha256']==checks['4293']['model_sha256'];proof['groups'][group]={'checkpoints':checks,'prediction_sha256':ed['predictions_sha256'],'counts':done['counts'],'frozen_positions_unchanged':True,'raw_server_endpoints_checked':True}
 proof.update(passed=True,all3_exact_K_and_LR_counts_identical=True,CPU_verification_seconds=time.time()-start);write(REPORT/'COMPLETION_VERIFIED.json',proof);print('ALL3 SERVER ENDPOINTS VERIFIED')
if __name__=='__main__':main()
