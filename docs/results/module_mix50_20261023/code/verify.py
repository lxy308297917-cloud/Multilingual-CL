"""Raw-server endpoint/state verification; not a substitute for downstream results."""
from settings import *
import torch,math

def main():
 start=time.time();torch.set_num_threads(4);assert (RUN/'SERIES_DONE.json').exists();protocol=json.loads((REPORT/'PROTOCOL_FROZEN.json').read_text());frozen=json.loads((REPORT/'SOURCE_FROZEN.json').read_text())
 for n,h in frozen['code'].items():assert sha(REPORT/'code'/n)==h,('Frozen code modified',n)
 assert sha(REPORT/'PROTOCOL_FROZEN.json')==frozen['protocol_sha256'];assert budget_seconds()[0]<=16*3600;models=json.loads((RUN/'MODELS_FROZEN.json').read_text());budget=json.loads((REPORT/'BUDGET_FROZEN.json').read_text());checked={}
 for name,j in JOBS.items():
  folder=OUT/name;config=json.loads((folder/'CONFIG.json').read_text());done=json.loads((folder/'TRAINING_DONE.json').read_text());steps=rows(folder/'STEPS.jsonl');assert len(steps)==4293 and [r['step'] for r in steps]==list(range(1,4294));order=json.loads((RUN/f"order_seed{j['seed']}.json").read_text());assert [r['source_block'] for r in steps]==order;assert done['counts']['main_input']==1536351 and done['counts']['main_supervised']==599469 and done['steps']==4293 and done['K']==77070336;assert config['order_sha256']==protocol['order_sha256'][str(j['seed'])]
  for i,r in enumerate(steps):
   peak=5e-5;lr=(peak*i/2124 if i<2124 else peak*.5*(1+math.cos(math.pi*(i-2124)/(42465-2124))))*j['multiplier'];assert r['lr']==lr and r['selected']==77070336;assert r['mask_refresh']==(i%20==0);assert {m:v['selected'] for m,v in r['per_module'].items()}==protocol['module_quota'][j['group']];assert math.isfinite(r['actual_norm'])
  assert math.isclose(sum(r['actual_norm'] for r in steps),done['cumulative_path_length'],abs_tol=1e-10);states={}
  for step in (2146,4293):
   cp=folder/f'checkpoint_{step:06d}';receipt=json.loads((cp/'RECOVERABLE.json').read_text())
   for rel,h in receipt['files'].items():assert sha(cp/rel)==h,(name,step,rel)
   state=torch.load(cp/'state.pt',map_location='cpu',mmap=True,weights_only=False);assert state['step']==state['optimizer']['steps']==step;assert state['protocol_sha256']==sha(REPORT/'PROTOCOL_FROZEN.json');assert state['optimizer']['metadata']['quota']==budget['groups'][j['group']]['quota'];assert all(v['m'].dtype==v['v'].dtype==torch.float32 for v in state['optimizer']['state'].values());assert state['scheduler']['multiplier']==j['multiplier'];assert set(state['rng'])=={'python','numpy','torch','cuda'};assert json.loads((cp/'TRANSPORT_VERIFIED.json').read_text())['actual_model_optimizer_scheduler_RNG_loader_executed'];states[str(step)]={'receipt_sha256':sha(cp/'RECOVERABLE.json'),'state_sha256':receipt['files']['state.pt'],'model_sha256':receipt['files']['model/model.safetensors']};del state
  assert done['final_model_sha256']==states['4293']['model_sha256'];checked[name]={'checkpoints':states,'counts':done['counts'],'cumulative_path_length':done['cumulative_path_length'],'final_base_delta_L2':done['final_base_delta_L2'],'module_path':done['module_path_length'],'nonzero_write_events':done['cumulative_nonzero_write_events'],'final_modules':done['final_modules'],'peak_physical_mib':done['physical_peak_mib_observed'],'order_sha256':config['order_sha256']}
 # Frozen row IDs/scorers/model hashes for all newly generated outputs.
 evaluated={}
 for split,names,manifest in [('old',list(JOBS),'OLD_FROZEN.json'),('confirmation',protocol['confirmation_models'],'CONFIRMATION_FROZEN.json'),('prompt',protocol['prompt_models'],'PROMPT_FROZEN.json')]:
  meta=json.loads((RUN/manifest).read_text());expected=rows(meta['data_path']);expected=sorted(expected,key=lambda r:('sw_reading','en_reading','translation','en_instruction').index(r['study_partition']));assert sha(meta['data_path'])==meta['data_sha256']
  for n in names:
   d=RUN/'evaluation'/split/n;done=json.loads((d/'DONE.json').read_text());cfg=json.loads((d/'CONFIG.json').read_text());rr=rows(d/'PREDICTIONS.jsonl');assert len(rr)==len(expected)==done['rows'];assert [r['id'] for r in rr]==[r['id'] for r in expected];assert sha(d/'PREDICTIONS.jsonl')==done['predictions_sha256'];assert cfg['scorer_hashes']==protocol['scorers'];expected_hash=models[n].get('sha256') or checked[n]['checkpoints']['4293']['model_sha256'];assert done['model_sha256']==cfg['model_sha256']==expected_hash;evaluated[split+'/'+n]={'rows':len(rr),'counts':done['counts'],'prediction_sha256':done['predictions_sha256'],'wall_seconds':done['wall_seconds']}
 write(REPORT/'COMPLETION_VERIFIED.json',{'passed':True,'new_scientific_runs':checked,'evaluated':evaluated,'total_GPU_context_hours':budget_seconds()[0]/3600,'GPU_ceiling_hours':16,'raw_state_hashes_FP32_RNG_scheduler_and_transport_verified':True,'all_steps_LR_K_order_module_quota_verified':True,'source_freeze_verified':True,'primary_confirmation_available':protocol['primary_confirmation_available'],'CPU_verification_seconds':time.time()-start});print('ALL4 RAW CHECKPOINTS AND NEW ENDPOINTS VERIFIED',flush=True)
if __name__=='__main__':main()
