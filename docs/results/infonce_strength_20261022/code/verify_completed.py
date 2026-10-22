"""Independent server artifact and exact trajectory-budget verification, without GPU work."""
from settings import *
import torch,math

def main():
 start=time.time();out=NEWTRAIN/'D1_seed42';done=json.loads((out/'TRAINING_DONE.json').read_text());config=json.loads((out/'CONFIG.json').read_text());cal=json.loads((RUN/'CALIBRATION_DONE.json').read_text());assert done['alpha']==config['alpha']==cal['alpha'];assert config['code_sha256']==sha(REPORT/'code/train_d1.py');assert config['strength_protocol_sha256']==sha(REPORT/'PROTOCOL_FROZEN.json');a=rows(out/'STEPS.jsonl');b=rows(TRAIN/'comparison_v3/D/STEPS.jsonl');assert len(a)==len(b)==4293
 for x,y in zip(a,b):assert x['step']==y['step'] and x['lr']==y['lr'] and x['counts']==y['counts']
 assert config['resume'] is None and not config['smoke'];assert config['model_sha256']==cal['base_model_sha256']==sha(BASE/'model.safetensors');assert config['main']==json.loads((OLD/'data/main_manifest.json').read_text());assert config['teacher_sha256']==cal['teacher_sha256']==sha(TRAIN/'EN_TEACHER.pt')
 checks={}
 for step in (2146,4293):
  cp=out/('checkpoint_%06d'%step);receipt=json.loads((cp/'RECOVERABLE.json').read_text())
  for rel,value in receipt['files'].items():assert sha(cp/rel)==value,(cp,rel)
  s=torch.load(cp/'state.pt',map_location='cpu',mmap=True,weights_only=False);assert s['step']==s['scheduler']['last_epoch']==step;assert s['alpha']==cal['alpha'] and s['strength_protocol_sha256']==sha(REPORT/'PROTOCOL_FROZEN.json');assert s['counts']==a[step-1]['counts'];assert s['aux_cursor']==step//4*8;assert len(s['optimizer']['state'])==338
  for st in s['optimizer']['state'].values():assert st['exp_avg'].dtype==st['exp_avg_sq'].dtype==torch.bfloat16 and int(st['step'])==step
  checks[str(step)]={'receipt_sha256':sha(cp/'RECOVERABLE.json'),'model_sha256':sha(cp/'model/model.safetensors'),'state_sha256':sha(cp/'state.pt'),'all338_moments_dtype_step_verified':True};del s
 ed=json.loads((RUN/'evaluation/D1/DONE.json').read_text());assert ed['rows']==512 and ed['scored_rows']==511 and ed['predictions_sha256']==sha(RUN/'evaluation/D1/PREDICTIONS.jsonl');assert ed['model_sha256']==checks['4293']['model_sha256'];assert done['steps']==4293 and done['counts']['main_input']==1536351 and done['counts']['main_supervised']==599469
 proof={'passed':True,'checks':checks,'all4293_LR_main_aux_input_supervised_padded_counts_identical_to_D0':True,'fixed_alpha':cal['alpha'],'rows_generated':512,'rows_scored':511,'counts':done['counts'],'raw_predictions_sha256':ed['predictions_sha256'],'model_sha256':ed['model_sha256'],'training_done_sha256':sha(out/'TRAINING_DONE.json'),'source_code_sha256':sha(REPORT/'code/train_d1.py'),'server_raw_artifacts_checked':True,'not_full_historical_retraining':True,'CPU_verification_seconds':time.time()-start};write(REPORT/'COMPLETION_VERIFIED.json',proof);print('D1 SERVER COMPLETION VERIFIED')
if __name__=='__main__':main()
