"""Independent end-of-run acceptance. Refuses incomplete or mismatched runs."""
from pathlib import Path
import sys,json,math,collections,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from settings import *
import torch,numpy as np
from safetensors import safe_open
from training_log_chain import verified_segments,logical_records

def audit(method):
 torch.set_num_threads(4)
 acceptance=REPORT/'training_acceptance'/f'{method}.json'
 if acceptance.exists():
  cached=read(acceptance)
  if cached.get('auditor_sha256')==sha(Path(__file__)) and cached.get('log_chain_helper_sha256')==sha(Path(__file__).with_name('training_log_chain.py')):
   assert cached['passed'] and cached['method']==method
   for path,digest in cached['audit_input_sha256'].items():assert sha(path)==digest,('Previously accepted input changed',path)
   checkpoint=Path(cached['final_checkpoint']);assert sha(checkpoint/'RECOVERABLE.json')==cached['checkpoint_receipt_sha256']
   for rel,digest in read(checkpoint/'RECOVERABLE.json')['files'].items():assert sha(checkpoint/rel)==digest
   return cached
 out=H/'models/checkpoint/sw_fullbudget_baselines_20261028'/f'{method}_first'
 done=read(out/'TRAINING_DONE.json');identity=read(out/'RUN_IDENTITY.json');scope=read(out/'PARAMETER_SCOPE.json')
 config=read(REPORT/'TRAINING_CONFIG_FROZEN.json');data=read(REPORT/'DATA_FROZEN.json');budget=read(REPORT/'COMMON_BUDGET_20261028.json')
 assert not identity['preflight'] and identity['method']==method
 assert identity['config_sha256']==sha(REPORT/'TRAINING_CONFIG_FROZEN.json')
 assert done['steps']==42465 and done['counts']['main_input']==15360935 and done['counts']['main_supervised']==5984371
 checkpoint=Path(done['complete_checkpoint']);receipt=read(checkpoint/'RECOVERABLE.json')
 assert receipt['step']==42465
 for rel,digest in receipt['files'].items():assert sha(checkpoint/rel)==digest
 weights=Path(done['final_model'])/'model.safetensors';assert sha(weights)==done['final_weights_sha256']['model.safetensors']
 logs=RUN/'training'/f'{method}_first';steps=[];selected=[];count_path=0.;write_events=0
 phase=collections.Counter();cal_input=cal_sup=0
 segments=verified_segments(logs/'steps.jsonl',done,method,sha(REPORT/'TRAINING_CONFIG_FROZEN.json'))
 for r in logical_records(segments):
  steps.append(r['step']);selected.append(r['selected'])
  assert all(math.isfinite(r[k]) for k in ('main_CE','actual_norm','gradient_norm','lr'))
  assert 0<=r['nonzero_write']<=scope['unique_model_elements'] and r['mask_union']<=scope['trainable_elements']
  if method in ('SSU','HFT15','Element','Protected'):assert r['selected']==budget['common_selected_elements']
  phase.update(r['seconds_by_phase']);count_path+=r['actual_norm'];write_events+=r['nonzero_write']
  if r['calibration']:
   cal_input+=r['calibration']['input_tokens'];cal_sup+=r['calibration']['supervised_tokens']
 assert steps==list(range(1,42466))
 assert math.isclose(count_path,done['counts']['cumulative_actual_path'],rel_tol=1e-12)
 assert write_events==done['counts']['nonzero_write_events']
 assert cal_input==done['counts']['calibration_input'] and cal_sup==done['counts']['calibration_supervised']
 for segment in segments:
  assert list((segment['path'].parent/'tensorboard').glob('events.out.tfevents.*'))
 state=torch.load(checkpoint/'state.pt',map_location='cpu',mmap=True,weights_only=False)
 assert state['protocol_sha256']==sha(REPORT/'TRAINING_CONFIG_FROZEN.json')
 assert state['data_position']==84930 and state['scheduler']==dict(step=42465,next_lr=0.)
 assert {'python','numpy','torch','cuda'}<=state['rng'].keys()
 opt=state['optimizer'];assert opt['steps']==42465 and opt['method']==method
 assert set(opt['state'])==set(scope['shapes'])
 for n,s in opt['state'].items():
  assert set(s)=={'m','v'} and all(t.dtype==torch.float32 and list(t.shape)==scope['shapes'][n] for t in s.values())
  for key,value in s.items():
   flat=value.flatten()
   for start in range(0,flat.numel(),1<<20):
    chunk=flat[start:start+(1<<20)];assert bool(torch.isfinite(chunk).all()),(n,key,'nonfinite optimizer state')
    if key=='v':assert bool((chunk>=0).all()),(n,'negative Adam second moment')
 assert opt['calibrations']==(data['protected_calibration_events'] if method=='Protected' else 0)
 assert all(t.dtype==torch.float32 for t in opt['english'].values())
 frozen_elements=0;total_sq=0.;outside_union_changes=0;per_module=collections.Counter();torch.set_num_threads(4)
 static= torch.load(out/'selection.pt',map_location='cpu',weights_only=False) if method=='SSU' else None
 with safe_open(str(BASE/'model.safetensors'),framework='pt',device='cpu') as base,safe_open(str(weights),framework='pt',device='cpu') as final:
  assert set(base.keys())==set(final.keys())
  for n in base.keys():
   original=base.get_tensor(n);adapted=final.get_tensor(n);assert adapted.dtype==torch.bfloat16 and original.shape==adapted.shape
   if n not in scope['shapes']:
    assert torch.equal(original,adapted),n+' frozen parameter changed';frozen_elements+=original.numel()
   if method=='SSU' and n in scope['shapes']:
    mask=torch.ones(original.shape[1],dtype=torch.bool);mask[static['indices'][n]]=False
    assert torch.equal(original[:,mask],adapted[:,mask]),n+' shielded column changed'
    for k in ('m','v'):
     value=opt['state'][n][k];assert not bool(torch.count_nonzero(value[:,mask])),n+' shielded moment nonzero'
   if n in opt['unions']:
    union=torch.from_numpy(np.unpackbits(opt['unions'][n].numpy(),bitorder='little',count=original.numel()).astype(bool))
    changed=int(torch.count_nonzero(original.flatten()[~union]!=adapted.flatten()[~union]))
    outside_union_changes+=changed
    # Official MoFO applies decay to every candidate, including unselected elements.
    if method!='MoFO':assert changed==0,n+' changed outside support union'
   for start in range(0,original.numel(),1<<20):
    delta=adapted.flatten()[start:start+(1<<20)].float()-original.flatten()[start:start+(1<<20)].float()
    assert bool(torch.isfinite(delta).all());sq=float(delta.double().square().sum());total_sq+=sq
    module='Attention' if '.self_attn.' in n else 'MLP' if '.mlp.' in n else 'Other';per_module[module]+=sq
   del original,adapted
 result=dict(passed=True,method=method,steps=42465,main_input_tokens=15360935,main_supervised_tokens=5984371,
  selected_min=min(selected),selected_max=max(selected),candidate_denominator=1310195712,
  trainable_elements=scope['trainable_elements'],unique_model_elements=scope['unique_model_elements'],
  cumulative_actual_path=count_path,actual_nonzero_write_events=write_events,final_net_displacement=math.sqrt(total_sq),
  module_net_displacements={k:math.sqrt(v) for k,v in per_module.items()},frozen_elements_bitwise_verified=frozen_elements,
  calibration_input_tokens=cal_input,calibration_supervised_tokens=cal_sup,seconds_by_phase=dict(phase),
  changed_elements_outside_adam_support_union=outside_union_changes,
  outside_union_policy='global candidate weight decay permitted' if method=='MoFO' else 'must remain bitwise unchanged',
  final_checkpoint=str(checkpoint),checkpoint_hashes_verified=True,FP32_moments_verified=True,FP32_moments_finite_and_v_nonnegative=True,RNG_scheduler_data_position_verified=True,
  final_weights_sha256=sha(weights),optimizer_parameter_count=sum(t['m'].numel() for t in opt['state'].values()),
  logical_training_log_segments=[dict(path=str(s['path']),first_step=s['first_step'],last_step=s['last_step']) for s in segments],
  scientific_limit='single frozen training order and budget; no best-checkpoint selection',completed_unix=time.time())
 inputs=[out/'TRAINING_DONE.json',out/'RUN_IDENTITY.json',out/'PARAMETER_SCOPE.json',
  REPORT/'TRAINING_CONFIG_FROZEN.json',REPORT/'DATA_FROZEN.json',REPORT/'COMMON_BUDGET_20261028.json']
 inputs += [segment['path'] for segment in segments]
 if method=='SSU':inputs.append(out/'selection.pt')
 if done.get('continuation_manifest'):inputs.append(Path(done['continuation_manifest']))
 result.update(auditor_sha256=sha(Path(__file__)),log_chain_helper_sha256=sha(Path(__file__).with_name('training_log_chain.py')),
  checkpoint_receipt_sha256=sha(checkpoint/'RECOVERABLE.json'),audit_input_sha256={str(path):sha(path) for path in inputs})
 if acceptance.exists():
  archive=REPORT/'training_acceptance_history'/f'{method}_{sha(acceptance)[:16]}.json'
  if not archive.exists():write(archive,read(acceptance))
 write(acceptance,result)
 return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);a=p.parse_args();print(json.dumps(audit(a.method),indent=2))
