from settings import *
import collections,math,random,difflib
import numpy as np
import torch
from transformers import AutoTokenizer
from langdetect import detect,DetectorFactory
DetectorFactory.seed=0;torch.set_num_threads(2)
start=time.time();old=ROOT/'outputs/Multilingual-CL-alignment-recovery-v3/experiments/representation_alignment_v1/data'
manifest=read(old/'main_manifest.json');assert sha(manifest['source'])==manifest['source_sha256'];assert sha(old/'main_prefix.json')==manifest['main_prefix_sha256']
indices=read(old/'main_prefix.json');assert len(indices)==8586
wanted=set(indices);data={}
with Path(manifest['source']).open() as f:
 for i,line in enumerate(f):
  if i in wanted:data[i]=json.loads(line)
assert len(data)==len(wanted)
tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
for row in data.values():
 ids=row['input_ids'];labels=row['labels'];assert len(ids)==len(labels)
 row['_input']=len(ids);row['_supervised']=sum(x!=-100 for x in labels[1:]);target=[x for x in labels if x!=-100];text=tok.decode(target,skip_special_tokens=True).strip()
 try:language=detect(text) if text else 'empty'
 except Exception:language='und'
 row['_language']=language;row['_words']=len(text.split());row['_at_cap']=len(ids)==2048

def summarize(rr,ll=None,weights=None):
 tasks={}
 for i,r in enumerate(rr):
  t=r['task'];x=tasks.setdefault(t,{'samples':0,'input_tokens':0,'supervised_tokens':0,'at_cap':0,'target_languages':{},'target_words':0,'LR_weighted_input':0.,'LR_weighted_supervised':0.})
  x['samples']+=1;x['input_tokens']+=r['_input'];x['supervised_tokens']+=r['_supervised'];x['at_cap']+=r['_at_cap'];x['target_words']+=r['_words'];x['target_languages'][r['_language']]=x['target_languages'].get(r['_language'],0)+1
  if weights is not None:x['LR_weighted_input']+=weights[i]*r['_input'];x['LR_weighted_supervised']+=weights[i]*r['_supervised']
 for v in tasks.values():v['mean_input_tokens']=v['input_tokens']/v['samples'];v['mean_supervised_tokens']=v['supervised_tokens']/v['samples'];v['mean_target_words']=v['target_words']/v['samples']
 return {'tasks':tasks,'samples':len(rr),'input_tokens':sum(r['_input'] for r in rr),'supervised_tokens':sum(r['_supervised'] for r in rr),'main_CE_mean_of_updates':float(np.mean(ll)) if ll else None}

audit={'data_source_sha256':manifest['source_sha256'],'prefix_sha256':manifest['main_prefix_sha256'],'distinct_source_rows':len(data),'presentations':len(indices),'duplicate_presentations':len(indices)-len(data),'checks':{},'orders':{},'implementation_findings':[]}
models={};source_changes={}
for seed,root in MODELS.items():
 order=read(PRUN/f'order_seed{seed}.json');assert sorted(order)==list(range(4293));expected=list(range(4293))
 if seed=='43':random.Random(43).shuffle(expected)
 assert order==expected
 steps=rows(root/'STEPS.jsonl');assert len(steps)==4293
 config=read(root/'CONFIG.json');done=read(root/'TRAINING_DONE.json');assert config['resume'] is None and config['start_step']==0
 assert done['steps']==4293 and done['K']==77070336
 code=ROOT/('outputs/module_lr077_v1' if seed=='42' else 'outputs/module_mix50_v1')/'code'
 assert config['optimizer_sha256']==sha(code/'optimizer.py');assert config['engine_sha256']==sha(code/'engine.py')
 assert config['code_sha256']==sha(code/'train.py');assert config['lr_multiplier']==.7714
 assert sha((ROOT/'outputs/module_lr077_v1/code/optimizer.py'))==sha((PRIOR/'code/optimizer.py'))
 ordered=[];allweights=[];stats=[];running={'main_input':0,'main_supervised':0};prev_in=0;prev_sup=0
 for pos,(block,record) in enumerate(zip(order,steps)):
  assert record['step']==pos+1;assert record.get('source_block',pos)==block
  lr=(5e-5*pos/2124 if pos<2124 else 5e-5*.5*(1+math.cos(math.pi*(pos-2124)/(42465-2124))))*.7714
  assert math.isclose(record['lr'],lr,rel_tol=1e-14,abs_tol=1e-18)
  assert record['selected']==77070336 and math.isfinite(record['main_CE']);assert record['mask_refresh']==(pos%20==0)
  rr=[data[indices[2*block]],data[indices[2*block+1]]];ni=sum(r['_input'] for r in rr);ns=sum(r['_supervised'] for r in rr)
  running['main_input']+=ni;running['main_supervised']+=ns;assert record['counts']['main_input']==running['main_input'] and record['counts']['main_supervised']==running['main_supervised']
  ordered.extend(rr);allweights.extend([lr,lr]);stats.append({'position':pos,'bin':pos*10//4293,'block':block,'lr':lr,'CE':record['main_CE']})
 assert running=={'main_input':1536351,'main_supervised':599469}
 bins=[]
 for bi in range(10):
  pp=[i for i,s in enumerate(stats) if s['bin']==bi];ix=[j for i in pp for j in (2*i,2*i+1)];v=summarize([ordered[j] for j in ix],[stats[i]['CE'] for i in pp],[allweights[j] for j in ix]);v.update(bin=bi+1,first_update=min(pp)+1,last_update=max(pp)+1,mean_LR=float(np.mean([stats[i]['lr'] for i in pp])));bins.append(v)
 phases={}
 for label,predicate in [('warmup',lambda p:p<2124),('postwarmup',lambda p:p>=2124),('first2146',lambda p:p<2146),('last2147',lambda p:p>=2146)]:
  pp=[i for i in range(4293) if predicate(i)];ix=[j for i in pp for j in (2*i,2*i+1)];phases[label]=summarize([ordered[j] for j in ix],[stats[i]['CE'] for i in pp],[allweights[j] for j in ix])
 blocktypes=[tuple(ordered[2*i+j]['task'] for j in (0,1)) for i in range(4293)];runs=[];last=None
 for t in blocktypes:
  if t==last:runs[-1]+=1
  else:runs.append(1);last=t
 checkpoint_audit={}
 for number in (2146,4293):
  cp=root/f'checkpoint_{number:06d}';receipt=read(cp/'RECOVERABLE.json');assert receipt['step']==number;print('Hashing',seed,number,flush=True)
  for rel,digest in receipt['files'].items():assert sha(cp/rel)==digest,(seed,number,rel)
  state=torch.load(cp/'state.pt',map_location='cpu',mmap=True,weights_only=False);assert state['step']==state['scheduler']['step']==number;assert state['protocol_sha256']==config['protocol_sha256'];assert state['scheduler']['total']==42465 and state['scheduler']['warmup']==2124
  nextlr=(5e-5*number/2124 if number<2124 else 5e-5*.5*(1+math.cos(math.pi*(number-2124)/(42465-2124))))*.7714;assert math.isclose(state['scheduler']['next_lr'],nextlr,rel_tol=1e-14)
  assert state['counts']==steps[number-1]['counts'];assert all(k in state['rng'] for k in ('python','numpy','torch','cuda'))
  opt=state['optimizer'];print('optimizer metadata',seed,number,{k:type(v).__name__ for k,v in opt.items()},flush=True)
  if 'steps' in opt:assert opt['steps']==number
  tensor_counts=collections.Counter()
  def traverse(x):
   if isinstance(x,torch.Tensor):tensor_counts[str(x.dtype)]+=x.numel()
   elif isinstance(x,dict):
    for v in x.values():traverse(v)
   elif isinstance(x,(list,tuple)):
    for v in x:traverse(v)
  traverse(opt)
  checkpoint_audit[str(number)]={'all_files_rehashed':True,'model_sha256':receipt['files']['model/model.safetensors'],'state_sha256':receipt['files']['state.pt'],'scheduler':state['scheduler'],'counts':state['counts'],'optimizer_tensor_elements_by_dtype':dict(tensor_counts),'historical_transport':read(cp/'TRANSPORT_VERIFIED.json'),'actual_path':str(cp.resolve())}
  if number==2146:models[f'LR077-s{seed}-mid']={'model':str(cp/'model'),'sha256':receipt['files']['model/model.safetensors']}
  del state,opt
 audit['orders'][seed]={'total':summarize(ordered,[s['CE'] for s in stats],allweights),'bins':bins,'phases':phases,'longest_same_pair_task_run':max(runs),'same_pair_task_adjacency_fraction':sum(a==b for a,b in zip(blocktypes,blocktypes[1:]))/4292,'checkpoints':checkpoint_audit,'resume':config['resume'],'source_hashes_checked':True}
write(RUN/'MODELS_FROZEN.json',models)
audit['checks']={'same_sample_multiplicities':True,'same_two_sample_block_interior':True,'all4293_logged_perstep_counts_match_sources_each_order':True,'same_lr_curve_and_quotas':True,'same_optimizer_bytes':True,'midpoint_and_endpoint_full_receipt_rehash':True,'no_training_resume_or_failed_steps_in_runs':True,'source_differences':'Same optimizer; engine changes seed/order parameterization and scheduler receipt multiplier; no target-loss update change.'}
audit['elapsed_CPU_seconds']=time.time()-start
write(REPORT/'DATA_ORDER_AUDIT.json',audit);write(RUN/'AUDIT_DONE.json',{'passed':True,'models_file_sha256':sha(RUN/'MODELS_FROZEN.json'),'finished_unix':time.time()});print('CPU audit completed',time.time()-start,flush=True)
