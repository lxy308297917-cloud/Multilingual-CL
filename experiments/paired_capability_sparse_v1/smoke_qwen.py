"""Real Qwen technical check; NOT mastery filtering or an efficacy experiment."""
import argparse,hashlib,json,math,os,random,sys,time,traceback,threading,subprocess
from pathlib import Path
import numpy as np
import torch
from core import SparseProjectedAdamW,calibration_context

H=Path('/data0/users/lixiaoyang')
BASE=H/'models/Qwen2.5-1.5B-Instruct'
RECIPE=H/'reproductions/sw_recipe_fft_20261010/data/aya80_mt20'
PREV=H/'Documents/Codex/2026-10-20/multilingual-continual-learning-github-https-github/work/gsm_bilingual_20261020/FROZEN_PROBE128.json'

def sha(p):
 d=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):d.update(b)
 return d.hexdigest()

def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def fingerprint(model,opt):
 d=hashlib.sha256()
 for n,p in model.named_parameters():d.update(n.encode());d.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 for n,s in opt.state.items():
  for k,v in s.items():d.update((n+k).encode());d.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 for n,v in opt.masks.items():d.update(n.encode());d.update(v.cpu().numpy().tobytes())
 return d.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--audit',required=True);ap.add_argument('--resume-checkpoint');args=ap.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 stop_sampling=threading.Event();physical=[]
 def sample_memory():
  while not stop_sampling.is_set():
   try:physical.append(int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
   except Exception:pass
   stop_sampling.wait(.5)
 sampler=threading.Thread(target=sample_memory,daemon=True);sampler.start()
 try:
  from transformers import AutoModelForCausalLM,AutoTokenizer
  from torch.utils.tensorboard import SummaryWriter
  if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one verified GPU UUID required')
  torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
  audit=json.loads(Path(args.audit).read_text());manifest=json.loads((RECIPE/'manifest.json').read_text())
  if sha(manifest['source_file'])!=manifest['source_sha256']:raise RuntimeError('Training source changed')
  tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
  model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
  model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  named={n:p for n,p in model.named_parameters() if '.layers.' in n and p.ndim==2}
  if len(named)!=196 or sum(p.numel() for p in named.values())!=1310195712:raise RuntimeError('Candidate identity mismatch')
  for n,p in model.named_parameters():p.requires_grad_(n in named)
  indices=json.loads((RECIPE/'indices.json').read_text())[:8]
  long_indices=audit['checks']['longest_training_indices'];wanted=set(indices+long_indices);pool={}
  with open(manifest['source_file']) as f:
   for i,line in enumerate(f):
    if i in wanted:pool[i]=json.loads(line)
  # Two longest samples stress activation memory; normal prefix samples follow.
  batches=[long_indices,indices[:2],indices[2:4],indices[4:6]]
  accepted=[x for x in json.loads(PREV.read_text()) if x['split']=='construction' and x['audit']['decision']=='provisionally_accept' and x['audit'].get('arithmetic_matches_source_gold')]
  # Gold-only short reference CE: an engineering proxy, not validated reasoning CE.
  pair=accepted[0];guards_data=[]
  for lang in ('en','sw'):
   messages=[{'role':'system','content':'Solve the problem. Give the final numeric answer.'},{'role':'user','content':pair['question_'+lang]}]
   prefix=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
   answer=tok(pair['gold']+'<|im_end|>',add_special_tokens=False)['input_ids']
   guards_data.append({'input_ids':prefix+answer,'labels':[-100]*len(prefix)+answer})
  opt=SparseProjectedAdamW(named,bf16_policy='reject');writer=SummaryWriter(str(out/'tensorboard'))
  cfg=dict(scope='Technical stress/resume check only; gold-only CE guards without mastery claims; no dev efficacy or formal scores',candidate_matrices=196,candidate_elements=1310195712,
    base=str(BASE),base_weights_sha256=audit['files'][str(BASE/'model.safetensors')]['sha256'],
    recipe_manifest_sha256=sha(RECIPE/'manifest.json'),main_batches=batches,guard_id=pair['id'],guard_source_sha256=sha(PREV),
    guard_target='numeric gold + EOS, both input languages; no forced boxed prefix',steps=4,
    lr=5e-5,lr_scope='maximum-LR engineering stress; not a formal-schedule training prefix',
    torch=torch.__version__,python=sys.executable,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],script_sha256=sha(__file__),core_sha256=sha(Path(__file__).with_name('core.py')),
    bf16_policy='reject whole rounded weight update on positive residual; retain moments',max_gpu_hours=1.)
  write(out/'CONFIG.json',cfg)
  counts=dict(input=0,supervised=0,calibration_input=0,calibration_supervised=0,generated=0,replay=0)
  times=dict(target=0.,calibration=0.,optimizer=0.,checkpoint=0.,recovery_check=0.)
  def batch(row):
   return {k:torch.tensor([row[k]],device='cuda') for k in ('input_ids','labels')}
  def loss(row):
   x=batch(row);labels=x.pop('labels');logits=model(**x,use_cache=False).logits
   targets=labels[:,1:];valid=targets!=-100
   # Exact supervised-token CE, chunked along positions; no full sequence float logits.
   logits=logits[:,:-1][valid].float();value=torch.nn.functional.cross_entropy(logits,targets[valid])
   return value
  def update(step,count=True):
   torch.cuda.synchronize();tic=time.monotonic();model.train();model.zero_grad(set_to_none=True);target_loss=0.
   for i in batches[step]:
    row=pool[i];value=loss(row);target_loss+=value.item()/2;(value/2).backward()
    if count:counts['input']+=len(row['input_ids']);counts['supervised']+=sum(x!=-100 for x in row['labels'][1:])
   torch.nn.utils.clip_grad_norm_(list(named.values()),1.);torch.cuda.synchronize();times['target']+=time.monotonic()-tic
   tic=time.monotonic();gs=[];guard_losses=[]
   # autograd.grad does not accumulate calibration into target .grad.
   with calibration_context(model):
    for row in guards_data:
     value=loss(row);guard_losses.append(value.item());g=torch.autograd.grad(value,list(named.values()))
     gs.append({n:v.detach().float().cpu() for (n,p),v in zip(named.items(),g)});del g,value
     if count:counts['calibration_input']+=len(row['input_ids']);counts['calibration_supervised']+=sum(x!=-100 for x in row['labels'][1:])
   torch.cuda.synchronize();times['calibration']+=time.monotonic()-tic;tic=time.monotonic()
   stats=opt.step(5e-5,gs);del gs
   torch.cuda.synchronize();times['optimizer']+=time.monotonic()-tic
   stats.update(target_loss=target_loss,guard_losses=guard_losses,global_step=step+1)
   if count:
    for n,v in stats.items():
     if isinstance(v,(float,int)):writer.add_scalar(n,v,step+1)
   return stats
  records=[];resume_start=0
  if args.resume_checkpoint:
   checkpoint=Path(args.resume_checkpoint)
   old_cfg=checkpoint.parent/'CONFIG.json'
   state=torch.load(checkpoint,map_location='cpu',weights_only=False)
   if state['config_sha256']!=sha(old_cfg) or state['next_step']!=2:raise RuntimeError('Invalid recovery checkpoint')
   prior_cfg=json.loads(old_cfg.read_text())
   for key in ('base_weights_sha256','recipe_manifest_sha256','main_batches','guard_id','guard_source_sha256','lr','bf16_policy'):
    if prior_cfg[key]!=cfg[key]:raise RuntimeError('Recovery scientific identity mismatch: '+key)
   def restore(saved):
    with torch.no_grad():
     for n,p in model.named_parameters():p.copy_(saved['model'][n])
    opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda']);random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy'])
   tic=time.monotonic();restore(state);first=update(2,count=False);expected=fingerprint(model,opt)
   expected_model={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
   expected_moments={n:{k:v.cpu().clone() for k,v in m.items()} for n,m in opt.state.items()}
   restore(state);second=update(2,count=False);actual=fingerprint(model,opt)
   diff=[]
   for n,p in model.named_parameters():
    old=expected_model[n];new=p.detach().cpu()
    if not torch.equal(old,new):diff.append({'name':n,'kind':'weight','different':int((old!=new).sum()),'max_abs':float((old.float()-new.float()).abs().max())})
   for n,m in opt.state.items():
    for key,v in m.items():
     old=expected_moments[n][key];new=v.cpu()
     if not torch.equal(old,new):diff.append({'name':n,'kind':key,'different':int((old!=new).sum()),'max_abs':float((old-new).abs().max())})
   write(out/'REPLAY_DIAGNOSTIC.json',{'first':first,'second':second,'fingerprint1':expected,'fingerprint2':actual,'tensor_differences':diff,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG')})
   del expected_model,expected_moments
   if expected!=actual or first!=second:raise RuntimeError('Recovered Qwen replay not exact; inspect REPLAY_DIAGNOSTIC.json')
   restore(state);del state;resume_start=2;times['recovery_check']+=time.monotonic()-tic
   write(out/'RESUME_CHECK.json',{'exact_model_moments_masks':True,'uninterrupted_fingerprint':expected,'resumed_fingerprint':actual,'metrics_exact':True,'compared_step':3,'extra_update_executions':2,'initial_checkpoint':str(checkpoint),'old_failure_preserved':True})
   write(out/'CHECKPOINT.json',{'path':str(checkpoint),'sha256':sha(checkpoint),'step':2,'full_state':True,'reused_without_overwrite':True})
   cfg['recovery_from']=str(checkpoint);cfg['recovery_reason']='reuse existing GPU moment buffers to avoid simultaneous old/new full states';write(out/'CONFIG.json',cfg)
   for ids in batches[:2]:
    for i in ids:counts['input']+=len(pool[i]['input_ids']);counts['supervised']+=sum(x!=-100 for x in pool[i]['labels'][1:])
    counts['calibration_input']+=sum(len(r['input_ids']) for r in guards_data);counts['calibration_supervised']+=sum(sum(v!=-100 for v in r['labels'][1:]) for r in guards_data)
   prior_log=H/'logs/paired_capability_sparse_20261021/technical_v1.log'
   records=[json.loads(line) for line in prior_log.read_text().splitlines() if line.startswith('{')][:2]
  for step in range(resume_start,4):
   if (time.monotonic()-start)>3600:raise RuntimeError('Frozen technical time budget reached')
   stats=update(step);records.append(stats);print(json.dumps(stats),flush=True)
   if step==1:
    tic=time.monotonic();checkpoint=out/'resume.pt'
    torch.save(dict(model={n:p.detach().cpu() for n,p in model.named_parameters()},optimizer=opt.state_dict(),
      next_step=2,scheduler={'technical_constant_lr':5e-5,'step':2},rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy':np.random.get_state()},config_sha256=sha(out/'CONFIG.json')),checkpoint)
    times['checkpoint']+=time.monotonic()-tic;write(out/'CHECKPOINT.json',{'path':str(checkpoint),'sha256':sha(checkpoint),'step':2,'full_state':True})
    tic=time.monotonic();uninterrupted=update(2,count=False);expected=fingerprint(model,opt)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['config_sha256']!=sha(out/'CONFIG.json') or state['next_step']!=2:raise RuntimeError('Resume config mismatch')
    with torch.no_grad():
     for n,p in model.named_parameters():p.copy_(state['model'][n].to(p.device))
    opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);del state
    resumed=update(2,count=False);actual=fingerprint(model,opt)
    if expected!=actual or uninterrupted!=resumed:raise RuntimeError('Real Qwen resume differs from uninterrupted step')
    write(out/'RESUME_CHECK.json',{'exact_model_moments_masks':True,'uninterrupted_fingerprint':expected,'resumed_fingerprint':actual,'metrics_exact':True,'compared_step':3,'extra_update_executions':2})
    # We have already committed step3 after restore; loop must not duplicate it.
    # Restore step2 once more so normal accounting runs step3 exactly once.
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    with torch.no_grad():
     for n,p in model.named_parameters():p.copy_(state['model'][n].to(p.device))
    opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);del state
    times['recovery_check']+=time.monotonic()-tic
  writer.close();write(out/'METRICS.json',records)
  # Check unchanged noncandidate tensors against the original local safetensors.
  from safetensors import safe_open
  with safe_open(str(BASE/'model.safetensors'),framework='pt',device='cpu') as f:
   for n,p in model.named_parameters():
    if n not in named and not torch.equal(p.detach().cpu(),f.get_tensor(n)):raise RuntimeError('Frozen tensor changed: '+n)
  write(out/'DONE.json',{'scope':cfg['scope'],'steps':opt.steps,'counts_logical_path':counts,'extra_recovery_update_executions':2,'extra_recovery_tokens':{'input':2*sum(len(pool[i]['input_ids']) for i in batches[2]),'supervised':2*sum(sum(v!=-100 for v in pool[i]['labels'][1:]) for i in batches[2]),'calibration_input':2*sum(len(r['input_ids']) for r in guards_data),'calibration_supervised':2*sum(sum(v!=-100 for v in r['labels'][1:]) for r in guards_data)},'seconds_by_phase':times,
    'wall_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
    'physical_device_peak_mib_including_display':max(physical) if physical else None,'physical_sampling_interval_seconds':.5,'device_total_bytes':torch.cuda.get_device_properties(0).total_memory,'resume_exact':True,'frozen_noncandidates_unchanged':True,
    'rejected_rounded_updates':sum(s['rejected_rounding'] for s in records),'no_efficacy_conclusion':True})
 except BaseException as e:
  write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-start});raise
 finally:
  stop_sampling.set();sampler.join(timeout=3)
if __name__=='__main__':main()
