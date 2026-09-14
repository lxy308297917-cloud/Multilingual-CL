"""Fixed, model-path-only baseline evaluation controller. Never trains models."""
import argparse, csv, fcntl, hashlib, json, os, runpy, shutil, statistics, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
R=Path(__file__).resolve().parents[1]
from baseline_eval_integrity import aggregate, require_scores, unique_completed

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def filemap(paths):return {str(p):sha(p) for p in sorted(set(paths)) if p.is_file()}
def model_identity(model,c):
 model=Path(model).resolve()
 if not (model/'config.json').is_file():raise ValueError('需要完整 Hugging Face checkpoint: '+str(model))
 config=json.loads((model/'config.json').read_text())
 base=json.loads((Path(c['base_model'])/'config.json').read_text())
 for k in ['model_type','vocab_size','hidden_size','num_hidden_layers']:
  if config[k]!=base[k]:raise ValueError('本协议限定同架构 Qwen2.5-1.5B: '+k)
 files=list(model.glob('*.safetensors'))+list(model.glob('*.json'))+list(model.glob('*.model'))+list(model.glob('*.txt'))
 if not list(model.glob('*.safetensors')):raise ValueError('缺少 safetensors 完整权重')
 # Serializer versions can rewrite merges or move chat_template into .jinja.
 # Token IDs must still agree; evaluation always loads the fixed Base tokenizer.
 effective=Path(c['base_model'])
 if (model/'tokenizer.json').exists():
  local=json.loads((model/'tokenizer.json').read_text());fixed=json.loads((effective/'tokenizer.json').read_text())
  if local['model']['vocab']!=fixed['model']['vocab'] or local.get('added_tokens')!=fixed.get('added_tokens'):
   raise ValueError('token ID mapping differs from fixed Base')
 if (model/'vocab.json').exists() and json.loads((model/'vocab.json').read_text())!=json.loads((effective/'vocab.json').read_text()):raise ValueError('vocabulary differs from Base')
 files+=list(model.glob('*.jinja'))
 return {'path':str(model),'files':{p.name:sha(p) for p in sorted(files)}}

def protocol(c,cp):
 if c.get('lighteval_source') and not (Path(c['lighteval_source'])/'lighteval/__init__.py').is_file():raise FileNotFoundError('Missing bundled LightEval source')
 code=list((R/'evaluation').rglob('*.py'))+[R/'scripts'/n for n in ['baseline_eval.sh','evaluate_cl.py','run_baseline_eval.py','baseline_eval_worker.py','baseline_eval_integrity.py','eval_plnd_downstream.py']]
 code+=list(Path(c.get('lighteval_source', str(R.parent/'lighteval_latest/src'))).rglob('*.py'))
 code+=[R/'analysis/lape_qwen_preexperiment.py']
 inherited=Path(c['evaluation_data_manifest']) if c.get('evaluation_data_manifest') else R.parent/'experiments/ig_baseline_repro_v1/manifest.json'
 inputs=json.loads(inherited.read_text())['evaluation_inputs_inherited']
 if c.get('evaluation_data_manifest'):
  for p,h in inputs.items():
   if not Path(p).is_file() or sha(p)!=h:raise ValueError('Rebuilt evaluation data changed: '+p)
 paths=[Path(p) for p in inputs]+[Path(c['ifeval_data']),Path(c['gsm8k_data'])]
 for root in [Path(c['ppl_data_root'])]:paths+=list(root.rglob('*.parquet'))+list(root.rglob('*.arrow'))
 missing=[str(p) for p in paths if not p.is_file()]
 if missing:raise FileNotFoundError(str(missing))
 environment={}
 for role in ['training_python','evaluation_python']:
  command=[c[role],'-c',"import importlib.metadata as m,json,platform; print(json.dumps({'python':platform.python_version(),'packages':{d.metadata['Name']:d.version for d in m.distributions() if d.metadata['Name']}}))"]
  environment[role]=json.loads(subprocess.check_output(command,text=True))
 return {'config_sha256':sha(cp),'code':filemap(code),'data':filemap(paths),'environment':environment,'tokenizer':filemap(list(Path(c['base_model']).glob('*.json'))+list(Path(c['base_model']).glob('*.jinja'))+list(Path(c['base_model']).glob('*.txt')))}

def jobs(c,smoke):
 result=[]
 for t in c['tasks']:
  for seed in c['generation_seeds'] if t['generative'] else [c['classification_seed']]:result.append(dict(t,seed=seed,kind='lighteval'))
 for task in ['ifeval','gsm8k']:
  for seed in c['generation_seeds']:result.append(dict(name=task,seed=seed,kind='general',generative=True))
 result.append(dict(name='ppl',seed=c['classification_seed'],kind='ppl',generative=False))
 return result

def valid_done(d,ident):
 if not (d/'DONE.json').exists():return False
 if (d/'FAILED.json').exists():raise RuntimeError('Contradictory DONE and FAILED: '+str(d))
 record=json.loads((d/'DONE.json').read_text())
 if not record.get('artifacts') or 'scores.json' not in record['artifacts'] or 'job.json' not in record['artifacts']:raise RuntimeError('Incomplete DONE manifest: '+str(d))
 if record['identity']!=ident:raise RuntimeError('任务身份变化: '+str(d))
 for name,h in record['artifacts'].items():
  if not (d/name).is_file() or sha(d/name)!=h:raise RuntimeError('结果损坏: '+str(d/name))
 return True

def summarize(dest,c,expected,identity):
 collected={};per_run=[]
 for job,attempt,scores in unique_completed(dest,expected,valid_done,identity):
  jf=attempt/'scores.json'
  for metric,value in scores.items():
   collected.setdefault(metric,[]).append(value);per_run.append(dict(metric=metric,seed=job['seed'],score=value,source=str(jf)))
 out={k:{'mean':statistics.mean(v),'std':statistics.stdev(v) if len(v)>1 else 0,'runs':len(v)} for k,v in collected.items()}
 write(dest/'summary.json',out)
 with (dest/'scores_by_run.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['metric','seed','score','source']);w.writeheader();w.writerows(per_run)
 return out

def run_model(model,c,a,prot):
 mid=model_identity(model,c);root=Path(c['output_root']);key=digest(mid)[:16]
 dest=(root/'smoke'/digest(prot)[:16] if a.smoke else root/'evaluation')/key/a.replica;dest.mkdir(parents=True,exist_ok=True)
 write(dest/'model.json',mid);write(dest/'protocol.json',prot);write(dest/'config.json',c)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(c['gpu']),HF_HOME=c['hf_cache'],HF_HUB_CACHE=c['hf_cache'],HF_DATASETS_CACHE=str(Path(c['hf_cache'])/'datasets'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 if c.get('evaluation_data_root'):env['BASELINE_EVAL_DATA_ROOT']=str(Path(c['evaluation_data_root']).resolve())
 if c.get('lighteval_source'):env['PYTHONPATH']=c['lighteval_source']+os.pathsep+env.get('PYTHONPATH','')
 def execute_job(j,gpu):
  task_env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu))
  unit=dest/(j['name']+'-seed'+str(j['seed']));ident={'protocol':digest(prot),'model':digest(mid),'job':j,'smoke':a.smoke}
  waiting=unit/'WAIT_FOR_REUSE.json'
  while waiting.exists():
   owner=json.loads(waiting.read_text())
   try:
    owner_cmd=Path('/proc/'+str(owner['pid'])+'/cmdline').read_bytes()
   except FileNotFoundError:
    if not waiting.exists():break
    raise RuntimeError('Result handoff stopped before completion: '+str(waiting))
   if owner['script'].encode() not in owner_cmd:raise RuntimeError('Result handoff PID identity changed')
   time.sleep(2)
  done=False
  for attempt in sorted(unit.glob('attempt-*')):
   if valid_done(attempt,ident):done=True;break
  if done:return
  for mount in [root,Path(c['hf_cache']),Path('/')]:
   if shutil.disk_usage(mount).free<c['reserve_bytes']+268435456:raise RuntimeError('剩余空间不足，未删除任何文件: '+str(mount))
  attempt=unit/('attempt-'+str(time.time_ns()));attempt.mkdir(parents=True)
  write(attempt/'job.json',ident)
  write(attempt/'execution.json',{'physical_gpu':gpu,'pid':os.getpid(),'scheduler':'static_disjoint_lanes'})
  cmd=[c['training_python'] if j['kind']=='ppl' else c['evaluation_python'],str(R/'scripts/baseline_eval_worker.py'),'--config',str(a.config),'--model',str(model),'--output',str(attempt)]
  print('RUN',j['name'],'seed',j['seed'],'model',model,flush=True)
  with (attempt/'run.log').open('w') as log:rc=subprocess.run(cmd,env=task_env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(a.lock_fd,)).returncode
  if rc:
   write(attempt/'FAILED.json',{'returncode':rc});raise RuntimeError('任务失败，详见 '+str(attempt/'run.log'))
  if not (attempt/'scores.json').exists():raise RuntimeError('缺少分数: '+str(attempt))
  require_scores(json.loads((attempt/'scores.json').read_text()))
  artifacts={str(p.relative_to(attempt)):sha(p) for p in attempt.rglob('*') if p.is_file()}
  write(attempt/'DONE.json',{'identity':ident,'artifacts':artifacts})
 gpu_ids=c.get('evaluation_gpus',[c['gpu']])
 if not gpu_ids or len(gpu_ids)!=len(set(gpu_ids)):raise ValueError('GPU lanes must be nonempty and unique')
 pending_jobs=jobs(c,a.smoke)
 stop=Event()
 def lane(index,gpu):
  try:
   for j in pending_jobs[index::len(gpu_ids)]:
    if stop.is_set():return
    execute_job(j,gpu)
  except BaseException:
   stop.set()
   raise
 with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
  futures=[pool.submit(lane,index,gpu) for index,gpu in enumerate(gpu_ids)]
  for future in futures:future.result()
 summary=summarize(dest,c,jobs(c,a.smoke),{'protocol':digest(prot),'model':digest(mid),'smoke':a.smoke});write(dest/'ALL_DONE.json',{'protocol':digest(prot),'model':mid,'jobs':len(jobs(c,a.smoke))})
 return dest,summary

def main():
 p=argparse.ArgumentParser(description='只需模型路径；固定英语/Igbo协议。')
 p.add_argument('model',type=Path);p.add_argument('--config',type=Path,required=True)
 p.add_argument('--check',action='store_true',help='仅校验和打印计划，不运行模型')
 p.add_argument('--smoke',action='store_true',help='独立小样本验证，不进入正式结果')
 p.add_argument('--replica',choices=['primary','repeat'],default='primary',help='独立重跑用于验证；默认安全续跑')
 a=p.parse_args();c=json.loads(a.config.read_text());root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=True)
 with (root/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  a.lock_fd=lock.fileno()
  prot=protocol(c,a.config);fp=root/'protocol_lock.json'
  if not a.smoke and fp.exists() and json.loads(fp.read_text())!=prot:raise RuntimeError('冻结协议发生变化；需检查并建立新版本，不能混合旧结果')
  model_identity(a.model,c)
  if a.check:print(json.dumps({'protocol':digest(prot),'jobs_per_model':len(jobs(c,a.smoke)),'model':str(a.model),'frozen':fp.exists()},indent=2));return
  if not a.smoke and not fp.exists():write(fp,prot)
  # Always establish the denominator under this very same protocol.
  base_dest,base=run_model(Path(c['base_model']),c,a,prot)
  dest,scores=(base_dest,base) if a.model.resolve()==Path(c['base_model']).resolve() else run_model(a.model,c,a,prot)
  relative,totals=aggregate(scores,base,c['aggregate'])
  write(dest/'relative_to_base.json',{'base':str(base_dest),'relative_percent':relative,'aggregate_percent':totals})
  table=['# 统一评测结果'+('（smoke，仅验证流程）' if a.smoke else ''),'','|指标|原始均值|跨生成种子标准差|Base均值|相对变化%|','|---|---:|---:|---:|---:|']
  records=[]
  for k,v in scores.items():
   delta=relative[k];shown='不可计算（Base为0）' if delta is None else f'{delta:.4f}'
   table.append(f"|{k}|{v['mean']:.6f}|{v['std']:.6f}|{base[k]['mean']:.6f}|{shown}|")
   records.append(dict(metric=k,mean=v['mean'],std=v['std'],runs=v['runs'],base_mean=base[k]['mean'],relative_percent=delta))
  table+=['','EN3 / Target3: '+json.dumps(totals,ensure_ascii=False),'','PPL/NLL仅诊断；MMLU/G-MMLU不进入综合。null为不可计算。']
  (dest/'RESULTS_ZH.md').write_text('\n'.join(table)+'\n')
  with (dest/'raw_and_relative.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  write(dest/'EVALUATION_DONE.json',{'protocol':digest(prot),'model_path':str(a.model),'base':str(base_dest)})
  print('COMPLETE',dest,totals,flush=True)
if __name__=='__main__':main()
