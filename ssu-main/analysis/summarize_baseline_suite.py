"""Publish verified partial scores without treating incomplete seeds as full results."""
import argparse,csv,fcntl,io,json,statistics,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts/src'))
from run_baseline_eval import digest,valid_done,jobs
from baseline_eval_integrity import require_scores

def write(path,text):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)

def refresh():
 c=json.loads((R/'configs/ig_baseline_eval_v3.json').read_text());root=Path(c['output_root']);suite=root.parent/'ig_baseline_suite_v1'
 protocol=digest(json.loads((root/'protocol_lock.json').read_text()));expected={(j['name'],j['seed']):j for j in jobs(c,False)}
 models={'base':str(Path(c['base_model']).resolve())}
 old=json.loads((R/'configs/ig_baseline_repro_v1.json').read_text())['models']
 for method in ['full_fft','ssu_en_freeze50']:models['legacy_'+method]=str(Path(old[method]).resolve())
 train=json.loads((R/'configs/ig_baseline_train_v1.json').read_text())
 for method in train['methods']:models['new_'+method]=str((Path(train['output_root'])/method).resolve())
 bypath={v:k for k,v in models.items()};data={k:{} for k in models};completed={k:0 for k in models};raw=[]
 for identity_file in (root/'evaluation').glob('*/primary/model.json'):
  dest=identity_file.parent;mid=json.loads(identity_file.read_text());name=bypath.get(str(Path(mid['path']).resolve()))
  if name is None:continue
  for (task,seed),job in expected.items():
   good=[]
   for attempt in (dest/(task+'-seed'+str(seed))).glob('attempt-*'):
    if valid_done(attempt,{'protocol':protocol,'model':digest(mid),'job':job,'smoke':False}):good.append(attempt)
   if len(good)>1:raise RuntimeError('Duplicate completed task: '+str(dest)+' '+task)
   if not good:continue
   attempt=good[0];scores=json.loads((attempt/'scores.json').read_text());require_scores(scores);completed[name]+=1
   for metric,value in scores.items():
    data[name].setdefault(metric,[]).append((seed,value,job['generative']))
    raw.append({'model':name,'task':task,'metric':metric,'seed':seed,'score':value,'source':str(attempt/'scores.json'),'protocol':protocol})
 complete={}
 for name,metrics in data.items():
  complete[name]={k:statistics.mean(x[1] for x in values) for k,values in metrics.items() if len(values)==(len(c['generation_seeds']) if values[0][2] else 1)}
 groups=c['aggregate'];base=complete['base'];totals={};summaries=[]
 for name,metrics in data.items():
  totals[name]={}
  for group,members in groups.items():
   totals[name][group]=statistics.mean((complete[name][k]/base[k]-1)*100 for k in members) if all(k in complete[name] and k in base and base[k]!=0 for k in members) else None
  for metric,values in metrics.items():
   final=complete[name].get(metric);b=base.get(metric)
   summaries.append({'model':name,'metric':metric,'completed_seeds':len(values),'required_seeds':len(c['generation_seeds']) if values[0][2] else 1,'available_seed_mean':statistics.mean(x[1] for x in values),'complete_metric_mean':final,'base_complete_mean':b,'relative_percent':None if final is None or b in (None,0) else (final/b-1)*100})
 def csvout(path,rows,fields):
  f=io.StringIO();w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);write(path,f.getvalue())
 csvout(suite/'formal_scores_by_seed.csv',raw,['model','task','metric','seed','score','source','protocol'])
 csvout(suite/'formal_metrics_progress.csv',summaries,['model','metric','completed_seeds','required_seeds','available_seed_mean','complete_metric_mean','base_complete_mean','relative_percent'])
 lines=['# 正式评测逐任务进度','','仅汇总通过DONE身份与文件哈希校验的正式结果。当前每个任务仅使用seed42；综合分须三项及同协议Base全部齐全。缺失项留空，不能当作零。','','|模型|完成任务数|EN3变化%|Target3变化%|','|---|---:|---:|---:|']
 for name in models:
  fmt=lambda v:'—' if v is None else f'{v:+.4f}'
  lines.append(f"|{name}|{completed[name]}/{len(expected)}|{fmt(totals[name]['en'])}|{fmt(totals[name]['target'])}|")
 lines+=['','## 已完成seed的原始分数','','当前统一seed42；单任务分数不代表整体效果。','','|模型|指标|seed|原始分数|','|---|---|---:|---:|']
 lines += [f"|{x['model']}|{x['metric']}|{x['seed']}|{x['score']:.6f}|" for x in raw]
 lines+=['','EN3：EN Belebele、EN摘要、IG→EN；Target3：IG Belebele、IG摘要、EN→IG。各项相对Base变化等权平均。通用任务及PPL单列，不混入综合。']
 write(suite/'FORMAL_RESULTS_PROGRESS_ZH.md','\n'.join(lines)+'\n')
 write(suite/'formal_progress.json',json.dumps({'protocol':protocol,'completed_jobs':completed,'aggregates':totals},ensure_ascii=False,indent=2)+'\n')
 from compare_baseline_history import generate
 generate()
 from summarize_baseline_training import generate as training_summary
 training_summary()
 return completed
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',type=int,default=0);a=p.parse_args();suite=R.parent/'experiments/ig_baseline_suite_v1'
 with (suite/'summary.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   try:
    result=refresh()
    if not a.watch:print(json.dumps(result));break
   except Exception as error:
    write(suite/'SUMMARY_FAILED.json',json.dumps({'error':repr(error),'time':time.time()}));raise
   time.sleep(max(30,a.watch))
