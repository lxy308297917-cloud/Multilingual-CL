from settings import *
import collections,csv
import numpy as np
sys.path.insert(0,str(PRIOR/'code'))
from paired_statistics import mean_ci
meta=read(RUN/'CONFIRMATION_FROZEN.json');data=rows(meta['data_path']);metrics={'sw_reading':'content_f1','en_reading':'content_correct','en_instruction':'prompt_strict'};allpred={};scores={};missing=[]
for seed in ('42','43'):
 for step in (2146,4293):
  name=f'LR077-s{seed}'+('-mid' if step==2146 else '')
  folder=(RUN if step==2146 else PRUN)/'evaluation/confirmation'/name
  if not (folder/'DONE.json').exists():missing.append(name);continue
  done=read(folder/'DONE.json');assert sha(folder/'PREDICTIONS.jsonl')==done['predictions_sha256'];assert done['data_sha256']==meta['data_sha256']
  rr=rows(folder/'PREDICTIONS.jsonl');assert len(rr)==256 and len({r['id'] for r in rr})==256;lookup={r['id']:r for r in rr};expected=[r['id'] for t in ('sw_reading','en_reading','translation','en_instruction') for r in data if r['study_partition']==t];assert [r['id'] for r in rr]==expected
  label=f'order{seed}-step{step}';allpred[label]=lookup;scores[label]={}
  for task,key in metrics.items():
   batch=[r for r in rr if r['study_partition']==task];scores[label][task]={'score':100*float(np.mean([r['scores'][key] for r in batch])),'n':len(batch),'capped':sum(r['capped'] for r in batch),'generated_token_mean':float(np.mean([r['generated_tokens'] for r in batch])),'language_counts':dict(collections.Counter(r['language_diagnostic'] for r in batch))}
comparisons={}
for first,second in [('order42-step2146','order43-step2146'),('order42-step4293','order43-step4293'),('order42-step4293','order42-step2146'),('order43-step4293','order43-step2146')]:
 if first not in allpred or second not in allpred:continue
 comparisons[first+' minus '+second]={}
 for task,key in metrics.items():
  rr=[r for r in data if r['study_partition']==task];a=allpred[first];b=allpred[second];comparisons[first+' minus '+second][task]=mean_ci([a[r['id']]['scores'][key] for r in rr],[b[r['id']]['scores'][key] for r in rr],[r['group_id'] for r in rr])
write(REPORT/'RESULTS.json',{'scores':scores,'paired_comparisons':comparisons,'missing':missing,'new_generation_count':sum(len(v) for k,v in allpred.items() if k.endswith('2146')),'GPU_context_hours':usage()[0]/3600,'checkpoint_selection':False,'statistical_scope':'Fixed-model source-item uncertainty, not training-seed population; existing dev diagnostic only.'})
with (REPORT/'SCORES.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['model','task','score','n','capped'])
 for name,ss in scores.items():
  for task,r in ss.items():writer.writerow([name,task,r['score'],r['n'],r['capped']])
print(json.dumps({'scores':scores,'comparisons':comparisons,'missing':missing},ensure_ascii=False,indent=2))
