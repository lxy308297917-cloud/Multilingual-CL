"""Replace two pre-inference, confirmed parallel duplicates; preserve other tasks."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
from prepare_dev import ROOT,H,DATA,SEED,rows,norm,digest,rank,sha
OLD=ROOT/'data/dev_v2';OUT=ROOT/'data/dev_v3'
def main():
 if OUT.exists():raise FileExistsError('No overwrite of development versions')
 m=json.loads((OLD/'manifest.json').read_text());audit=json.loads((OLD/'INDEPENDENT_AUDIT.json').read_text());bad={x['group_id'] for x in audit['parallel_high_similarity']}
 if len(bad)!=2:raise RuntimeError('Unexpected repair scope; inspect audit')
 used_ids=set();used_texts=set()
 for entry in m['prior_exclusion_files']:
  if sha(entry['path'])!=entry['sha256']:raise RuntimeError('Prior pool identity')
  for r in rows(entry['path']):
   if not isinstance(r,dict):continue
   used_ids.add(str(r.get('id','')))
   for k in ('prompt','question','context','text','source_text'):
    if isinstance(r.get(k),str):used_texts.add(digest(r[k]))
 for p in m['formal_exclusion_sha256']:
  for r in rows(p):
   used_ids.add(str(r.get('id','')))
   for k in ('prompt','question','context','text','source_text','flores_passage'):
    if isinstance(r.get(k),str):used_texts.add(digest(r[k]))
 for r in rows(ROOT/'data/calibration_pairs.v1.jsonl'):used_texts.update((digest(r['question_en']),digest(r['question_sw'])))
 paired=defaultdict(dict)
 for r in rows(DATA/'sw-d2-inputs/d2_bilingual_v1/sw/dev.jsonl'):
  if r['task'] in ('translation_en2target','translation_target2en'):paired[r['group_key']][r['task']]=r
 selected=[]
 for group,pair in sorted(paired.items(),key=lambda x:rank(x[0])):
  if group in bad or len(pair)!=2 or any(r['id'] in used_ids for r in pair.values()):continue
  rr=[]
  for task,r in pair.items():
   text=r['prompt'].split('\n\n',1)[-1]
   if digest(text) in used_texts:break
   rr.append({'id':r['id'],'group_id':group,'task':'en_to_sw' if task=='translation_en2target' else 'sw_to_en','prompt':r['prompt'],'reference':r['answer'],'source_text':text})
  if len(rr)==2:selected.extend(rr)
  if len(selected)==256:break
 if len(selected)!=256:raise RuntimeError('Insufficient independent pairs')
 OUT.mkdir()
 for task,entry in m['partitions'].items():
  target=OUT/(task+'.jsonl')
  if task=='translation':target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
  else:target.write_bytes(Path(entry['path']).read_bytes())
  rr=rows(target);entry.update(path=str(target),sha256=sha(target),rows=len(rr),ids=[r['id'] for r in rr])
 m['parent_manifest_sha256']=sha(OLD/'manifest.json');m['pre_inference_repair']={'code_sha256':sha(__file__),'excluded_groups':sorted(bad),'reason':'Confirmed same sentence instance across main/dev despite different NLLB group hashes: case/punctuation paraphrase and recognized/recognised spelling','model_scores_used':False,'other_task_bytes_unchanged':True}
 (OUT/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');print(json.dumps(m['pre_inference_repair'],indent=2))
if __name__=='__main__':main()
