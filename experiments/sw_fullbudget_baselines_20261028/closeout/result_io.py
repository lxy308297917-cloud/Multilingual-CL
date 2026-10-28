"""Read and verify saved endpoint outputs without re-running inference."""
from pathlib import Path
import sys,json,hashlib,math,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from settings import *
import numpy as np
import pyarrow.parquet as pq

TASKS=('belebele_en','belebele_sw','knowledge_en','knowledge_sw','translation','ifeval','gsm8k')
COUNTS=dict(belebele_en=900,belebele_sw=900,knowledge_en=14042,knowledge_sw=14042,translation=2024,ifeval=541,gsm8k=1319)
def normalize(s):return ' '.join(s.split()).casefold()
def source_rows(task):
 p=read(REPORT/'EVALUATION_FROZEN.json')['sources'][task]
 assert sha(p['path'])==p['sha256']
 return [json.loads(s) for s in Path(p['path']).open() if s.strip()]
def scalar_gold(x):
 if isinstance(x,list):assert len(x)==1;return int(x[0])
 return int(x)
def read_mcf(model,task,stage='formal'):
 out=evaluation_output(model,stage,task);done=read(out/'DONE.json');assert done['smoke']==(stage=='smoke')
 files=list(out.glob('details/**/*.parquet'));assert files
 source=source_rows(task);by_id={r['id']:r for r in source} if task.startswith('knowledge') else None
 records={};subjects=collections.defaultdict(list)
 for path in files:
  for r in pq.read_table(path).to_pylist():
   doc=r['doc'];gold=scalar_gold(doc['gold_index']);scores=r['model_response']['logprobs']
   assert len(scores)==4 and all(math.isfinite(v) for v in scores)
   assert r['model_response']['truncated_tokens_count']==0
   if task.startswith('knowledge'):
    subject=doc['task_name'].split(':',1)[1].split('|',1)[0]
    key=f'{subject}/test/{int(doc["id"])}';origin=by_id[key];group=subject
    assert 'ABCD'.index(origin['answer'])==gold
    assert normalize(origin['question']) in normalize(doc['query'])
    for choice in origin['choices']:assert normalize(choice) in normalize(doc['query'])
   else:
    origin=source[int(doc['id'])];key=origin['link']+':'+str(origin['question_number']);group=origin['link']
    assert int(origin['correct_answer_num'])-1==gold
    assert normalize(origin['question']) in normalize(doc['query'])
    assert normalize(origin['flores_passage']) in normalize(doc['query'])
    for i in range(1,5):assert normalize(origin['mc_answer'+str(i)]) in normalize(doc['query'])
   prediction=int(np.argmax(scores));correct=int(prediction==gold);assert correct==int(r['metric']['acc'])
   inputs=r['model_response']['input_tokens'];assert isinstance(inputs,list) and inputs
   assert key not in records,key
   records[key]=dict(id=key,group=group,gold=gold,predicted=prediction,correct=correct,logprobs=scores,
    input_sha256=hashlib.sha256(json.dumps(inputs,separators=(',',':')).encode()).hexdigest(),input_tokens=len(inputs))
   subjects[group].append(correct)
 expected=1 if stage=='smoke' else COUNTS[task];assert len(records)==expected
 assert abs(100*sum(r['correct'] for r in records.values())/expected-done['scores']['ACC'])<1e-9
 return records,dict(n=expected,ACC_percent=100*sum(r['correct'] for r in records.values())/expected,
  macro_source_ACC_percent=100*np.mean([np.mean(v) for v in subjects.values()]),source_groups=len(subjects))
def read_generation(model,task):
 out=evaluation_output(model,'formal',task);done=read(out/'DONE.json');assert not done['smoke']
 rows=[json.loads(s) for s in (out/'PREDICTIONS.jsonl').open() if s.strip()]
 if task=='gsm8k':
  grouped=collections.defaultdict(dict)
  for r in rows:
   label=r['filter'];assert label not in grouped[str(r['doc_id'])];grouped[str(r['doc_id'])][label]=r
  assert len(grouped)==1319
  for group in grouped.values():
   assert set(group)=={'strict-match','flexible-extract'}
   assert group['strict-match']['resps']==group['flexible-extract']['resps']
  return grouped,done['scores']
 assert len(rows)==COUNTS[task]
 keyed={r['id']:r for r in rows};assert len(keyed)==len(rows)
 if task=='translation':assert collections.Counter(r['task'] for r in rows)=={'en_to_sw':1012,'sw_to_en':1012}
 if task=='ifeval':
  denominator=sum(len(r['scores']['instruction_pass']) for r in rows)
  assert denominator==done['scores']['instruction_denominator']==834
 return keyed,done['scores']

def paired_interval(x,y,groups,seed=42,repeats=10000):
 """Source-cluster paired accuracy difference; not training-seed uncertainty."""
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);assert len(x)==len(y)==len(groups)
 keys=list(dict.fromkeys(groups));index=np.array([keys.index(g) for g in groups])
 numer=np.bincount(index,weights=x-y,minlength=len(keys));denom=np.bincount(index,minlength=len(keys))
 rng=np.random.default_rng(seed);w=rng.multinomial(len(keys),[1/len(keys)]*len(keys),size=repeats)
 values=(w@numer)/(w@denom)
 return dict(difference_pp=float(100*(x-y).mean()),paired95CI_pp=(100*np.quantile(values,[.025,.975])).tolist(),
  n=len(x),source_groups=len(keys),bootstrap_replicates=repeats,training_seed_uncertainty=False)

if __name__=='__main__':
 for task in ('belebele_en','belebele_sw','knowledge_en','knowledge_sw'):
  rows,metric=read_mcf('Base',task,stage='smoke');print(task,metric)
