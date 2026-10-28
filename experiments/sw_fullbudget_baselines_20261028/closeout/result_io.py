"""Read and verify saved endpoint outputs without re-running inference."""
from pathlib import Path
import sys,json,hashlib,math,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from settings import *
import numpy as np
import pyarrow.parquet as pq
from functools import lru_cache

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
@lru_cache(maxsize=1)
def generation_scorers():
 # Load the real harness before the vendored IFEval hierarchy can register stubs.
 legacy=H/'reproductions/sw_d2_20260921'
 for path in (legacy/'harness_local',legacy/'lm_eval_metadata'):sys.path.insert(0,str(path))
 os.environ.setdefault('NLTK_DATA',str(H/'nltk_data'))
 import lm_eval
 from lm_eval.filters.extraction import RegexFilter
 from lm_eval.api.metrics import exact_match_hf_evaluate
 from transformers import AutoTokenizer
 from scoring import instruction,translation
 import yaml
 config_path=legacy/'harness_local/lm_eval/tasks/gsm8k/gsm8k.yaml'
 assert sha(config_path)==read(REPORT/'SCORER_ACCEPTANCE.json')['official_GSM8K_yaml_sha256']
 config=yaml.safe_load(config_path.read_text())
 return AutoTokenizer.from_pretrained(BASE,local_files_only=True),instruction,translation,RegexFilter,exact_match_hf_evaluate,config

def read_generation(model,task,stage='formal'):
 out=evaluation_output(model,stage,task);done=read(out/'DONE.json');assert done['smoke']==(stage=='smoke')
 rows=[json.loads(s) for s in (out/'PREDICTIONS.jsonl').open() if s.strip()]
 source=source_rows(task);tok,instruction,translation,RegexFilter,exact_match,config=generation_scorers()
 if task=='gsm8k':
  grouped=collections.defaultdict(dict)
  for r in rows:
   label=r['filter'];assert label not in grouped[str(r['doc_id'])];grouped[str(r['doc_id'])][label]=r
   assert r['doc']==source[int(r['doc_id'])] and r['target']==r['doc']['answer']
   spec=next(f for f in config['filter_list'] if f['name']==label)['filter'][0]
   filtered=RegexFilter(**{k:v for k,v in spec.items() if k!='function'}).apply(r['resps'],[r['doc']])[0][0]
   assert r['filtered_resps']==[filtered]
   args={k:v for k,v in config['metric_list'][0].items() if k not in ('metric','aggregation','higher_is_better')}
   recomputed=exact_match(predictions=[filtered],references=[r['target']],**args)['exact_match']
   assert float(recomputed)==float(r['exact_match'])
  expected=1 if stage=='smoke' else 1319
  assert set(grouped)==set(map(str,range(expected)))
  for group in grouped.values():
   assert set(group)=={'strict-match','flexible-extract'}
   assert group['strict-match']['resps']==group['flexible-extract']['resps']
  for label,name in (('strict-match','strict_ACC'),('flexible-extract','flexible_ACC')):
   assert abs(100*np.mean([r[label]['exact_match'] for r in grouped.values()])-done['scores'][name])<1e-9
  return grouped,done['scores']
 if stage=='smoke':source=source[:4]
 assert len(rows)==len(source)==(4 if stage=='smoke' else COUNTS[task])
 keyed={r['id']:r for r in rows};assert len(keyed)==len(rows)
 origins={('ifeval:'+str(r['key'])) if task=='ifeval' else r['id']:r for r in source}
 assert set(keyed)==set(origins)
 for id,r in keyed.items():
  origin=origins[id];assert r['model']==model
  assert r['prompt_token_ids']==tok.apply_chat_template([dict(role='user',content=origin['prompt'])],tokenize=True,add_generation_prompt=True)
  assert r['generated_tokens']==len(r['token_ids']) and r['prompt_tokens']==len(r['prompt_token_ids'])
  assert r['text']==tok.decode(r['token_ids'],skip_special_tokens=True).strip()
  rescored=instruction(origin,r['text']) if task=='ifeval' else translation(r['text'],origin['reference'])
  assert rescored==r['scores'],(task,id,'saved score differs from frozen scorer')
 if task=='translation':
  if stage!='smoke':assert collections.Counter(r['task'] for r in rows)=={'en_to_sw':1012,'sw_to_en':1012}
  from sacrebleu.metrics import CHRF
  metric=CHRF(char_order=6,word_order=2,beta=2)
  for direction in {r['task'] for r in rows}:
   here=[r for r in rows if r['task']==direction]
   score=metric._compute_score_from_stats(np.array([r['scores']['chrfpp_stats'] for r in here]).sum(0).tolist()).score
   assert len(here)==done['scores'][direction]['n'] and abs(score-done['scores'][direction]['corpus_chrfpp'])<1e-9
 if task=='ifeval':
  denominator=sum(len(r['scores']['instruction_pass']) for r in rows)
  assert denominator==done['scores']['instruction_denominator']
  if stage!='smoke':assert denominator==834
  assert abs(100*np.mean([r['scores']['prompt_strict'] for r in rows])-done['scores']['prompt_strict_ACC'])<1e-9
  assert abs(100*sum(sum(r['scores']['instruction_pass']) for r in rows)/denominator-done['scores']['instruction_strict_ACC'])<1e-9
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
