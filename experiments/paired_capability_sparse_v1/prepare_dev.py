"""Create a fresh grouped train-side development set, before any new model score."""
import ast,hashlib,json,re,sys,unicodedata
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import DetectorFactory,detect
from mastery import numeric_value
from prepare_calibration import SYSTEM
ROOT=Path(__file__).resolve().parent;H=Path('/data0/users/lixiaoyang');DATA=H/'data';OLD=DATA/'sw_d2_next_dev_20261008';OUT=ROOT/'data/dev_v2'
SEED='paired-capability-unused-dev-20261021-v1';DetectorFactory.seed=0

def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def digest(s):return hashlib.sha256(norm(s).encode()).hexdigest()
def rank(s):return hashlib.sha256((SEED+':'+str(s)).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def neartext(s):return re.sub(r'\d+(?:[.,]\d+)*',' NUM ',norm(s))
def main():
 if OUT.exists():raise FileExistsError('Immutable dev output already exists; inspect before preparing another version')
 tok=AutoTokenizer.from_pretrained(H/'models/Qwen2.5-1.5B-Instruct',local_files_only=True)
 prior_paths=sorted(set(DATA.glob('*/*.jsonl')) | {DATA/'sw-d2-inputs/sw_update_utility_v1/calibration_candidates.jsonl'})
 used_ids=set();used_articles=set();used_stories=set();used_texts=set();prior_sources=[]
 for p in prior_paths:
  if p.stat().st_size>20<<20:continue
  try:rr=rows(p)
  except (ValueError,UnicodeDecodeError):continue
  prior_sources.append({'path':str(p),'sha256':sha(p)})
  for r in rr:
   if not isinstance(r,dict):continue
   qid=str(r.get('id',''));used_ids.add(qid)
   used_ids.update(re.findall(r'personahub_[A-Za-z0-9_-]+',qid))
   if qid.startswith('RACE:train:'):used_articles.add(qid.split(':')[2])
   if r.get('article_id'):used_articles.add(str(r['article_id']))
   if r.get('story_id'):used_stories.add(str(r['story_id']))
   for k in ('prompt','question','context','text','source_text'):
    if isinstance(r.get(k),str):used_texts.add(digest(r[k]))
 selected_path=H/'Documents/Codex/2026-10-20/multilingual-continual-learning-github-https-github/work/gsm_bilingual_20261020/SELECTED.json'
 selected=json.loads(selected_path.read_text());used_ids.update(x['id'] for x in selected)
 cal=rows(ROOT/'data/calibration_pairs.v1.jsonl');used_ids.update(x['id'] for x in cal)
 for x in cal:used_texts.update((digest(x['question_en']),digest(x['question_sw'])))
 evalroot=DATA/'sw-d2-eval-inputs/data/sw_fineweb2_aya_v1/eval_data'
 formal_paths=[evalroot/d/'test.jsonl' for d in ('sum','en2sw','sw2en','belebele_sw')]
 # Historical summary directory names are explicit and verified below.
 if not formal_paths[0].exists():formal_paths[0]=evalroot/'sum_sw/test.jsonl'
 source_manifest=json.loads((DATA/'sw_next_dev_reserve_20261009/manifest.json').read_text())
 source_expected={}
 for info in source_manifest['partitions'].values():
  src=info['sources'];source_expected[src['source']]=src['source_sha256']
 kensw=json.loads((DATA/'sw_reading_kenswquad_20261009/manifest.json').read_text());source_expected[kensw['source_path']]=kensw['source_sha256']
 ifpin=json.loads((OLD/'english_instruction_manifest.json').read_text());ifsource=OLD/'sources/if-multi-train-7c1813b.parquet';source_expected[str(ifsource)]=ifpin['source_sha256']
 source_checks={p:sha(p) for p in source_expected}
 for p,expected in source_expected.items():
  if source_checks[p]!=expected:raise RuntimeError('Pinned source changed: '+p)
 for p in formal_paths:
  if not p.exists():raise FileNotFoundError(p)
  for r in rows(p):
   for key in ('id',):
    if key in r:used_ids.add(str(r[key]))
   for k in ('text','source_text','flores_passage','question','prompt'):
    if isinstance(r.get(k),str):used_texts.add(digest(r[k]))
 result={};audit={}
 # Bidirectional translation: group the same source parallel pair and keep both directions.
 parallel=defaultdict(dict)
 for r in rows(DATA/'sw-d2-inputs/d2_bilingual_v1/sw/dev.jsonl'):
  if r['task'] in ('translation_en2target','translation_target2en'):parallel[r['group_key']][r['task']]=r
 candidates=[]
 for group,pair in sorted(parallel.items(),key=lambda x:rank(x[0])):
  if len(pair)!=2 or any(r['id'] in used_ids for r in pair.values()):continue
  entries=[]
  for task,r in pair.items():
   text=r['prompt'].split('\n\n',1)[-1]
   if digest(text) in used_texts:break
   entries.append({'id':r['id'],'group_id':group,'task':'en_to_sw' if task=='translation_en2target' else 'sw_to_en','prompt':r['prompt'],'reference':r['answer'],'source_text':text})
  if len(entries)==2:candidates.append(entries)
 result['translation']=[r for entries in candidates[:128] for r in entries]
 # Summary: entirely new source articles, no model-based filtering.
 summary=[]
 for r in sorted(pq.read_table(OLD/'sources/xlsum-swahili-train-fcbaaea.parquet').to_pylist(),key=lambda r:rank(r['id'])):
  if r['id'] in used_ids or digest(r['text']) in used_texts or not r['summary'].strip():continue
  if len(tok.encode(r['text'],add_special_tokens=False))>1024:continue
  summary.append({'id':r['id'],'group_id':'xlsum:'+r['id'],'task':'sw_summary','prompt':'Fupisha maandishi yafuatayo kwa Kiswahili: '+r['text'],'reference':r['summary'],'source_text':r['text']})
  if len(summary)==128:break
 result['sw_summary']=summary
 # English reading: one question per entirely unused article.
 articles={}
 for i,r in enumerate(pq.read_table(OLD/'sources/race-all-train-e37ee6f.parquet').to_pylist()):
  key=r['example_id']
  if key in used_articles or digest(r['article']) in used_texts or len(r['options'])!=4 or r['answer'] not in 'ABCD':continue
  prompt=('Read the passage and choose the correct answer. Reply with the option letter only.\n\nPassage:\n'+r['article']+'\n\nQuestion: '+r['question']+'\n'+'\n'.join(a+'. '+b for a,b in zip('ABCD',r['options'])))
  if len(tok.encode(prompt,add_special_tokens=False))>2048:continue
  item={'id':f'RACE:train:{key}:source-row-{i}','group_id':'race:'+key,'task':'en_reading','prompt':prompt,'reference':r['answer'],'source_text':r['article']}
  if key not in articles or rank(item['id'])<rank(articles[key]['id']):articles[key]=item
 result['en_reading']=[articles[k] for k in sorted(articles,key=rank)[:256]]
 # Swahili extractive reading: new story IDs, with answer demonstrably in context.
 stories={}
 for r in pq.read_table(kensw['source_path']).to_pylist():
  story=str(r['story_id']);context=r['context'].strip();question=r['question'].strip();answer=r['answer'].strip()
  if story in used_stories or r['id'] in used_ids or any(digest(s) in used_texts for s in (context,question)):continue
  if not answer or len(answer)>80 or norm(answer) not in norm(context):continue
  if len(tok.encode(context,add_special_tokens=False))>1024:continue
  prompt='Soma kifungu kifuatacho kisha jibu swali kwa maneno machache kutoka kwenye kifungu.\n\nKifungu:\n'+context+'\n\nSwali: '+question+'\nJibu:'
  item={'id':r['id'],'group_id':'kenswquad:'+story,'task':'sw_reading_kenswquad','prompt':prompt,'reference':answer,'source_text':context}
  if story not in stories or rank(r['id'])<rank(stories[story]['id']):stories[story]=item
 result['sw_reading_kenswquad']=[stories[k] for k in sorted(stories,key=rank)[:128]]
 # Mathematics: fresh GSM8K train questions; exclude prior source IDs and high lexical/template similarities to calibration, old dev, and formal test.
 gsmpath=Path(source_manifest['partitions']['en_math']['sources']['source']);train=list(Dataset.from_file(str(gsmpath)));test=list(Dataset.from_file(str(gsmpath.with_name('gsm8k-test.arrow'))))
 blocked_indices={int(x.split(':')[1]) for x in used_ids if re.fullmatch(r'gsm8k_train:\d+',x)}
 blocked=[r['question'] for r in test]+[train[i]['question'] for i in sorted(blocked_indices) if i<len(train)]
 vector=TfidfVectorizer(ngram_range=(1,2),stop_words='english',sublinear_tf=True,min_df=1)
 matrix=vector.fit_transform([neartext(r['question']) for r in train]+[neartext(q) for q in blocked]);sim=matrix[:len(train)]@matrix[len(train):].T
 maxima=np.asarray(sim.max(axis=1).toarray()).ravel();order=sorted(range(len(train)),key=lambda i:rank('gsm8k_train:'+str(i)))
 math_rows=[];math_idxs=[]
 for i in order:
  r=train[i];qid='gsm8k_train:'+str(i);goldtext=r['answer'].rsplit('####',1)[-1].strip();gold=numeric_value(goldtext)
  if i in blocked_indices or len(r['question'])>650 or gold is None or maxima[i]>=.72:continue
  if math_idxs and float((matrix[i]@matrix[math_idxs].T).max())>=.72:continue
  math_rows.append({'id':qid,'group_id':qid,'task':'en_math','prompt':r['question'],'reference':goldtext,'source_worked_answer':r['answer'],'nearest_blocked_tfidf':float(maxima[i]),'system':SYSTEM});math_idxs.append(i)
  if len(math_rows)==128:break
 result['en_math']=math_rows;audit['math_near_exclusion']={'threshold':.72,'method':'English TFIDF unigrams+bigrams with numeric masking; prior source IDs+formal source questions; one selection per near-similar candidate','blocked_questions':len(blocked),'selected_max_similarity':max(x['nearest_blocked_tfidf'] for x in math_rows),'limits':'Not a proof of semantic decontamination; top-neighbor review retained separately'}
 # IF-multi train personahub prompts; formal IFEval used only for content exclusion.
 formal=DATA/'sw-d2-eval-inputs/hf_cache/hub/datasets--google--IFEval/snapshots/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl'
 used_texts.update(digest(r['prompt']) for r in rows(formal))
 scorer=DATA/'sw-d2-eval-inputs/project/ssu-main/evaluation/src/ifeval.py';sys.path.insert(0,str(scorer.parent));import ifeval
 _,_,registry=ifeval.load_official_ifeval_modules(scorer.parent/'utils');instruction=[];options=[]
 for batch in pq.ParquetFile(ifsource).iter_batches(batch_size=1024):
  for r in batch.to_pylist():
   if not r['key'].startswith('personahub_') or r['key'] in used_ids or len(r['messages'])!=1 or r['messages'][0]['role']!='user':continue
   text=r['messages'][0]['content']
   if not 80<=len(text)<=4000 or digest(text) in used_texts:continue
   try:
    ground=ast.literal_eval(r['ground_truth'])
    if len(ground)!=1:continue
    ids,kwargs=ground[0]['instruction_id'],ground[0]['kwargs']
    if not ids or len(ids)!=len(kwargs) or any(i not in registry.INSTRUCTION_DICT for i in ids):continue
    if any(v is not None and not isinstance(v,dict) for v in kwargs):continue
   except (ValueError,SyntaxError,TypeError,KeyError):continue
   options.append({'id':r['key'],'group_id':'ifmulti:'+digest(text),'task':'en_instruction','prompt':text,'instruction_id_list':ids,'kwargs':[x or {} for x in kwargs]})
 for r in sorted(options,key=lambda x:rank(x['id'])):
  try:
   if detect(r['prompt'])!='en':continue
   for name,kw in zip(r['instruction_id_list'],r['kwargs']):
    check=registry.INSTRUCTION_DICT[name](name);check.build_description(**{k:v for k,v in kw.items() if v is not None})
    if 'prompt' in (check.get_instruction_args() or []):check.build_description(prompt=r['prompt'])
    if not isinstance(check.check_following('Acknowledged.'),bool):raise ValueError('Unsupported check')
  except Exception:continue
  if any(digest(x['prompt'])==digest(r['prompt']) for x in instruction):continue
  instruction.append(r)
  if len(instruction)==128:break
 result['en_instruction']=instruction
 expected={'translation':256,'sw_summary':128,'en_reading':256,'sw_reading_kenswquad':128,'en_math':128,'en_instruction':128}
 if {k:len(v) for k,v in result.items()}!=expected:raise RuntimeError('Insufficient new dev rows: '+str({k:len(v) for k,v in result.items()}))
 OUT.mkdir();entries={}
 for name,rr in result.items():
  p=OUT/(name+'.jsonl');p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rr));entries[name]={'path':str(p),'sha256':sha(p),'rows':len(rr),'ids':[x['id'] for x in rr]}
 manifest={'status':'frozen_inputs_before_new_model_scoring; main-stream and semantic neighbor audit pending','seed':SEED,'partitions':entries,'sources':source_checks,'prior_exclusion_files':prior_sources,'calibration_sha256':sha(ROOT/'data/calibration_pairs.v1.jsonl'),'prepare_sha256':sha(__file__),'scorer':str(scorer),'scorer_sha256':sha(scorer),'formal_exclusion_sha256':{str(p):sha(p) for p in formal_paths+[formal]},'audit':audit,'no_model_inference':True,'not_formal_benchmarks':['KenSwQuAD extractive dev is not Belebele','IF-multi train instruction dev is not formal IFEval'],'generation_max_new_tokens':{'translation':256,'sw_summary':256,'en_reading':16,'sw_reading_kenswquad':64,'en_math':1024,'en_instruction':1280},'generation':'greedy, same explicit batch and tokenizer/model configuration for all groups; retain raw tokens/caps/endings','selection':'one endpoint; frozen paired sample bootstrap10000 seed42; no checkpoint or hyperparameter search'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:len(v) for k,v in result.items()},indent=2))
if __name__=='__main__':main()
