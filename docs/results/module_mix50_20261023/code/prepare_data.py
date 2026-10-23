"""Fresh source-isolated confirmation development inputs; no model scores read."""
from settings import *
import collections,unicodedata,re,csv,ast
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import DetectorFactory,detect
DetectorFactory.seed=0
SELECTION='module-mix50-confirmation-20261023-v1'
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def key(s):return hashlib.sha256(norm(s).encode()).hexdigest()
def rank(s):return key(SELECTION+':'+str(s))
def main():
 assert not (RUN/'CONFIRMATION_FROZEN.json').exists()
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);used=set();texts=set();groups=set();articles=set();stories=set();prior=[];seen=set()
 # Historical exclusion manifest is an explicit, already-audited inventory.
 prior_m=json.loads((ROOT/'work/evaluation_audit_v2/DATA_FROZEN.json').read_text())
 paths={pathlib.Path(x['path']) for x in prior_m['prior_exclusion_files']}
 paths.update((ROOT/'work/evaluation_audit_v2').glob('*.jsonl'))
 paths.update((OLD.parent/'paired_capability_sparse_v1/data').glob('dev_*/*.jsonl'))
 paths.update((OLD/'data').glob('*.jsonl'))
 # Original full recipe, formal input and all previous dev/calibration included above.
 for p in sorted(paths):
  if not p.is_file() or p.stat().st_size>100<<20:continue
  h=sha(p)
  if h in seen:continue
  seen.add(h)
  try:rr=json.loads(p.read_text()) if p.suffix=='.json' else rows(p)
  except (ValueError,UnicodeDecodeError):continue
  if not isinstance(rr,list):continue
  prior.append({'path':str(p),'sha256':h,'rows':len(rr)})
  for r in rr:
   if not isinstance(r,dict):continue
   qid=str(r.get('id',''));used.add(qid);groups.add(str(r.get('group_id',r.get('group_key',''))))
   if qid.startswith('RACE:train:'):articles.add(qid.split(':')[2])
   for k in ('article_id','example_id'):
    if r.get(k):articles.add(str(r[k]))
   if r.get('story_id'):stories.add(str(r['story_id']))
   if re.fullmatch(r'\d+_swa_\d+',qid):stories.add(qid.rsplit('_',1)[0])
   for k in ('prompt','question','context','text','source_text','question_en','question_sw','flores_passage','reference','answer','en','sw'):
    if isinstance(r.get(k),str):texts.add(key(r[k]));texts.add(key(r[k].split('\n\n')[-1]))
 oldmeta=json.loads((ROOT/'work/module_lr077_v1/EVALUATION_DATA_FROZEN.json').read_text());oldrows=rows(oldmeta['data_path']);assert sha(oldmeta['data_path'])==oldmeta['data_sha256']
 for r in oldrows:used.add(r['id']);groups.add(r['group_id']);texts.add(key(r['prompt']));texts.add(key(r.get('source_text','')))
 print('prior exclusion',len(prior),len(used),flush=True)
 recipe=H/'reproductions/sw_recipe_fft_20261010/data/aya80_mt20';manifest=json.loads((recipe/'manifest.json').read_text());assert sha(manifest['source_file'])==manifest['source_sha256'];indices=set(json.loads((recipe/'indices.json').read_text()));maintexts=[]
 with open(manifest['source_file']) as f:
  for i,line in enumerate(f):
   if i not in indices:continue
   r=json.loads(line);used.add(r['id'])
   for k in ('prompt','answer'):
    s=r.get(k,'')
    if s:maintexts.append(norm(s));texts.add(key(s));texts.add(key(s.split('\n\n')[-1]))
 for split in ('train','dev','diagnostic'):
  for r in rows(OLD/'data'/f'{split}.jsonl'):
   used.add(r['id'])
   for k in ('en','sw'):
    if r.get(k):maintexts.append(norm(r[k]));texts.add(key(r[k]))
 sources={};result=[];audits={}
 # FLORES uses document isolation including all formal-devtest URLs.
 raw=OLD.parent/'paired_capability_sparse_v1/data/flores_source';en=(raw/'eng_Latn.dev').read_text().splitlines();sw=(raw/'swh_Latn.dev').read_text().splitlines();meta=list(csv.DictReader((raw/'metadata_dev.tsv').open(),delimiter='\t'));test_urls={x['URL'] for x in csv.DictReader((raw/'metadata_devtest.tsv').open(),delimiter='\t')}
 assert len(en)==len(sw)==len(meta)==997
 blocked=set(test_urls)|{x['URL'] for x in meta[:32]}
 for i,(a,b,m) in enumerate(zip(en,sw,meta)):
  gid='flores200:dev:document:'+hashlib.sha256(m['URL'].encode()).hexdigest()
  if gid in groups or key(a) in texts or key(b) in texts or any(f'flores200:dev:{i+1}:{t}' in used for t in ('en_to_sw','sw_to_en')):blocked.add(m['URL'])
 pool=[norm(s) for pair in zip(en,sw) for s in pair];vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,dtype=np.float32);mat=vec.fit_transform(maintexts+pool);scores=[]
 for start in range(0,len(pool),64):scores.extend(np.asarray((mat[len(maintexts)+start:len(maintexts)+start+64]@mat[:len(maintexts)].T).max(axis=1).toarray()).ravel().tolist())
 del mat
 for i,m in enumerate(meta):
  if max(scores[2*i:2*i+2])>=.82:blocked.add(m['URL'])
 docs=collections.defaultdict(list)
 for i,m in enumerate(meta):
  if m['URL'] not in blocked:docs[m['URL']].append(i)
 print('remaining FLORES documents',len(docs),flush=True)
 translation_available=len(docs)>=128
 for url in (sorted(docs,key=rank)[:128] if translation_available else []):
  i=min(docs[url],key=lambda i:rank('sentence:'+str(i)));gid='flores200:dev:document:'+hashlib.sha256(url.encode()).hexdigest()
  for task,source,reference,instruction in [('en_to_sw',en[i],sw[i],'Tafsiri maandishi yafuatayo kwenda Kiswahili:'),('sw_to_en',sw[i],en[i],'Tafsiri maandishi yafuatayo kwenda Kiingereza:')]:result.append({'id':f'flores200:dev:{i+1}:{task}','pair_id':f'flores200:dev:{i+1}','group_id':gid,'task':task,'prompt':instruction+'\n\n'+source,'source_text':source,'reference':reference,'study_partition':'translation','score_eligible':True})
 audits['translation']={'remaining_documents':len(docs),'selected_documents':128 if translation_available else 0,'requested_documents':128,'shortfall':not translation_available,'decision':'No source substitution or formal-test access; missing primary confirmation prevents upgrade','main_near_threshold':.82,'formal_sentence_text_read':False}
 for name in ('eng_Latn.dev','swh_Latn.dev','metadata_dev.tsv','metadata_devtest.tsv'):sources[str(raw/name)]=sha(raw/name)
 # Remaining source-disjoint KenSwQuAD and RACE train-side proxies; no official test sampling.
 src=H/'data/sw_d2_next_dev_20261008/sources';rp=src/'race-all-train-e37ee6f.parquet';race=pq.read_table(rp).to_pylist();sm=json.loads((H/'data/sw_reading_kenswquad_20261009/manifest.json').read_text());sp=pathlib.Path(sm['source_path']);swr=pq.read_table(sp).to_pylist();sources[str(rp)]=sha(rp);sources[str(sp)]=sha(sp);assert sha(sp)==sm['source_sha256']
 blocked_read=[r['article'] for r in race if str(r['example_id']) in articles or 'race:'+str(r['example_id']) in groups or key(r['article']) in texts]+[r['context'] for r in swr if str(r['story_id']) in stories or 'kenswquad:'+str(r['story_id']) in groups or key(r['context']) in texts]
 blocked_fields=maintexts+list(map(norm,blocked_read));candidates={'en_reading':{},'sw_reading':{}}
 for i,r in enumerate(race):
  g=str(r['example_id']);qid=f'RACE:train:{g}:source-row-{i}';a=r['article'];options=r['options']
  if g in articles or 'race:'+g in groups or qid in used or key(a) in texts or len(options)!=4 or r['answer'] not in 'ABCD':continue
  if any(re.search(r'(^|\s)[A-G][.)]\s',x) for x in options):continue
  if 'johnson' in norm(a) and 'arkansas' in norm(a):continue
  prompt='Read the passage and choose the correct answer. Reply with the option letter only.\n\nPassage:\n'+a+'\n\nQuestion: '+r['question']+'\n'+'\n'.join(k+'. '+v for k,v in zip('ABCD',options))
  if len(tok.encode(prompt,add_special_tokens=False))>2048:continue
  row={'id':qid,'group_id':'race:'+g,'task':'en_reading','study_partition':'en_reading','prompt':prompt,'reference':r['answer'],'options':list(options),'source_text':a,'score_eligible':True}
  if g not in candidates['en_reading'] or rank(qid)<rank(candidates['en_reading'][g]['id']):candidates['en_reading'][g]=row
 for r in swr:
  g=str(r['story_id']);context=r['context'].strip();q=r['question'].strip();answer=r['answer'].strip()
  if g in stories or 'kenswquad:'+g in groups or r['id'] in used or key(context) in texts or key(q) in texts:continue
  if not answer or len(answer)>80 or norm(answer) not in norm(context) or len(tok.encode(context,add_special_tokens=False))>1024:continue
  if 'umuhimu wa elimu' in norm(context) and 'mfumo wa mafunzo' in norm(context):continue
  row={'id':r['id'],'group_id':'kenswquad:'+g,'task':'sw_reading','study_partition':'sw_reading','prompt':'Soma kifungu kifuatacho kisha jibu swali kwa maneno machache kutoka kwenye kifungu.\n\nKifungu:\n'+context+'\n\nSwali: '+q+'\nJibu:','reference':answer,'source_text':context,'score_eligible':True}
  if g not in candidates['sw_reading'] or rank(r['id'])<rank(candidates['sw_reading'][g]['id']):candidates['sw_reading'][g]=row
 for task,bygroup in candidates.items():
  rr=sorted(bygroup.values(),key=lambda r:rank(r['group_id']))[:1000];print(task,'candidates',len(rr),flush=True);vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,dtype=np.float32);mat=vec.fit_transform(blocked_fields+[norm(r['source_text']) for r in rr]);cm=mat[len(blocked_fields):];bm=mat[:len(blocked_fields)];maxima=np.asarray((cm@bm.T).max(axis=1).toarray()).ravel();own=cm@cm.T;selected=[];ix=[]
  for i,r in enumerate(rr):
   if maxima[i]>=.85 or (ix and float(own[i,ix].max())>=.85):continue
   r['maximum_blocked_tfidf']=float(maxima[i]);selected.append(r);ix.append(i)
   if len(selected)==64:break
  assert len(selected)==64,(task,'unused sources insufficient',len(selected));result+=selected;audits[task]={'eligible_source_groups':len(bygroup),'near_threshold':.85,'max_selected_similarity':max(x['maximum_blocked_tfidf'] for x in selected)};del mat,cm,bm,own
 # Instructions: same source and same strict implementation, English persona prompts unused anywhere.
 formal=H/'data/sw-d2-eval-inputs/hf_cache/hub/datasets--google--IFEval/snapshots/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl'
 for r in rows(formal):texts.add(key(r['prompt']))
 scorer=H/'data/sw-d2-eval-inputs/project/ssu-main/evaluation/src/ifeval.py';sys.path.insert(0,str(scorer.parent));import ifeval
 _,_,registry=ifeval.load_official_ifeval_modules(scorer.parent/'utils');ip=src/'if-multi-train-7c1813b.parquet';sources[str(ip)]=sha(ip);options=[]
 for batch in pq.ParquetFile(ip).iter_batches(batch_size=1024):
  for r in batch.to_pylist():
   if not r['key'].startswith('personahub_') or r['key'] in used or len(r['messages'])!=1 or r['messages'][0]['role']!='user':continue
   text=r['messages'][0]['content']
   if not 80<=len(text)<=4000 or key(text) in texts:continue
   try:
    ground=ast.literal_eval(r['ground_truth']);assert len(ground)==1;ids=ground[0]['instruction_id'];kwargs=ground[0]['kwargs'];assert ids and len(ids)==len(kwargs) and all(x in registry.INSTRUCTION_DICT for x in ids);assert all(x is None or isinstance(x,dict) for x in kwargs)
   except (ValueError,SyntaxError,TypeError,KeyError,AssertionError):continue
   options.append({'id':r['key'],'group_id':'ifmulti:'+key(text),'task':'en_instruction','study_partition':'en_instruction','prompt':text,'instruction_id_list':ids,'kwargs':[x or {} for x in kwargs],'source_text':text,'score_eligible':True})
 selected=[];seen_instruction=set()
 for r in sorted(options,key=lambda r:rank(r['id'])):
  if r['group_id'] in seen_instruction:continue
  try:
   if detect(r['prompt'])!='en':continue
   for n,kw in zip(r['instruction_id_list'],r['kwargs']):
    check=registry.INSTRUCTION_DICT[n](n);check.build_description(**{k:v for k,v in kw.items() if v is not None})
    if 'prompt' in (check.get_instruction_args() or []):check.build_description(prompt=r['prompt'])
    assert isinstance(check.check_following('Acknowledged.'),bool)
  except Exception:continue
  selected.append(r);seen_instruction.add(r['group_id'])
  if len(selected)==256:break
 # Lexical near screening of pre-ranked instructions; no output-based screening.
 assert len(selected)>=128;blocked_instruction=[r['prompt'] for r in oldrows if r['task']=='en_instruction']+maintexts+[r['prompt'] for r in rows(formal)]
 for entry in prior:
  if 'en_instruction' in entry['path']:blocked_instruction.extend(r['prompt'] for r in rows(entry['path']) if isinstance(r,dict) and 'prompt' in r)
 vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,dtype=np.float32);mat=vec.fit_transform(list(map(norm,blocked_instruction))+[norm(r['prompt']) for r in selected]);sims=np.asarray((mat[len(blocked_instruction):]@mat[:len(blocked_instruction)].T).max(axis=1).toarray()).ravel();accepted=[]
 for r,v in zip(selected,sims):
  if v<.85:r['maximum_blocked_tfidf']=float(v);accepted.append(r)
  if len(accepted)==128:break
 assert len(accepted)==128,('instruction near-screen shortage',len(accepted));result+=accepted;audits['en_instruction']={'near_threshold':.85,'max_selected_similarity':max(r['maximum_blocked_tfidf'] for r in accepted)}
 assert len(result)==(512 if translation_available else 256) and not any(r['id'] in used or r['group_id'] in groups for r in result)
 path=RUN/'confirmation.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result));frozen={'frozen_unix':time.time(),'data_path':str(path),'data_sha256':sha(path),'rows':len(result),'primary_confirmation_available':translation_available,'task_counts':dict(collections.Counter(r['task'] for r in result)),'selection_seed':SELECTION,'sources':sources,'exclusion_inventory':prior,'full_main_manifest_sha256':sha(recipe/'manifest.json'),'audits':audits,'caps':oldmeta['caps'],'batch':4,'scorer_hashes':{n:sha(REPORT/'code'/n) for n in ['reading_scoring.py','aux_scoring.py']},'no_model_scores_used':True,'not_formal_test':True,'limits':['Source-ID and lexical near screening cannot prove semantic decontamination','Train-side reading/instruction proxies and official FLORES dev, not formal test','Instruction checkers accepted by previous finite acceptance; no fresh human SW gold']};write(RUN/'CONFIRMATION_FROZEN.json',frozen);write(REPORT/'CONFIRMATION_AUDIT.json',{k:v for k,v in frozen.items() if k not in ('data_path','exclusion_inventory')})
 print('CONFIRMATION_FROZEN',frozen['task_counts'],flush=True)
if __name__=='__main__':main()
