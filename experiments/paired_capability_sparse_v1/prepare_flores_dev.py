"""Replace noisy MT reserve with source-document grouped official FLORES dev."""
import csv,hashlib,json,shutil,tarfile
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from prepare_dev import ROOT,H,DATA,rows,norm,sha,digest
SOURCE=ROOT.parents[3]/'work/flores_metadata';OUT=ROOT/'data/dev_v4'
SEED='paired-capability-flores-dev-doc-v1'
def rank(s):return hashlib.sha256((SEED+':'+str(s)).encode()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError('No overwrite of dev versions')
 origin=json.loads((SOURCE/'SOURCE.json').read_text());archive=SOURCE/'flores200_dataset.tar.gz'
 if sha(archive)!=origin['archive_sha256']:raise RuntimeError('Official archive changed')
 raw=ROOT/'data/flores_source';raw.mkdir(exist_ok=False)
 for name in ('eng_Latn.dev','swh_Latn.dev','metadata_dev.tsv','SOURCE.json'):shutil.copyfile(SOURCE/name,raw/name)
 # Only document URLs from devtest metadata are used to enforce a document-level
 # boundary. No devtest sentence/answer text or model results are read here.
 with tarfile.open(archive) as tar:
  member=next(n for n in tar.getnames() if n.endswith('/metadata_devtest.tsv'))
  (raw/'metadata_devtest.tsv').write_bytes(tar.extractfile(member).read())
 en=(raw/'eng_Latn.dev').read_text().splitlines();sw=(raw/'swh_Latn.dev').read_text().splitlines();meta=list(csv.DictReader((raw/'metadata_dev.tsv').open(),delimiter='\t'))
 local_path=DATA/'sw-d2-eval-inputs/data/sw_fineweb2_aya_v1/eval_data/eng_Latn-swh_Latn_validation.jsonl';local=rows(local_path)
 if len(en)!=997 or len(sw)!=997 or len(meta)!=997 or len(local)!=997:raise RuntimeError('FLORES dev cardinality')
 if any(norm(a)!=norm(r['source_text']) or norm(b)!=norm(r['target_text']) for a,b,r in zip(en,sw,local)):raise RuntimeError('Original official EN/SW dev does not match locally locked validation split')
 expected='d959f5e7943169f022e04c582081433bd66d2ff939e7709d11abc4dd03b9b0e6'
 if sha(local_path)!=expected:raise RuntimeError('Historical protocol dev identity changed')
 old=ROOT/'data/dev_v3';manifest=json.loads((old/'manifest.json').read_text());used=set()
 for entry in manifest['prior_exclusion_files']:
  if sha(entry['path'])!=entry['sha256']:raise RuntimeError('Old pool changed')
  for r in rows(entry['path']):
   if not isinstance(r,dict):continue
   for key in ('prompt','source_text','reference','target_text','question','context','text'):
    value=r.get(key)
    if isinstance(value,str):used.add(norm(value));used.add(norm(value.split('\n\n')[-1]))
 for path in manifest['formal_exclusion_sha256']:
  for r in rows(path):
   for key in ('prompt','source_text','target_text','reference'):
    if isinstance(r.get(key),str):used.add(norm(r[key]))
 devtest_urls={r['URL'] for r in csv.DictReader((raw/'metadata_devtest.tsv').open(),delimiter='\t')}
 blocked_docs=set(devtest_urls)|{r['URL'] for r in meta[:32]}
 for a,b,m in zip(en,sw,meta):
  if norm(a) in used or norm(b) in used:blocked_docs.add(m['URL'])
 recipe=H/'reproductions/sw_recipe_fft_20261010/data/aya80_mt20';selected=set(json.loads((recipe/'indices.json').read_text()));main=[];main_ids=[]
 with open(json.loads((recipe/'manifest.json').read_text())['source_file']) as f:
  for i,line in enumerate(f):
   if i not in selected:continue
   r=json.loads(line)
   if r['task'] not in ('translation_en2target','translation_target2en'):continue
   for text in (r['prompt'].split('\n\n',1)[-1],r['answer']):main.append(norm(text));main_ids.append(r['id'])
 pool=[norm(s) for pair in zip(en,sw) for s in pair]
 vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,dtype=np.float32);mat=vec.fit_transform(main+pool);high=[];scores=[]
 for start in range(0,len(pool),128):
  sim=mat[len(main)+start:len(main)+start+128]@mat[:len(main)].T
  scores.extend(np.asarray(sim.max(axis=1).toarray()).ravel().tolist())
 for i,(a,b,m) in enumerate(zip(en,sw,meta)):
  score=max(scores[2*i],scores[2*i+1])
  if score>=.82:blocked_docs.add(m['URL']);high.append({'original_row':i,'URL':m['URL'],'score':score})
 docs=defaultdict(list)
 for i,m in enumerate(meta):
  if m['URL'] not in blocked_docs:docs[m['URL']].append(i)
 if len(docs)<64:raise RuntimeError('Fewer than64 independent FLORES development documents after exclusions')
 ordered=sorted(docs,key=rank);chosen=[min(docs[url],key=lambda i:rank('sentence:'+str(i))) for url in ordered[:128]]
 if len(chosen)<128:
  remaining=sorted([i for url in ordered for i in docs[url] if i not in chosen],key=lambda i:rank('extra:'+str(i)));chosen+=remaining[:128-len(chosen)]
 if len(chosen)!=128:raise RuntimeError('Not enough held-out sentences')
 translation=[]
 for i in chosen:
  group='flores200:dev:document:'+hashlib.sha256(meta[i]['URL'].encode()).hexdigest()
  for task,source,reference,instruction in [('en_to_sw',en[i],sw[i],'Tafsiri maandishi yafuatayo kwenda Kiswahili:'),('sw_to_en',sw[i],en[i],'Tafsiri maandishi yafuatayo kwenda Kiingereza:')]:
   translation.append({'id':f'flores200:dev:{i+1}:{task}','pair_id':f'flores200:dev:{i+1}','group_id':group,'source_document':meta[i]['URL'],'source_row_index':i,'task':task,'prompt':instruction+'\n\n'+source,'source_text':source,'reference':reference})
 OUT.mkdir()
 for task,entry in manifest['partitions'].items():
  p=OUT/(task+'.jsonl')
  if task=='translation':p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in translation))
  else:p.write_bytes(Path(entry['path']).read_bytes())
  rr=rows(p);entry.update(path=str(p),sha256=sha(p),rows=len(rr),ids=[r['id'] for r in rr])
 for p in raw.iterdir():manifest['sources'][str(p)]=sha(p)
 manifest['translation_origin']={'dataset':'official originalFLORES-200 dev, professionally translated; not NLLB-mined pairs','primary_readme':origin['primary_readme'],'archive_url':origin['official_public_url'],'archive_sha256':origin['archive_sha256'],'local_locked_validation_sha256':expected,'all997_local_EN_SW_rows_match_official_original':True,'formal_devtest_metadata':'URLs only for document exclusion; no new formal test scores','first32_dev_rows_and_all_their_documents_excluded':True,'all_prior_exact_sentence_documents_excluded':True,'main_neighbor_threshold_before_selection':.82,'main_neighbor_doc_flags':high,'eligible_documents':len(docs),'selected_documents':len({meta[i]['URL'] for i in chosen}),'selected_sentences':128,'blocked_document_count':len(blocked_docs),'code_sha256':sha(__file__),'no_model_scores_used':True,'other_task_bytes_unchanged':True}
 manifest['parent_manifest_sha256']=sha(old/'manifest.json');manifest['status']='frozen_inputs_before_any_new_model_score; final independent audit pending';manifest['selection']='source-document clustered MT bootstrap; one endpoint; no method/checkpoint/hyperparameter search'
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(json.dumps(manifest['translation_origin'],indent=2,ensure_ascii=False))
if __name__=='__main__':main()
