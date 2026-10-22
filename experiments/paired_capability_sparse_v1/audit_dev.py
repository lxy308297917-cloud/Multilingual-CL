"""Independent data-role, prior-pool, full-main-stream, and parallel-neighbor audit."""
import hashlib,json,re,unicodedata
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT=Path(__file__).resolve().parent;H=Path('/data0/users/lixiaoyang');DEV=ROOT/'data/dev_v4'
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def digest(s):return hashlib.sha256(norm(s).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def main():
 m=json.loads((DEV/'manifest.json').read_text());tasks={k:rows(v['path']) for k,v in m['partitions'].items()}
 for k,rr in tasks.items():
  assert len(rr)==m['partitions'][k]['rows'] and sha(m['partitions'][k]['path'])==m['partitions'][k]['sha256']
  assert [r['id'] for r in rr]==m['partitions'][k]['ids']
 for p,h in m['sources'].items():assert sha(p)==h,p
 for p,h in m['formal_exclusion_sha256'].items():assert sha(p)==h,p
 cal=rows(ROOT/'data/calibration_pairs.v1.jsonl');calids={r['id'] for r in cal};newids={r['id'] for rr in tasks.values() for r in rr};assert not calids&newids
 prior=rows(H/'data/sw-d2-inputs/sw_update_utility_v1/calibration_candidates.jsonl');priorids={r['id'] for r in prior}
 exact_prior=sorted(priorids&newids)
 old_articles={r['id'].split(':')[2] for r in prior if r['id'].startswith('RACE:train:')}
 new_articles={r['group_id'].removeprefix('race:') for r in tasks['en_reading']}
 old_article_overlap=sorted(old_articles&new_articles)
 # Compare normalized full passages against all old source article groups, including different question IDs.
 used_articles=set(old_articles)
 for entry in m['prior_exclusion_files']:
  for r in rows(entry['path']):
   qid=str(r.get('id',''))
   if qid.startswith('RACE:train:'):used_articles.add(qid.split(':')[2])
   if r.get('article_id'):used_articles.add(str(r['article_id']))
 racepath=H/'data/sw_d2_next_dev_20261008/sources/race-all-train-e37ee6f.parquet';old_passage_hashes=set()
 for batch in pq.ParquetFile(racepath).iter_batches(batch_size=1024):
  for r in batch.to_pylist():
   if r['example_id'] in used_articles:old_passage_hashes.add(digest(r['article']))
 formal_en=H/'data/sw-d2-eval-inputs/data/eval_datasets_local/belebele/eng_Latn/test.jsonl'
 assert formal_en.exists()
 for r in rows(formal_en):
  for key in ('flores_passage','passage','text'):
   if isinstance(r.get(key),str):old_passage_hashes.add(digest(r[key]))
 old_passage_overlap=[r['id'] for r in tasks['en_reading'] if digest(r['source_text']) in old_passage_hashes]
 # Whole-question/context containment against actual frozen main samples. Use a
 # token anchor only as a candidate accelerator; report a hit only after the
 # entire normalized query occurs, never from a shared phrase alone.
 queries=[]
 for r in cal:
  for lang in ('en','sw'):queries.append({'role':'calibration','id':r['id'],'text':norm(r['question_'+lang])})
 for task,rr in tasks.items():
  if task=='translation':continue
  for r in rr:queries.append({'role':'dev','id':r['id'],'text':norm(r.get('source_text',r['prompt']))})
 lookup=defaultdict(lambda:defaultdict(list))
 for q in queries:
  words=q['text'].split();n=min(8,len(words))
  if n:lookup[n][tuple(words[-n:])].append(q)
 recipe=H/'reproductions/sw_recipe_fft_20261010/data/aya80_mt20';manifest=json.loads((recipe/'manifest.json').read_text());indices=set(json.loads((recipe/'indices.json').read_text()))
 assert sha(manifest['source_file'])==manifest['source_sha256']
 overlaps=[];train_parallel=[];train_groups=set()
 with open(manifest['source_file']) as f:
  for i,line in enumerate(f):
   if i not in indices:continue
   r=json.loads(line);text=norm(r.get('prompt','')+' '+r.get('answer',''));words=text.split();seen=set()
   for n,table in lookup.items():
    for start in range(len(words)-n+1):
     for q in table.get(tuple(words[start:start+n]),()):
      key=(q['role'],q['id'])
      if key not in seen and q['text'] in text:
       overlaps.append({'role':q['role'],'id':q['id'],'main_id':r['id'],'main_index':i});seen.add(key)
   if r['task'] in ('translation_en2target','translation_target2en'):
    source=r['prompt'].split('\n\n',1)[-1];target=r['answer'];train_groups.add(r['group_key'])
    train_parallel.append({'id':r['id'],'group_id':r['group_key'],'source':source,'reference':target})
 groups={r['group_id'] for r in tasks['translation']};parallel_group_overlap=sorted(groups&train_groups)
 # High character-ngram similarity is a conservative review flag, not proof of
 # semantic identity. Both languages are checked independently of model scores.
 main_sentences=[];main_meta=[]
 for r in train_parallel:
  for side in ('source','reference'):main_sentences.append(norm(r[side]));main_meta.append((r['id'],side))
 dev_sentences=[norm(r['source_text']) for r in tasks['translation']]
 vector=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,dtype=np.float32)
 mat=vector.fit_transform(main_sentences+dev_sentences);sims=mat[len(main_sentences):]@mat[:len(main_sentences)].T
 scores=np.asarray(sims.max(axis=1).toarray()).ravel();neighbors=np.asarray(sims.argmax(axis=1)).ravel()
 parallel_near=[];nearest=[]
 for r,score,n in zip(tasks['translation'],scores,neighbors):
  item={'id':r['id'],'group_id':r['group_id'],'similarity':float(score),'main_id':main_meta[int(n)][0],'main_side':main_meta[int(n)][1]}
  if score>=.85:parallel_near.append(item)
  nearest.append(item)
 doc_gate=True
 if 'translation_origin' in m:
  urls={r['source_document'] for r in tasks['translation']}
  doc_gate=len(urls)==128 and len({r['pair_id'] for r in tasks['translation']})==128
  if not m['translation_origin']['all997_local_EN_SW_rows_match_official_original']:doc_gate=False
 gates={'parallel_source_document_grouping':doc_gate,'source_and_file_hashes':True,'calibration_dev_source_ids_disjoint':not(calids&newids),'old_calibration_ids_disjoint':not exact_prior,'prior_reading_articles_disjoint':not old_article_overlap,'prior_reading_passages_disjoint':not old_passage_overlap,'main_full_query_containment_absent':not overlaps,'parallel_main_group_ids_disjoint':not parallel_group_overlap,'parallel_high_similarity_absent':not parallel_near}
 report={'audit_code_sha256':sha(__file__),'formal_english_reading_sha256':sha(formal_en),'status':'passed' if all(gates.values()) else 'failed_before_inference','gates':gates,'dev_manifest_sha256':sha(DEV/'manifest.json'),'calibration_sha256':sha(ROOT/'data/calibration_pairs.v1.jsonl'),'old_calibration_id_overlap':exact_prior,'old_article_overlap':old_article_overlap,'old_passage_overlap':old_passage_overlap,'main_containment':overlaps,'parallel_group_overlap':parallel_group_overlap,'parallel_near_threshold':.85,'parallel_high_similarity':parallel_near,'parallel_top20_neighbors':sorted(nearest,key=lambda x:-x['similarity'])[:20],'actual_main_samples_scanned':len(indices),'limitations':['Character/TFIDF checks do not prove absence of all semantic paraphrases','KenSwQuAD and IF-multi are development proxies, not formal Belebele/IFEval','Shared task families do not establish source identity; only identified problem/paragraph/source groups are excluded'],'model_inference_or_scores':False}
 (DEV/'INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':report['status'],'gates':gates,'main_containment_hits':len(overlaps),'parallel_near':len(parallel_near),'old_passage_overlap':len(old_passage_overlap)},indent=2))
if __name__=='__main__':main()
