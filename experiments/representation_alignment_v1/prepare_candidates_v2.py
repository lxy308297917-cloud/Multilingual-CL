"""Deterministic content-only parallel data and original-schedule prefix, no model scores."""
import json,hashlib,re,unicodedata,collections
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext
ROOT=Path(__file__).resolve().parent; H=Path('/data0/users/lixiaoyang'); OUT=ROOT/'data_candidates_v2'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def rank(s):return hashlib.sha256(('representation-alignment-v1:42:'+s).encode()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
 OUT.mkdir(exist_ok=False)
 recipe=H/'reproductions/sw_recipe_fft_20261010/data/aya80_mt20'; manifest=json.loads((recipe/'manifest.json').read_text()); indices=json.loads((recipe/'indices.json').read_text()); source=Path(manifest['source_file'])
 assert sha(source)==manifest['source_sha256']; allrows=[]; groups=collections.defaultdict(dict)
 with source.open() as f:
  for line in f:
   r=json.loads(line);allrows.append(r)
   if r['task'] in ('translation_en2target','translation_target2en'):groups[r['group_key']][r['task']]=r
 selected={allrows[i]['group_key'] for i in indices if allrows[i]['task'].startswith('translation_')}
 n=0; tokens=0
 while tokens<manifest['input_tokens']*.1:
  tokens+=sum(len(allrows[i]['input_ids']) for i in indices[n:n+2]);n+=2
 save(OUT/'main_prefix.json',indices[:n]);save(OUT/'main_manifest.json',{'source':str(source),'source_sha256':sha(source),'recipe_manifest_sha256':sha(recipe/'manifest.json'),'indices_sha256':sha(recipe/'indices.json'),'steps':n//2,'full_schedule_steps':42465,'warmup_steps':2124,'input_tokens':tokens,'supervised_tokens':sum(sum(t!=-100 for t in allrows[i]['labels'][1:]) for i in indices[:n]),'main_prefix_sha256':sha(OUT/'main_prefix.json')})
 tok=AutoTokenizer.from_pretrained(H/'models/Qwen2.5-1.5B-Instruct',local_files_only=True); lid=fasttext.load_model(str(H/'models/glotlid/model_v3.bin'))
 rejected=collections.Counter(); candidates=[];devgroups=collections.defaultdict(dict)
 with (source.parent/'dev.jsonl').open() as f:
  for line in f:
   r=json.loads(line)
   if r['task'].startswith('translation_'):devgroups[r['group_key']][r['task']]=r
 # Ban exact content of existing development and fixed formal task files; near-ban applied below.
 forbidden=[]
 paths=list((ROOT.parent/'paired_capability_sparse_v1/data/dev_v4').glob('*.jsonl'))
 evalroot=H/'data/sw-d2-eval-inputs/data/sw_fineweb2_aya_v1/eval_data'
 paths+=list(evalroot.glob('*/test.jsonl'))
 for p in paths:
  for line in p.open():
   r=json.loads(line)
   for k in ('source_text','text','prompt','reference','answer','question','flores_passage'):
    if isinstance(r.get(k),str) and len(r[k])>15:forbidden.append(r[k])
 forbidnorm={norm(s) for s in forbidden}
 def extract(g,pair,origin):
  if len(pair)!=2:rejected['missing_reverse']+=1;return
  a=pair['translation_en2target'];b=pair['translation_target2en']; en=a['prompt'].split('\n\n',1)[-1].strip();sw=a['answer'].strip()
  if norm(b['answer'])!=norm(en) or norm(b['prompt'].split('\n\n',1)[-1])!=norm(sw):rejected['reverse_mismatch']+=1;return
  ei=tok.encode(en,add_special_tokens=False);si=tok.encode(sw,add_special_tokens=False)
  if not(8<=len(ei)<=192 and 8<=len(si)<=256) or min(len(en),len(sw))<35:rejected['length']+=1;return
  if norm(en)==norm(sw) or any(norm(x) in forbidnorm for x in (en,sw)):rejected['exact_forbidden']+=1;return
  if sorted(re.findall(r'\d+',en))!=sorted(re.findall(r'\d+',sw)):rejected['numeric_mismatch']+=1;return
  el,ep=lid.predict(en.replace('\n',' '),k=3);sl,sp=lid.predict(sw.replace('\n',' '),k=5)
  eprob=sum(float(p) for l,p in zip(el,ep) if l=='__label__eng_Latn');sprob=sum(float(p) for l,p in zip(sl,sp) if l in ('__label__swh_Latn','__label__swc_Latn'))
  if eprob<.8 or sprob<.8:rejected['language']+=1;return
  return {'id':g,'en':en,'sw':sw,'en_ids':ei,'sw_ids':si,'en_lid':eprob,'sw_lid':sprob,'source_split':origin,'source_ids':[a['id'],b['id']],'dedup_groups':[a.get('dedup_group'),b.get('dedup_group')],'ce_rows':[{'input_ids':x['input_ids'],'labels':x['labels']} for x in (a,b)]}
 for origin,gg in [('train',groups),('dev',devgroups)]:
  for g,p in sorted(gg.items(),key=lambda x:rank(x[0])):
   candidate_origin='unused_train' if origin=='train' and g not in selected else origin
   r=extract(g,p,candidate_origin)
   if r:candidates.append(r)
 print('QA candidates',collections.Counter(r['source_split'] for r in candidates),dict(rejected),flush=True)
 # Character 3-5gram cosine >= .80 exclusion (numeric masking) across splits, against formal/development and all main MT content.
 def near(s):return re.sub(r'\d+',' NUM ',norm(s))
 train=[r for r in candidates if r['source_split']=='train'];held=[r for r in candidates if r['source_split']!='train']
 allmain=[]
 for g in sorted(selected):
  p=groups[g]
  if 'translation_en2target' in p:
   a=p['translation_en2target'];allmain.extend([a['prompt'].split('\n\n',1)[-1],a['answer']])
 blocked=forbidden+allmain
 vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=150000,sublinear_tf=True,dtype=np.float32)
 texts=[near(s) for r in candidates for s in (r['en'],r['sw'])]+[near(s) for s in blocked];mat=vec.fit_transform(texts); cb=mat[2*len(candidates):]; ci={r['id']:i for i,r in enumerate(candidates)}
 cb_all_T=cb.T.tocsr();cb_formal_T=cb[:len(forbidden)].T.tocsr()
 print('TFIDF_READY',mat.shape,flush=True)
 def against_block(r,all_main):
  i=ci[r['id']];base=cb_all_T if all_main else cb_formal_T
  return float((mat[2*i:2*i+2]@base).max()) if base.shape[1] else 0.
 kept=[];seen=set()
 for r in train:
  if any(norm(r[k]) in seen for k in ('en','sw')):continue
  if against_block(r,False)>=.80:continue
  # Enforce near dedup within selected training pool with both language views.
  if kept:
   jj=[2*ci[x['id']]+j for x in kept for j in (0,1)];ii=2*ci[r['id']]
   if float((mat[ii:ii+2]@mat[jj].T).max())>=.80:continue
  kept.append(r);seen.update(norm(r[k]) for k in ('en','sw'))
  if len(kept)%200==0:print('TRAIN_SELECTED',len(kept),flush=True)
  if len(kept)==3500:break
 cleanheld=[]
 for r in held:
  risk=against_block(r,True)
  if risk>=.80:continue
  ii=2*ci[r['id']]
  if cleanheld:
   jj=[2*ci[x['id']]+j for x in cleanheld for j in (0,1)]
   if float((mat[ii:ii+2]@mat[jj].T).max())>=.80:continue
  r['nearest_forbidden_main_similarity']=risk;cleanheld.append(r)
  if len(cleanheld)%200==0:print('HELD_SELECTED',len(cleanheld),flush=True)
  if len(cleanheld)==2500:break
 assert len(kept)==3500 and len(cleanheld)>=2500,(len(kept),len(cleanheld))
 for name,rr in [('train',kept),('dev',cleanheld[:1000]),('diagnostic',cleanheld[1000:2500])]:
  with (OUT/(name+'.jsonl')).open('w') as f:
   for r in rr:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 save(OUT/'manifest.json',{'recipe':manifest,'source_sha256':sha(source),'dev_source_sha256':sha(source.parent/'dev.jsonl'),'splits':{k:{'pairs':n,'sha256':sha(OUT/(k+'.jsonl'))} for k,n in [('train',3500),('dev',1000),('diagnostic',1500)]},'filters':dict(rejected),'exclusion_files':{str(p):sha(p) for p in paths},'near_duplicate_rule':'numeric-masked character 3-5gram TFIDF cosine < .80, min_df2 max_features150k; all main MT excluded from heldout; within each and across dev/diagnostic dedup','quality_limits':'GlotLID+numeric/reverse consistency is not bilingual semantic verification; reviewed examples retained. swc+swh pooled language probability >= .8','train_is_replay_of_main':True})
 save(OUT/'review_samples.json',[{k:r[k] for k in ('id','en','sw')} for r in kept[:30]+cleanheld[:30]])
 print('DATA_READY',tokens,n//2,flush=True)
if __name__=='__main__':main()
