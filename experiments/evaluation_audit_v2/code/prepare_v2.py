"""Select grouped fresh train-side scorer acceptance and comparison before inference."""
import re,collections,unicodedata,time,sys
import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from settings import *
from scoring_v2 import numeric
SEED='qwen15b-v2-unused-source-grouped-20261022'
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def key(s):return hashlib.sha256(norm(s).encode()).hexdigest()
def rank(s):return key(SEED+':'+s)
def near(s):return re.sub(r'\d+(?:[.,]\d+)*',' NUMBER ',norm(s))
def excluded_family(s):
    s=norm(s)
    return ('john' in s and 'johnson' in s and 'arkansas' in s) or ('umuhimu wa elimu' in s and 'mfumo wa mafunzo' in s)
def main():
    assert not (RUN/'DATA_FROZEN.json').exists()
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    used=set();texts=set();articles=set();stories=set();prior=[];seenhash=set()
    paths=set((H/'data').glob('*/*.jsonl'))|set((H/'data').glob('*/*/*.jsonl'))
    for repo in (ROOT/'outputs').glob('Multilingual-CL-*'):
        paths.update(repo.glob('experiments/paired_capability_sparse_v1/data/**/*.jsonl'))
    paths.update((H/'Documents/Codex/2026-10-20').glob('**/gsm_bilingual_20261020/*.json'))
    paths.add(H/'data/sw-d2-inputs/sw_update_utility_v1/calibration_candidates.jsonl')
    paths.update((H/'data/sw-d2-eval-inputs/data').glob('**/test.jsonl'))
    for p in sorted(paths):
        if p.stat().st_size>25<<20:continue
        h=sha(p)
        if h in seenhash:continue
        seenhash.add(h)
        try:rr=json.loads(p.read_text()) if p.suffix=='.json' else rows(p)
        except (ValueError,UnicodeDecodeError):continue
        if not isinstance(rr,list):continue
        prior.append({'path':str(p),'sha256':h,'rows':len(rr)})
        for r in rr:
            if not isinstance(r,dict):continue
            qid=str(r.get('id',''));used.add(qid)
            if qid.startswith('RACE:train:'):articles.add(qid.split(':')[2])
            for k in ('article_id','example_id'):
                if r.get(k):articles.add(str(r[k]))
            if r.get('story_id'):stories.add(str(r['story_id']))
            if re.fullmatch(r'\d+_swa_\d+',qid):stories.add(qid.rsplit('_',1)[0])
            for k in ('prompt','question','context','text','source_text','question_en','question_sw','flores_passage'):
                if isinstance(r.get(k),str):texts.add(key(r[k]))
    print('prior exclusions loaded',len(prior),flush=True)
    # Full frozen main recipe, including beyond the short prefix, is excluded.
    recipe=H/'reproductions/sw_recipe_fft_20261010/data/aya80_mt20'
    manifest=json.loads((recipe/'manifest.json').read_text());assert sha(manifest['source_file'])==manifest['source_sha256']
    indices=set(json.loads((recipe/'indices.json').read_text()));maintexts=[]
    with open(manifest['source_file']) as f:
        for i,line in enumerate(f):
            if i not in indices:continue
            r=json.loads(line);used.add(str(r['id']))
            for k in ('prompt','answer'):
                s=r.get(k,'')
                if s:maintexts.append(norm(s));texts.add(key(s))
    for split in ('train','dev','diagnostic'):
        for r in rows(OLD/'representation_alignment_v1/data'/f'{split}.jsonl'):
            used.add(r['id'])
            for k in ('en','sw'):
                if r.get(k):maintexts.append(norm(r[k]));texts.add(key(r[k]))
    print('full main and paired fields loaded',len(maintexts),flush=True)
    src=H/'data/sw_d2_next_dev_20261008/sources';sources={}
    racepath=src/'race-all-train-e37ee6f.parquet';race=pq.read_table(racepath).to_pylist();sources[str(racepath)]=sha(racepath)
    swmanifest=json.loads((H/'data/sw_reading_kenswquad_20261009/manifest.json').read_text());swpath=pathlib.Path(swmanifest['source_path']);sources[str(swpath)]=sha(swpath);assert sources[str(swpath)]==swmanifest['source_sha256']
    sw=pq.read_table(swpath).to_pylist()
    reserve=json.loads((H/'data/sw_next_dev_reserve_20261009/manifest.json').read_text());mathpath=pathlib.Path(reserve['partitions']['en_math']['sources']['source']);sources[str(mathpath)]=sha(mathpath)
    gsm=list(Dataset.from_file(str(mathpath)));gsmtest=list(Dataset.from_file(str(mathpath.with_name('gsm8k-test.arrow'))))
    blocked_read=[r['article'] for r in race if r['example_id'] in articles or key(r['article']) in texts]
    blocked_read.extend(r['context'] for r in sw if str(r['story_id']) in stories or key(r['context']) in texts)
    # Query containment is checked against every selected main field and source text.
    mainblob='\n'.join(maintexts)
    candidates={'en_reading':{},'sw_reading':{}}
    for i,r in enumerate(race):
        group=str(r['example_id']);article=r['article'];qid=f'RACE:train:{group}:source-row-{i}'
        if group in articles or qid in used or key(article) in texts or excluded_family(article):continue
        prompt='Read the passage and choose the correct answer. Reply with the option letter only.\n\nPassage:\n'+article+'\n\nQuestion: '+r['question']+'\n'+'\n'.join(a+'. '+b for a,b in zip('ABCD',r['options']))
        row={'id':qid,'group_id':'race:'+group,'task':'en_reading','prompt':prompt,'reference':r['answer'],'options':list(r['options']),'source_text':article}
        if group not in candidates['en_reading'] or rank(qid)<rank(candidates['en_reading'][group]['id']):candidates['en_reading'][group]=row
    for r in sw:
        group=str(r['story_id']);context=r['context'].strip();answer=r['answer'].strip();question=r['question'].strip()
        if group in stories or r['id'] in used or any(key(s) in texts for s in (context,question)) or excluded_family(context):continue
        if not answer or len(answer)>80 or norm(answer) not in norm(context) or norm(context) in mainblob:continue
        if len(tok.encode(context,add_special_tokens=False))>1024:continue
        prompt='Soma kifungu kifuatacho kisha jibu swali kwa maneno machache kutoka kwenye kifungu.\n\nKifungu:\n'+context+'\n\nSwali: '+question+'\nJibu:'
        row={'id':r['id'],'group_id':'kenswquad:'+group,'task':'sw_reading','prompt':prompt,'reference':answer,'source_text':context}
        if group not in candidates['sw_reading'] or rank(row['id'])<rank(candidates['sw_reading'][group]['id']):candidates['sw_reading'][group]=row
    # Compare pre-ranked candidate passages to old passages and all main fields.
    selected={};near_audit={};blocked=maintexts+blocked_read
    for task,bygroup in candidates.items():
        print('near audit',task,len(bygroup),flush=True)
        rr=[]
        for r in sorted(bygroup.values(),key=lambda r:rank(r['group_id'])):
            if norm(r['source_text']) in mainblob:continue
            if task=='en_reading' and len(tok.encode(r['source_text'],add_special_tokens=False))>1400:continue
            rr.append(r)
            if len(rr)==350:break
        vector=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,dtype=np.float32,min_df=1)
        mat=vector.fit_transform([norm(x) for x in blocked]+[norm(r['source_text']) for r in rr]);bm=mat[:len(blocked)];cm=mat[len(blocked):]
        maxima=np.asarray((cm@bm.T).max(axis=1).toarray()).ravel();own=cm@cm.T;accepted=[];ids=[]
        for i,r in enumerate(rr):
            if maxima[i]>=.85 or (ids and float(own[i,ids].max())>=.85):continue
            r['maximum_blocked_tfidf']=float(maxima[i]);accepted.append(r);ids.append(i)
            if len(accepted)==76:break
        assert len(accepted)==76,(task,len(accepted));selected[task]=accepted
        near_audit[task]={'candidates':len(rr),'threshold':.85,'blocked_fields':len(blocked),'max_selected':max(r['maximum_blocked_tfidf'] for r in accepted)}
        del mat,bm,cm,own
        print('selected',task,flush=True)
    blocked_indices={int(q.split(':')[1]) for q in used if re.fullmatch(r'gsm8k_train:\d+',q)}
    blockedmath=[r['question'] for r in gsmtest]+[gsm[i]['question'] for i in blocked_indices if i<len(gsm)]
    # Full main fields also participate in numeric-masked near-template exclusion.
    blockedmath.extend(maintexts)
    vec=TfidfVectorizer(ngram_range=(1,2),stop_words='english',sublinear_tf=True,dtype=np.float32)
    print('math near audit',len(blockedmath),flush=True)
    mat=vec.fit_transform([near(r['question']) for r in gsm]+[near(s) for s in blockedmath]);gm=mat[:len(gsm)];bm=mat[len(gsm):]
    accepted=[];selectedidx=[]
    for i in sorted(range(len(gsm)),key=lambda i:rank('gsm8k_train:'+str(i))):
        if i in blocked_indices:continue
        r=gsm[i];gold=r['answer'].rsplit('####',1)[-1].strip();qid='gsm8k_train:'+str(i)
        if len(r['question'])>650 or numeric(gold)[0] is None or norm(r['question']) in mainblob:continue
        sim=float((gm[i]@bm.T).max())
        if sim>=.72 or (selectedidx and float((gm[i]@gm[selectedidx].T).max())>=.72):continue
        accepted.append({'id':qid,'group_id':qid,'task':'en_math','prompt':r['question'],'reference':gold,'source_worked_answer':r['answer'],'system':'Solve the problem. Reason briefly in English and put the final numeric answer in \\boxed{...}.','maximum_blocked_tfidf':sim});selectedidx.append(i)
        if len(accepted)==76:break
    assert len(accepted)==76,len(accepted);selected['en_math']=accepted;near_audit['en_math']={'threshold':.72,'blocked_questions_and_main_fields':len(blockedmath),'max_selected':max(r['maximum_blocked_tfidf'] for r in accepted)}
    parts={}
    for role in ('acceptance','comparison'):
        rr=[r for task in ('sw_reading','en_reading','en_math') for r in (selected[task][:12] if role=='acceptance' else selected[task][12:])]
        path=RUN/f'{role}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr));parts[role]={'path':str(path),'sha256':sha(path),'rows':len(rr),'task_counts':dict(collections.Counter(r['task'] for r in rr)),'groups':[r['group_id'] for r in rr]}
    assert set(parts['acceptance']['groups']).isdisjoint(parts['comparison']['groups'])
    write(RUN/'DATA_FROZEN.json',{'seed':SEED,'frozen_unix':time.time(),'model_scores_accessed':False,'partitions':parts,'sources':sources,'prior_exclusion_files':prior,'full_main_samples':len(indices),'main_source_sha256':manifest['source_sha256'],'near_checks':near_audit,'limits':['Lexical and source-ID checks cannot establish absence of all semantic paraphrases','Train-side development proxies, not formal tests','Scorer acceptance uses distinct 36 source groups and controlled variants; comparison uses192 new groups']})
    write(REPORT/'DATA_AGGREGATE_AUDIT.json',{'data_manifest_sha256':sha(RUN/'DATA_FROZEN.json'),'acceptance_rows':36,'comparison_rows':192,'same_source_role_overlap':0,'sources':sources,'near_checks':near_audit,'full_main_samples':len(indices),'prior_exclusion_unique_files':len(prior),'scores_used_for_selection':False,'no_formal_test':True})
    print(json.dumps({'status':'DATA_FROZEN','rows':{k:v['rows'] for k,v in parts.items()},'near_checks':near_audit}))
if __name__=='__main__':main()
