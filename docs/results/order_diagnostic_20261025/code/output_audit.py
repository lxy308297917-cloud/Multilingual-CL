from settings import *
import collections,random,re
import numpy as np
sys.path.insert(0,str(PRIOR/'code'))
from reading_scoring import tokens,sw_score,behavior
meta=read(RUN/'CONFIRMATION_FROZEN.json');data=rows(meta['data_path']);rr={r['id']:r for r in data};ends={}
for seed in ('42','43'):
 p=PRUN/'evaluation/confirmation'/f'LR077-s{seed}';done=read(p/'DONE.json');assert sha(p/'PREDICTIONS.jsonl')==done['predictions_sha256'];assert done['data_sha256']==meta['data_sha256'];pred=rows(p/'PREDICTIONS.jsonl');assert len(pred)==256;ends[seed]={r['id']:r for r in pred}
 for r in pred:
  if r['study_partition']=='sw_reading':assert r['scores']==sw_score(r['text'],rr[r['id']]['reference']) and r['behavior']==behavior(r['text'])

def measures(pred):
 acc=[]
 for p in pred:
  a,b=tokens(re.sub(r'^\s*(?:jibu|answer)\s*:\s*','',p['text'],flags=re.I)),tokens(rr[p['id']]['reference']);overlap=sum((collections.Counter(a)&collections.Counter(b)).values())
  acc.append({'overlap':overlap,'words':len(a),'gold_words':len(b),'precision':overlap/max(1,len(a)),'recall':overlap/max(1,len(b)),'empty':not p['text'].strip(),'no_lexical_overlap':overlap==0})
 return {'n':len(pred),'F1':100*float(np.mean([r['scores']['content_f1'] for r in pred])),'exact':sum(r['scores']['content_exact'] for r in pred),'generated_token_mean':float(np.mean([r['generated_tokens'] for r in pred])),'generated_token_median':float(np.median([r['generated_tokens'] for r in pred])),'words_mean':float(np.mean([r['words'] for r in acc])),'capped':sum(r['capped'] for r in pred),'empty':sum(r['empty'] for r in acc),'zero_lexical_overlap':sum(r['no_lexical_overlap'] for r in acc),'language_counts':dict(collections.Counter(r['language_diagnostic'] for r in pred)),'repeated_4gram_fraction_mean':float(np.mean([r['behavior']['repeated_4gram_fraction'] for r in pred])),'repeat_fraction_positive_count':sum(r['behavior']['repeated_4gram_fraction']>0 for r in pred),'precision_mean':float(np.mean([r['precision'] for r in acc])),'recall_mean':float(np.mean([r['recall'] for r in acc])),'off_topic':'Not automatically inferred from zero overlap; requires content adjudication; assistant review auxiliary only'}
ids=[r['id'] for r in data if r['study_partition']=='sw_reading'];summary={s:measures([ends[s][i] for i in ids]) for s in ends}
paired=[]
for i in ids:
 a,b=ends['42'][i],ends['43'][i];paired.append({'id':i,'group_id':rr[i]['group_id'],'difference_F1_42_minus43':100*(a['scores']['content_f1']-b['scores']['content_f1']),'generated_tokens42':a['generated_tokens'],'generated_tokens43':b['generated_tokens'],'capped42':a['capped'],'capped43':b['capped'],'same_text':a['text']==b['text']})
selected=read(REPORT/'PROTOCOL_FROZEN.json')['case_ids'];mapping={'A':'43','B':'42'};write(RUN/'BLIND_KEY_PRIVATE.json',mapping)
cases=[]
for i in selected:
 r=rr[i];cases.append({'id':i,'prompt':r['prompt'],'reference':r['reference'],'A':ends[mapping['A']][i]['text'],'B':ends[mapping['B']][i]['text']})
write(RUN/'BLIND_CASES_PRIVATE.json',cases);write(RUN/'PAIRED_SW_DETAILS_PRIVATE.json',paired)
result={'cached_endpoint_hashes_verified':True,'all64_SW_scores_recomputed_identically':True,'SW_endpoint_behavior':summary,'paired_gain_loss':{'seed42_higher':sum(r['difference_F1_42_minus43']>0 for r in paired),'seed43_higher':sum(r['difference_F1_42_minus43']<0 for r in paired),'ties':sum(r['difference_F1_42_minus43']==0 for r in paired),'identical_text':sum(r['same_text'] for r in paired)},'blind_sample_ids_frozen_before_review':selected,'no_scoring_changes':True,'sample_size':64,'limits':'Language detector unreliable on short answers; F1 overlap is not full semantic equivalence or reading correctness; output-generated subsets descriptive, no causal decomposition.'}
write(REPORT/'SW_OUTPUT_AUDIT.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
