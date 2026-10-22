"""Predeclared source-paired comparisons. No tuning and no pooled task score."""
from settings import *
import numpy as np,torch,torch.nn.functional as F,collections,csv,math
from sacrebleu.metrics import CHRF

def cluster_samples(groups):
 unique=sorted(set(groups));ix={g:i for i,g in enumerate(unique)};gi=np.array([ix[g] for g in groups]);weights=np.random.default_rng(42).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=10000);return unique,gi,weights

def mean_ci(a,b,groups,scale=100.):
 a=np.asarray(a);b=np.asarray(b);unique,gi,w=cluster_samples(groups);s=np.bincount(gi,weights=a-b,minlength=len(unique));n=np.bincount(gi,minlength=len(unique));samples=(w@s)/(w@n)*scale;ci=np.quantile(samples,[.025,.975]);return {'difference':float((a-b).mean()*scale),'low':float(ci[0]),'high':float(ci[1]),'n':len(a),'source_groups':len(unique),'paired':True}

def corpus_score(stats):return CHRF(char_order=6,word_order=2,beta=2)._compute_score_from_stats(np.asarray(stats).sum(0).tolist()).score

def chrf_ci(a,b,groups):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);unique,gi,w=cluster_samples(groups);ga=np.zeros((len(unique),a.shape[1]));gb=np.zeros_like(ga);np.add.at(ga,gi,a);np.add.at(gb,gi,b);sa=w@ga;sb=w@gb;metric=CHRF(char_order=6,word_order=2,beta=2);delta=np.array([metric._compute_score_from_stats(x.tolist()).score-metric._compute_score_from_stats(y.tolist()).score for x,y in zip(sa,sb)]);lo,hi=np.quantile(delta,[.025,.975]);return {'difference':corpus_score(a)-corpus_score(b),'low':float(lo),'high':float(hi),'n':len(a),'source_groups':len(unique),'paired':True}


def main():
 torch.set_num_threads(4);names=['Base','S-All','S-Attn','S-MLP'];tasks=['sw_reading','en_reading','en_to_sw','sw_to_en','en_instruction'];records={};result={'models':{},'comparisons':{},'exploratory_one_seed':True,'not_formal_test':True,'math_excluded':True};flat=[];cost={};traces={}
 for name in names:
  folder=RUN/'evaluation'/name;done=json.loads((folder/'DONE.json').read_text());assert sha(folder/'PREDICTIONS.jsonl')==done['predictions_sha256'];rr=rows(folder/'PREDICTIONS.jsonl');assert len(rr)==512;records[name]=[r for r in rr if r['score_eligible']];result['models'][name]={}
  if name!='Base':
   out=OUT/name;train=json.loads((out/'TRAINING_DONE.json').read_text());assert train['K']==77070336 and train['steps']==4293;logs=rows(out/'STEPS.jsonl');assert [r['step'] for r in logs]==list(range(1,4294));assert all(r['selected']==77070336 for r in logs);assert [r['step'] for r in logs if r['mask_refresh']]==list(range(1,4294,20));traces[name]=logs;cost[name]={'training':train,'evaluation':done,'midpoint_receipt_sha256':sha(out/'checkpoint_002146/RECOVERABLE.json'),'endpoint_receipt_sha256':sha(out/'checkpoint_004293/RECOVERABLE.json')}
  for task in tasks:
   rs=[r for r in records[name] if r['task']==task];assert len(rs)=={'sw_reading':64,'en_reading':63,'en_to_sw':128,'sw_to_en':128,'en_instruction':128}[task];field={'sw_reading':'content_f1','en_reading':'content_correct','en_instruction':'prompt_strict'}.get(task);metric=corpus_score([r['scores']['chrfpp_stats'] for r in rs]) if field is None else 100*np.mean([r['scores'][field] for r in rs]);value={'score':float(metric),'n':len(rs),'source_groups':len(set(r['group_id'] for r in rs)),'metric':'corpus_chrF++' if field is None else field,'capped':sum(r['capped'] for r in rs),'generated_tokens':sum(r['generated_tokens'] for r in rs),'mean_repeat_4gram_fraction':float(np.mean([r['behavior']['repeated_4gram_fraction'] for r in rs])),'languages':dict(collections.Counter(r['language_diagnostic'] for r in rs))}
   if task=='en_reading':value['format_strict']=100*np.mean([r['scores']['format_correct'] for r in rs])
   result['models'][name][task]=value;flat.append({'model':name,'task':task,**{k:v for k,v in value.items() if k!='languages'}})
 pairs=[('S-Attn','S-All'),('S-MLP','S-All'),('S-Attn','S-MLP')]+[(name,'Base') for name in names if name!='Base']
 private_pairs=[]
 for left,right in pairs:
  key=left+'-minus-'+right;result['comparisons'][key]={}
  for task in tasks:
   aa=[r for r in records[left] if r['task']==task];bb=[r for r in records[right] if r['task']==task];assert [r['id'] for r in aa]==[r['id'] for r in bb];groups=[r['group_id'] for r in aa];field={'sw_reading':'content_f1','en_reading':'content_correct','en_instruction':'prompt_strict'}.get(task);value=chrf_ci([r['scores']['chrfpp_stats'] for r in aa],[r['scores']['chrfpp_stats'] for r in bb],groups) if field is None else mean_ci([r['scores'][field] for r in aa],[r['scores'][field] for r in bb],groups);result['comparisons'][key][task]=value
   for a,b in zip(aa,bb):private_pairs.append({'comparison':key,'task':task,'id':a['id'],'group_id':a['group_id'],'left_text':a['text'],'right_text':b['text'],'left_scores':a['scores'],'right_scores':b['scores'],'left_capped':a['capped'],'right_capped':b['capped'],'left_language':a['language_diagnostic'],'right_language':b['language_diagnostic']})
 # Exact common prefix/LR/counts, all groups; excludes genuine model-dependent loss/displacement.
 ref=traces['S-All']
 for name in ('S-Attn','S-MLP'):
  assert all(a['counts']==b['counts'] and a['lr']==b['lr'] and a['step']==b['step'] for a,b in zip(traces[name],ref))
 write(REPORT/'DOWNSTREAM_COMPARISON.json',result);write(REPORT/'TRAINING_COST_AND_DISPLACEMENT.json',cost);(RUN/'PAIRED_ITEM_COMPARISONS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in private_pairs))
 with (REPORT/'ENDPOINT_SCORES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in flat for k in r}));w.writeheader();w.writerows(flat)
 comparison=result['comparisons']['S-Attn-minus-S-All'];gates={'ENtoSW_learning_close':comparison['en_to_sw']['low']>=-1,'SWtoEN_learning_close':comparison['sw_to_en']['low']>=-1,'SWreading_learning_close':comparison['sw_reading']['low']>=-2,'ENreading_retention_better':comparison['en_reading']['low']>0,'ENinstruction_noninferior':comparison['en_instruction']['low']>=-2};base=cost['S-All']['training'];attn=cost['S-Attn']['training'];ratios={'cumulative_path_length':attn['cumulative_path_length']/base['cumulative_path_length'],'final_base_displacement_L2':attn['final_base_delta_L2']/base['final_base_delta_L2']};confound=any(v<.8 for v in ratios.values());passed=all(gates.values());decision={'gates':gates,'passed_task_screen':passed,'Attn_to_All_displacement_ratios':ratios,'displacement_confound_flag':confound,'decision':'Only discuss matched-displacement control before attributing module advantage' if passed and confound else 'Only discuss secondseed; no automatic subspace or layer expansion' if passed else 'No evidence meeting predeclared Attention-first screen; stop expansion and report negatives/uncertainty','all3_K_LR_maincounts_mask_refreshes_match':True,'interpretation':'CI failure can be uncertainty, not proof of damage/equivalence. Single seed, development proxies only; weights/SFTrecipe differ from historical D2 and dense alignment track.'};write(REPORT/'MANDATORY_DECISION.json',decision);print(json.dumps(decision),flush=True)
if __name__=='__main__':main()
