"""Fixed per-order and source-synchronous two-order analysis; missing primary never passes."""
from paired_statistics import *
TASKS=['en_to_sw','sw_to_en','sw_reading','en_reading','en_instruction']
FIELDS={'sw_reading':'content_f1','en_reading':'content_correct','en_instruction':'prompt_strict'}
def select(rr,t):return [r for r in rr if r['task']==t and r['score_eligible']]
def summary(rr):
 out={}
 for t in TASKS:
  rs=select(rr,t)
  if not rs:continue
  v={'n':len(rs),'score':float(corpus_score([r['scores']['chrfpp_stats'] for r in rs]) if t not in FIELDS else 100*np.mean([r['scores'][FIELDS[t]] for r in rs])),'capped':sum(r['capped'] for r in rs),'languages':dict(collections.Counter(r['language_diagnostic'] for r in rs)),'generated_tokens':sum(r['generated_tokens'] for r in rs),'mean_repeat_4gram_fraction':float(np.mean([r['behavior']['repeated_4gram_fraction'] for r in rs]))}
  if t=='en_reading':v['format_strict']=100*np.mean([r['scores']['format_correct'] for r in rs])
  out[t]=v
 return out
def compare(a,b):
 out={}
 for t in TASKS:
  aa=select(a,t);bb=select(b,t)
  if not aa or not bb:continue
  assert [r['id'] for r in aa]==[r['id'] for r in bb];g=[r['group_id'] for r in aa]
  out[t]=mean_ci([r['scores'][FIELDS[t]] for r in aa],[r['scores'][FIELDS[t]] for r in bb],g) if t in FIELDS else chrf_ci([r['scores']['chrfpp_stats'] for r in aa],[r['scores']['chrfpp_stats'] for r in bb],g)
 return out
def mean_orders(records,t):
 aa=[select(records['Mix50-s'+str(s)],t) for s in (42,43)];bb=[select(records['LR077-s'+str(s)],t) for s in (42,43)]
 if not all(aa+bb):return None
 ids=[r['id'] for r in aa[0]];assert all([r['id'] for r in rs]==ids for rs in aa+bb);g=[r['group_id'] for r in aa[0]];unique,gi,w=cluster_samples(g)
 if t in FIELDS:
  ds=np.mean([[x['scores'][FIELDS[t]]-y['scores'][FIELDS[t]] for x,y in zip(a,b)] for a,b in zip(aa,bb)],axis=0);d=np.bincount(gi,weights=ds,minlength=len(unique));n=np.bincount(gi,minlength=len(unique));samples=(w@d)/(w@n)*100;point=float(np.mean(ds)*100)
 else:
  metric=CHRF(char_order=6,word_order=2,beta=2);deltas=[];points=[]
  for a,b in zip(aa,bb):
   a=np.array([r['scores']['chrfpp_stats'] for r in a],dtype=float);b=np.array([r['scores']['chrfpp_stats'] for r in b],dtype=float);ga=np.zeros((len(unique),a.shape[1]));gb=np.zeros_like(ga);np.add.at(ga,gi,a);np.add.at(gb,gi,b);sa=w@ga;sb=w@gb;deltas.append(np.array([metric._compute_score_from_stats(x.tolist()).score-metric._compute_score_from_stats(y.tolist()).score for x,y in zip(sa,sb)]));points.append(corpus_score(a)-corpus_score(b))
  samples=np.mean(deltas,axis=0);point=float(np.mean(points))
 low,high=np.quantile(samples,[.025,.975]);return {'difference':point,'low':float(low),'high':float(high),'n_sources':len(unique),'orders':2,'not_two_times_independent_items':True,'fixed_models_item_uncertainty_not_seed_population':True}
def load(split,name):
 folder=RUN/'evaluation'/split/name
 if split=='old' and not folder.exists():folder=pathlib.Path(json.loads((RUN/'CACHE_FROZEN.json').read_text())[name]['path'])
 done=json.loads((folder/'DONE.json').read_text());assert sha(folder/'PREDICTIONS.jsonl')==done['predictions_sha256'];return rows(folder/'PREDICTIONS.jsonl')
def main():
 torch.set_num_threads(4);assert (RUN/'SERIES_DONE.json').exists();protocol=json.loads((REPORT/'PROTOCOL_FROZEN.json').read_text());allresults={};flat=[];paired=[]
 for split in ('old','confirmation'):
  names=['Base','LR077-s42','Attn-s42',*JOBS] if split=='old' else protocol['confirmation_models'];records={n:load(split,n) for n in names};result={'models':{n:summary(rs) for n,rs in records.items()},'comparisons':{},'mean_order_comparisons':{}}
  for s in (42,43):
   for left in ('Mix50','Attn'):
    a=left+'-s'+str(s);b='LR077-s'+str(s);key=a+'-minus-'+b;result['comparisons'][key]=compare(records[a],records[b]);lookup={r['id']:r for r in records[b]}
    for r in records[a]:
     if not r['score_eligible']:continue
     q=lookup[r['id']];paired.append({'split':split,'comparison':key,'id':r['id'],'group_id':r['group_id'],'task':r['task'],'left':r,'right':q})
  for t in TASKS:
   v=mean_orders(records,t)
   if v:result['mean_order_comparisons'][t]=v
  # Absolute English loss relative to same-split Base, not only relative to adapted baseline.
  result['English_vs_Base']={n:{t:v for t,v in compare(rs,records['Base']).items() if t in ('en_reading','en_instruction')} for n,rs in records.items() if n!='Base'}
  write(REPORT/(split.upper()+'_RESULTS.json'),result);allresults[split]=result
  for n,ts in result['models'].items():
   for t,v in ts.items():flat.append({'split':split,'model':n,'task':t,**{k:x for k,x in v.items() if k!='languages'}})
 with (REPORT/'ENDPOINT_SCORES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in flat for k in r}));w.writeheader();w.writerows(flat)
 (RUN/'PAIRED_ITEMS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in paired))
 # All64, same fixed subset, original cache vs one stronger prompt; no output-selected subgroup.
 ids=json.loads((RUN/'PROMPT_FROZEN.json').read_text())['ids'];prompt={'models':{},'gaps':{}};pred={}
 for n in protocol['prompt_models']:
  lookup={r['id']:r for r in load('old',n)};old=[lookup[i] for i in ids];new=load('prompt',n);assert [r['id'] for r in new]==ids;pred[n]={'old':old,'new':new};prompt['models'][n]={'old':summary(old),'new':summary(new),'new_minus_old':compare(new,old)}
 for version in ('old','new'):prompt['gaps'][version]=compare(pred['Attn-s42'][version],pred['LR077-s42'][version])['en_to_sw']
 # Source-synchronous paired difference of differences in corpuschrF++.
 g=[r['group_id'] for r in pred['Attn-s42']['old']];u,gi,w=cluster_samples(g);metric=CHRF(char_order=6,word_order=2,beta=2);draws={}
 for version in ('old','new'):
  vv={}
  for n in ('Attn-s42','LR077-s42'):
   a=np.array([r['scores']['chrfpp_stats'] for r in pred[n][version]],dtype=float);ga=np.zeros((len(u),a.shape[1]));np.add.at(ga,gi,a);vv[n]=np.array([metric._compute_score_from_stats(x.tolist()).score for x in w@ga])
  draws[version]=vv['Attn-s42']-vv['LR077-s42']
 d=draws['new']-draws['old'];lo,hi=np.quantile(d,[.025,.975]);prompt['gap_change_new_minus_old']={'difference':prompt['gaps']['new']['difference']-prompt['gaps']['old']['difference'],'low':float(lo),'high':float(hi),'n':64,'causal_decomposition':False};write(REPORT/'PROMPT_SENSITIVITY.json',prompt)
 conf=allresults['confirmation'];primary=protocol['primary_confirmation_available'];gates={}
 if primary:
  for s in (42,43):
   comp=conf['comparisons'][f'Mix50-s{s}-minus-LR077-s{s}'];gates[str(s)]={t:comp[t]['difference']>=bound for t,bound in [('en_to_sw',1),('sw_to_en',-1),('sw_reading',-2),('en_reading',-2),('en_instruction',-2)]}
  gates['mean_ENtoSW_CI_positive']=conf['mean_order_comparisons']['en_to_sw']['low']>0
 passed=primary and all(all(x.values()) if isinstance(x,dict) else x for x in gates.values());write(REPORT/'MANDATORY_DECISION.json',{'primary_new_translation_available':primary,'gates':gates,'passed':passed,'decision':'Evidence insufficient for original progression rule: fresh source-isolated translation confirmation unavailable' if not primary else 'Simple quota baseline warrants bounded further research; no novelty established' if passed else 'Stop expansion of current module-budget prototype; no stable predeclared joint benefit','no_automatic_new_training':True,'actual_strength_not_matched':True,'two_orders_not_training_seed_population':True})
 print('FROZEN ALL-SPLIT AND PROMPT ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':main()
