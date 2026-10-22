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

def downstream():
 allrec={}
 for name in ('Base','B','D0','D1'):
  folder=RUN/'evaluation'/name;done=json.loads((folder/'DONE.json').read_text());assert sha(folder/'PREDICTIONS.jsonl')==done['predictions_sha256'];rr=rows(folder/'PREDICTIONS.jsonl');assert len(rr)==512;allrec[name]=[r for r in rr if r['score_eligible']]
 result={'models':{},'comparisons':{},'not_formal_test':True,'math_excluded':True};flat=[]
 for name,rr in allrec.items():
  result['models'][name]={}
  for task in ('sw_reading','en_reading','en_to_sw','sw_to_en','en_instruction'):
   rs=[r for r in rr if r['task']==task];assert len(rs)=={'sw_reading':64,'en_reading':63,'en_to_sw':128,'sw_to_en':128,'en_instruction':128}[task]
   field={'sw_reading':'content_f1','en_reading':'content_correct','en_instruction':'prompt_strict'}.get(task);metric=corpus_score([r['scores']['chrfpp_stats'] for r in rs]) if field is None else 100*np.mean([r['scores'][field] for r in rs]);record={'score':float(metric),'n':len(rs),'source_groups':len(set(r['group_id'] for r in rs)),'metric':'corpus_chrF++' if field is None else field,'capped':sum(r['capped'] for r in rs),'generated_tokens':sum(r['generated_tokens'] for r in rs),'mean_repeat_4gram_fraction':float(np.mean([r['behavior']['repeated_4gram_fraction'] for r in rs])),'languages':dict(collections.Counter(r['language_diagnostic'] for r in rs)),'score_status_counts':dict(collections.Counter(r['scores'].get('status','strict_or_translation') for r in rs))}
   if task=='en_reading':record['format_strict']=100*np.mean([r['scores']['format_correct'] for r in rs])
   result['models'][name][task]=record;flat.append({'model':name,'task':task,**{k:v for k,v in record.items() if k not in ('languages','score_status_counts')}})
 for control in ('B','D0'):
  result['comparisons']['D1-minus-'+control]={}
  for task in ('sw_reading','en_reading','en_to_sw','sw_to_en','en_instruction'):
   aa=[r for r in allrec['D1'] if r['task']==task];bb=[r for r in allrec[control] if r['task']==task];assert [r['id'] for r in aa]==[r['id'] for r in bb];groups=[r['group_id'] for r in aa];field={'sw_reading':'content_f1','en_reading':'content_correct','en_instruction':'prompt_strict'}.get(task)
   value=chrf_ci([r['scores']['chrfpp_stats'] for r in aa],[r['scores']['chrfpp_stats'] for r in bb],groups) if field is None else mean_ci([r['scores'][field] for r in aa],[r['scores'][field] for r in bb],groups);result['comparisons']['D1-minus-'+control][task]=value
 write(REPORT/'DOWNSTREAM_COMPARISON.json',result)
 with (REPORT/'ENDPOINT_SCORES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in flat for k in r}));w.writeheader();w.writerows(flat)
 return result

def representation():
 diagnostic=rows(OLD/'data/diagnostic.jsonl');assert len({tuple(r['dedup_groups']) for r in diagnostic})==1000;groups=[r['dedup_groups'][0] for r in diagnostic];base=torch.load(TRAIN/'diagnostics/Base/diagnostic_representations.pt',map_location='cpu',weights_only=True,mmap=True);be=base['en']['mean'][10].float();cached={};identities={}
 for name,cache in [('Base','Base'),('B','align-B'),('D0','align-D')]:
  folder=TRAIN/'diagnostics'/cache;identity=json.loads((folder/'DONE.json').read_text());modelpath=BASE if name=='Base' else TRAIN/'comparison_v3'/('D' if name=='D0' else name)/'checkpoint_004293/model';assert identity['model_sha256']==sha(modelpath/'model.safetensors');loaded=torch.load(folder/'diagnostic_representations.pt',map_location='cpu',weights_only=True,mmap=True);cached[name]={'en':loaded['en']['mean'][10].float(),'sw':loaded['sw']['mean'][10].float()};identities[name]={'model_sha256':identity['model_sha256'],'cache_file_sha256':sha(folder/'diagnostic_representations.pt'),'DONE_sha256':sha(folder/'DONE.json')}
 folder=RUN/'representation/D1';done=json.loads((folder/'DONE.json').read_text());assert sha(folder/'LAYER10_VECTORS.pt')==done['vectors_sha256'] and done['data_sha256']==sha(OLD/'data/diagnostic.jsonl');cached['D1']=torch.load(folder/'LAYER10_VECTORS.pt',map_location='cpu',weights_only=True);identities['D1']=done;result={'models':{},'identities':identities,'comparisons':{}};perpair={}
 def scores(en,sw):
  ne=F.normalize(en.float(),dim=-1);ns=F.normalize(sw.float(),dim=-1);c=(ne@ns.T).numpy();assert c.shape==(1000,1000);pos=np.diag(c);idx=np.arange(1000);p1=((c.argmax(1)==idx).astype(float)+(c.argmax(0)==idx).astype(float))/2;negrow=(c.sum(1)-pos)/999;negcol=(c.sum(0)-pos)/999;gap=pos-(negrow+negcol)/2
  return {'bidirectional_P1':float(p1.mean()*100),'ENtoSW_P1':float((c.argmax(1)==idx).mean()*100),'SWtoEN_P1':float((c.argmax(0)==idx).mean()*100),'positive_negative_gap':float(gap.mean()),'positive_cosine':float(pos.mean()),'normalized_SW_variance_trace':float(ns.var(0,unbiased=False).sum()),'normalized_SW_mean_norm':float(ns.mean(0).norm())},{'P1':p1,'gap':gap}
 for name,vec in cached.items():
  fixed,pair=scores(be,vec['sw']);moving,_=scores(vec['en'],vec['sw']);result['models'][name]={'fixed_Base_EN':fixed,'current_EN_SW':moving};perpair[name]=pair
 for control in ('B','D0'):
  result['comparisons']['D1-minus-'+control]={'fixed_anchor_bidirectional_P1':mean_ci(perpair['D1']['P1'],perpair[control]['P1'],groups,100),'positive_negative_gap':mean_ci(perpair['D1']['gap'],perpair[control]['gap'],groups,1)}
 write(REPORT/'REPRESENTATION_COMPARISON.json',result);return result

def main():
 torch.set_num_threads(4);d=downstream();r=representation();dc=d['comparisons']['D1-minus-B'];rc=r['comparisons']['D1-minus-D0'];gates={'independent_alignment':rc['fixed_anchor_bidirectional_P1']['low']>0 and rc['positive_negative_gap']['low']>0,'ENtoSW_primary':dc['en_to_sw']['low']>0,'SWreading_noninferiority':dc['sw_reading']['low']>=-2,'reverse_translation_noninferiority':dc['sw_to_en']['low']>=-1,'ENreading_noninferiority':dc['en_reading']['low']>=-2,'ENinstruction_noninferiority':dc['en_instruction']['low']>=-2};passed=all(gates.values());write(REPORT/'MANDATORY_DECISION.json',{'passed':passed,'gates':gates,'decision':'Eligible only to discuss secondseed; do not automatically launch' if passed else 'STOP fixedprototype; no automatic alpha/layer/pool/temperature scan or seed addition','frozen_protocol_sha256':sha(REPORT/'PROTOCOL_FROZEN.json'),'scope':'Exploratory one-seed development comparison, English retention relative B only, not comprehensive ability or formal test','uncertainty':'Failed noninferiority with wide interval is insufficient evidence of acceptable retention, not proof of true damage.'});print(json.dumps({'passed':passed,'gates':gates}),flush=True)
if __name__=='__main__':main()
