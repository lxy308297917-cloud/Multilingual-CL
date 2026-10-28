"""Run only after all four matched endpoint caches pass identity checks.

No checkpoint selection, layer selection, score tuning, or automatic patching.
"""
from pathlib import Path
import sys,json,csv,time,collections
ROOT=Path(__file__).resolve().parents[3];H=Path('/data0/users/lixiaoyang')
REPORT=ROOT/'docs/results/representation_drift_functional_forgetting_20261028'
WORK=ROOT/'work/representation_drift_functional_forgetting_20261028'
sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import read,write,sha
import numpy as np
from metrics import paired_metrics,covariance_drift,covariance_spectral_drift,functional_state,group_mean_intervals
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def dump_csv(path,rows):
 assert rows
 with path.open('x') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def auc_interval(y,a,b,groups):
 keys=list(dict.fromkeys(groups));by={k:np.flatnonzero(np.array(groups)==k) for k in keys}
 rng=np.random.default_rng(42);samples=[]
 for _ in range(2000):
  ix=np.concatenate([by[keys[k]] for k in rng.integers(0,len(keys),len(keys))])
  if len(np.unique(y[ix]))<2:continue
  samples.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]))
 assert samples
 return dict(difference=float(roc_auc_score(y,a)-roc_auc_score(y,b)),
  paired95CI=np.quantile(samples,[.025,.975]).tolist(),valid_replicates=len(samples),
  source_groups=len(keys),resampling_unit='source group',training_seed_uncertainty=False)

def predict_model(meta,preds,base_preds,metrics,signed,orthogonal):
 indices=[i for i,r in enumerate(meta) if r['language']=='en' and r['task']!='gsm8k' and base_preds[i]['correct'] is True]
 y=np.array([int(preds[i]['correct'] is False) for i in indices])
 discovery=np.array([meta[i]['phase']=='discovery' for i in indices]);confirmation=~discovery
 counts={split:dict(Forgotten=int(y[mask].sum()),Retained=int((1-y[mask]).sum())) for split,mask in [('discovery',discovery),('confirmation',confirmation)]}
 if min(v for counts_for_split in counts.values() for v in counts_for_split.values())<20:
  return dict(status='insufficient_functional_state_counts',counts=counts,gate_passed=False)
 bnorm=metrics['base_norm'][:,:,0].mean(axis=1)
 # Both drift and norm controls use the decision position; sensitivities are not selected for performance.
 drift=metrics['cosine_drift'][:,:,0].mean(axis=1)
 normchange=np.abs(np.log(np.maximum(metrics['norm_ratio'][:,:,0],1e-12))).mean(axis=1)
 features=[]
 for i in indices:
  features.append([float(meta[i]['task']=='knowledge'),np.log1p(len(meta[i]['input_ids'])),
   base_preds[i]['gold_label_probability'],base_preds[i]['gold_vs_best_other_margin'],
   np.log(max(bnorm[i],1e-12)),normchange[i],drift[i]])
 x=np.array(features);scores={}
 for name,n in [('base_controls',5),('controls_with_normchange',6),('controls_with_normchange_and_drift',7)]:
  model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=42))
  model.fit(x[discovery,:n],y[discovery]);assert int(model[-1].n_iter_.max())<1000,'Primary fixed logistic solver did not converge';scores[name]=model.predict_proba(x[confirmation,:n])[:,1]
 groups=[meta[i]['group'] for i,k in zip(indices,confirmation) if k]
 yy=y[confirmation]
 primary=auc_interval(yy,scores['controls_with_normchange_and_drift'],scores['controls_with_normchange'],groups)
 # Predeclared secondary blocks/positions/direction; display all, never pick a winner for the gate.
 depth=np.stack([metrics['cosine_drift'][:,start:stop,0].mean(1) for start,stop in ((0,9),(9,19),(19,28))],axis=1)
 positions=metrics['cosine_drift'].mean(axis=1)
 magnitude=metrics['normalized_l2'][:,:,0].mean(1)
 secondary_features={
  'fixed_depth_bins':np.concatenate((x[:,:6],depth[indices]),axis=1),
  'fixed_three_positions':np.concatenate((x[:,:6],positions[indices]),axis=1),
  'controls_drift_and_magnitude':np.column_stack((x,magnitude[indices])),
  'controls_drift_magnitude_and_direction':np.column_stack((x,magnitude[indices],signed[indices].mean(1),orthogonal[indices].mean(1)))}
 secondary_scores={}
 for name,features2 in secondary_features.items():
  fit=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=42))
  fit.fit(features2[discovery],y[discovery]);assert int(fit[-1].n_iter_.max())<1000,'Secondary fixed logistic solver did not converge';secondary_scores[name]=fit.predict_proba(features2[confirmation])[:,1]
 secondary={
  'depth_vs_overall_drift':auc_interval(yy,secondary_scores['fixed_depth_bins'],scores['controls_with_normchange_and_drift'],groups),
  'positions_vs_decision_only':auc_interval(yy,secondary_scores['fixed_three_positions'],scores['controls_with_normchange_and_drift'],groups),
  'direction_beyond_drift_magnitude':auc_interval(yy,secondary_scores['controls_drift_magnitude_and_direction'],secondary_scores['controls_drift_and_magnitude'],groups)}
 passed=primary['difference']>=.05 and primary['paired95CI'][0]>0
 return dict(status='evaluated',counts=counts,confirmation_AUROC={k:float(roc_auc_score(yy,v)) for k,v in scores.items()},
  primary_drift_over_controls=primary,gate_passed=passed,
  secondary_comparisons_exploratory_no_gate_effect=secondary,
  model='StandardScaler fit on discovery; logistic C=1, no hyperparameter search',
  interpretation='Association/prediction on exposed development materials; no causal or multi-seed claim',
  confirmation_predictions=[dict(id=meta[i]['id'],forgotten=int(y[j]),**{k:float(v[q]) for k,v in scores.items()})
   for q,(j,i) in enumerate((j,i) for j,i in enumerate(indices) if confirmation[j])])

def forgotten_retained_difference(values,states,groups):
 x=np.asarray(values,dtype=np.float64);keys=list(dict.fromkeys(groups));mapping={k:i for i,k in enumerate(keys)}
 sf=np.zeros((len(keys),x.shape[1]));sr=sf.copy();nf=np.zeros(len(keys));nr=nf.copy()
 for row,state,group in zip(x,states,groups):
  j=mapping[group]
  if state=='Forgotten':sf[j]+=row;nf[j]+=1
  elif state=='Retained':sr[j]+=row;nr[j]+=1
 if not nf.sum() or not nr.sum():return None
 rng=np.random.default_rng(42);w=rng.multinomial(len(keys),np.ones(len(keys))/len(keys),size=2000)
 good=(w@nf>0)&(w@nr>0);w=w[good]
 if not len(w):return None
 boot=(w@sf)/(w@nf)[:,None]-(w@sr)/(w@nr)[:,None]
 difference=sf.sum(0)/nf.sum()-sr.sum(0)/nr.sum()
 return difference,np.quantile(boot,[.025,.975],axis=0),int(nf.sum()),int(nr.sum()),len(w)

def label_contrasts(meta,base_preds,protocol):
 import torch
 from safetensors import safe_open
 path=H/'models/Qwen2.5-1.5B-Instruct/model.safetensors'
 assert sha(path)==read(REPORT/'SOURCE_IDENTITIES.json')['base_weights_sha256']
 with safe_open(str(path),framework='pt',device='cpu') as f:
  key='lm_head.weight' if 'lm_head.weight' in f.keys() else 'model.embed_tokens.weight'
  view=f.get_slice(key)
  weights=np.concatenate([view[token:token+1].float().numpy() for token in protocol['prompt_protocol']['label_token_ids']])
 directions=np.zeros((len(meta),1536),dtype=np.float64)
 for i,r in enumerate(meta):
  if r['task']=='gsm8k':continue
  gold=base_preds[i]['gold_index'];wrong=max((j for j in range(4) if j!=gold),key=lambda j:base_preds[i]['label_logprobs'][j])
  direction=(weights[gold]-weights[wrong]).astype(np.float64)
  directions[i]=direction/max(np.linalg.norm(direction),1e-12)
 return directions

def main():
 protocol=read(REPORT/'PROTOCOL_FROZEN.json');assert read(REPORT/'NUMERICAL_ACCEPTANCE.json')['passed']
 assert sha(protocol['input_manifest']['path'])==protocol['input_manifest']['sha256']
 meta=[json.loads(line) for line in Path(protocol['input_manifest']['path']).open()]
 caches={};predictions={};identities={}
 for name in protocol['models']:
  out=WORK/name/'first';done=read(out/'DONE.json');assert done['n']==1152
  assert done['protocol_sha256']==sha(REPORT/'PROTOCOL_FROZEN.json')
  for path,digest in done['files'].items():assert sha(out/path)==digest,(name,path)
  identity=read(out/'IDENTITY.json');identities[name]=identity
  content=read(out/'CONTENT_V2_ACCEPTANCE.json')
  assert content['passed'] and content['raw_prediction_sha256']==done['files']['PREDICTIONS.jsonl']
  assert content['parser_sha256']==read(REPORT/'MATH_CONTENT_V2_PROTOCOL.json')['parser_sha256']
  assert sha(out/'PREDICTIONS_CONTENT_V2.jsonl')==content['content_prediction_sha256']
  rows=[json.loads(line) for line in (out/'PREDICTIONS_CONTENT_V2.jsonl').open()]
  assert [r['id'] for r in rows]==[r['id'] for r in meta]
  assert [r['input_sha256'] for r in rows]==[r['input_sha256'] for r in meta]
  caches[name]=np.load(out/'REPRESENTATIONS.npy',mmap_mode='r');predictions[name]=rows
  assert caches[name].shape==(1152,28,3,1536)
  if name!='Base':
   req=protocol['adapted_model_requirement'];train=identity['training_done']
   assert train['steps']==req['steps'] and train['counts']['main_input']==req['main_input_tokens']
   assert train['counts']['main_supervised']==req['main_supervised_tokens']
 # All-or-none output directory prevents a partial comparison masquerading as complete.
 assert not (REPORT/'DRIFT_LAYERWISE.csv').exists()
 directions=label_contrasts(meta,predictions['Base'],protocol)
 contrasts=[];layerwise=[];transitions=[];covariances=[];paper_style=[];directional=[];prediction={};raw=WORK/'DRIFT_PER_ITEM_PRIVATE.csv'
 assert not raw.exists()
 with raw.open('x') as stream:
  fields=['model','id','phase','task','language','state','block','position','cosine_drift','normalized_l2','base_norm','norm_ratio']
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
  for name in protocol['models'][1:]:
   values={k:np.empty((1152,28,3),dtype=(bool if k=='near_zero_norm' else np.float64)) for k in ('cosine_drift','normalized_l2','base_norm','adapted_norm','norm_ratio','near_zero_norm')}
   for block in range(28):
    m=paired_metrics(caches['Base'][:,block],caches[name][:,block])
    for k in values:values[k][:,block]=m[k]
   states=[functional_state(b['correct'],a['correct']) for b,a in zip(predictions['Base'],predictions[name])]
   for i,r in enumerate(meta):
    for block in range(28):
     for position in range(3):
      row=dict(model=name,id=r['id'],phase=r['phase'],task=r['task'],language=r['language'],state=states[i],block=block,position=position)
      row.update({k:float(values[k][i,block,position]) for k in ('cosine_drift','normalized_l2','base_norm','norm_ratio')});writer.writerow(row)
   signed=[];orthogonal=[]
   for block in range(28):
    delta=caches[name][:,block,0].astype(np.float64)-caches['Base'][:,block,0].astype(np.float64)
    projection=np.sum(delta*directions,axis=-1)
    norms=values['base_norm'][:,block,0]
    signed.append(projection/np.maximum(norms,1e-12))
    orthogonal.append(np.sqrt(np.maximum(0.,np.sum(delta*delta,axis=-1)-projection**2))/np.maximum(norms,1e-12))
   signed=np.stack(signed,axis=1);orthogonal=np.stack(orthogonal,axis=1)
   for phase in ('discovery','confirmation'):
    for state in ('Retained','Forgotten','Acquired','Still Wrong'):
     ix=[i for i,r in enumerate(meta) if r['language']=='en' and r['task']!='gsm8k' and r['phase']==phase and states[i]==state]
     if not ix:continue
     combined=np.concatenate((signed[ix],orthogonal[ix]),axis=1)
     means,ci,groups=group_mean_intervals(combined,[meta[i]['group'] for i in ix])
     for block in range(28):
      directional.append(dict(model=name,phase=phase,state=state,block=block,n=len(ix),source_groups=groups,
       signed_contrast_projection=float(means[block]),signed_lower=float(ci[0,block]),signed_upper=float(ci[1,block]),
       orthogonal_drift=float(means[28+block]),orthogonal_lower=float(ci[0,28+block]),orthogonal_upper=float(ci[1,28+block]),
       caveat='Fixed Base label-unembedding contrast in residual coordinates; exploratory, not a pure language or semantic direction'))
   prediction[name]=predict_model(meta,predictions[name],predictions['Base'],values,signed,orthogonal)
   for task,language,phase in sorted({(r['task'],r['language'],r['phase']) for r in meta}):
    fixed=[i for i,r in enumerate(meta) if (r['task'],r['language'],r['phase'])==(task,language,phase)]
    counts=collections.Counter(states[i] for i in fixed)
    if language=='en':
     ix=[i for i in fixed if states[i] in ('Forgotten','Retained')]
     if ix:
      x=values['cosine_drift'][ix,:,0];x=np.concatenate((x,x.mean(axis=1,keepdims=True)),axis=1)
      comparison=forgotten_retained_difference(x,[states[i] for i in ix],[meta[i]['group'] for i in ix])
      if comparison is not None:
       difference,ci,nf,nr,valid=comparison
       for block in range(29):
        contrasts.append(dict(model=name,task=task,phase=phase,block=block if block<28 else 'all28_mean',
         Forgotten=nf,Retained=nr,cosine_drift_difference=float(difference[block]),
         lower=float(ci[0,block]),upper=float(ci[1,block]),valid_bootstraps=valid,
         inference='pointwise exploratory source CI; primary gate uses fixed mean-all-layer adjusted prediction'))
    for state in ('Retained','Forgotten','Acquired','Still Wrong','Unresolved'):
     transitions.append(dict(model=name,task=task,language=language,phase=phase,state=state,n=counts[state],denominator=len(fixed)))
    for state in ('All','Retained','Forgotten','Acquired','Still Wrong','Unresolved'):
     ix=fixed if state=='All' else [i for i in fixed if states[i]==state]
     if not ix:continue
     for position in range(3):
      combined=np.concatenate([values[k][ix,:,position] for k in ('cosine_drift','normalized_l2','norm_ratio')],axis=1)
      means,ci,groups=group_mean_intervals(combined,[meta[i]['group'] for i in ix])
      for block in range(28):
       row=dict(model=name,task=task,language=language,phase=phase,state=state,position=position,block=block,n=len(ix),source_groups=groups)
       for j,k in enumerate(('cosine_drift','normalized_l2','norm_ratio')):
        row.update({k:float(means[28*j+block]),k+'_lower':float(ci[0,28*j+block]),k+'_upper':float(ci[1,28*j+block])})
       layerwise.append(row)
    # Covariance always uses the same full fixed sample set, not model-dependent state subsets.
    for block in range(28):
     for position in range(3):
      covariances.append(dict(model=name,task=task,language=language,phase=phase,block=block,position=position,
       **covariance_drift(caches['Base'][fixed,block,position],caches[name][fixed,block,position])))
    b=caches['Base'][fixed,:,0].astype(np.float64).mean(0);a=caches[name][fixed,:,0].astype(np.float64).mean(0)
    paper_metrics=covariance_drift(b,a);paper_metrics['definition']='covariance across 28 decoder-block means, each averaged over the fixed examples'
    paper_style.append(dict(model=name,task=task,language=language,phase=phase,row_axis='decoder_block',original_examples=len(fixed),**paper_metrics,**covariance_spectral_drift(b,a),
     caveat='Rows are 28 layer means, NOT individual samples. One across-layer covariance scalar; not paper per-layer curve replication.'))
   print('Computed',name,flush=True)
 if contrasts:dump_csv(REPORT/'FORGOTTEN_RETAINED_CONTRASTS.csv',contrasts)
 dump_csv(REPORT/'DRIFT_LAYERWISE.csv',layerwise);dump_csv(REPORT/'FUNCTIONAL_TRANSITIONS.csv',transitions)
 dump_csv(REPORT/'COVARIANCE_LAYERWISE.csv',covariances);dump_csv(REPORT/'PAPER_STYLE_COVARIANCE.csv',paper_style)
 dump_csv(REPORT/'DIRECTIONAL_DRIFT.csv',directional)
 evaluated=[v for v in prediction.values() if v['status']=='evaluated']
 passed=[k for k,v in prediction.items() if v['gate_passed']]
 contradiction=any(v['primary_drift_over_controls']['paired95CI'][1]<0 for v in evaluated)
 gate=len(passed)>=2 and not contradiction
 write(REPORT/'DRIFT_FORGETTING_ANALYSIS.json',dict(status='completed_primary_association_analysis',prediction=prediction,
  patching_gate_passed=gate,individually_passing_models=passed,contradictory_adequately_powered_model=contradiction,
  patching_execution='pending technical implementation and acceptance' if gate else 'not authorized by frozen evidence gate; stop patching',
  no_causal_claim=True,mathematical_unparsed_excluded=True,formal_benchmark_table_unchanged=True,
  all_28_layers_reported=True,plots_and_cost_closeout_pending=True,directional_analysis_exploratory=True,finished_unix=time.time()))
 write(WORK/'MODEL_CACHE_IDENTITIES_VERIFIED.json',dict(models={n:dict(model_sha256=r['model_sha256'],source_sha256=r['source_sha256']) for n,r in identities.items()},unix=time.time()))

if __name__=='__main__':main()
