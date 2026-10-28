"""Separate aggregation, precision and tie effects using one activation cache."""
from pathlib import Path
import sys,json,csv,time,collections,hashlib
import torch
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import H,read,write,sha
REPORT=ROOT/'outputs/sw_fullbudget_baselines_20261028/audit_optimization_20261028'
RUN=ROOT/'work/sw_fullbudget_baselines_20261028/audit_optimization_20261028'
torch.set_num_threads(4)
cache=torch.load(RUN/'SSU_ACTIVATION_CACHE.pt',map_location='cpu',mmap=True,weights_only=False)
selection_path=H/'models/checkpoint/sw_fullbudget_baselines_20261028/SSU_first/selection.pt'
assert sha(selection_path)==cache['original_selection_sha256']
saved=torch.load(selection_path,map_location='cpu',weights_only=False)
assert saved['metadata']['calibration_ids']==cache['calibration_ids']
budget=read(ROOT/'outputs/sw_fullbudget_baselines_20261028/COMMON_BUDGET_20261028.json')
shapes=read(ROOT/'outputs/sw_fullbudget_baselines_20261028/DATA_FROZEN.json')['shapes']
lengths=torch.tensor(cache['lengths'],dtype=torch.float64);results=[];chosen={};summary={}
for name,s in cache['statistics'].items():
 total=torch.zeros_like(s['sum_fp32'][0],dtype=torch.float64)
 for value in s['sum_fp32']:total.add_(value.double()) # frozen serial accumulation
 token=(total/lengths.sum()).float()
 sample=(s['sum_fp32'].double()/lengths[:,None]).mean(0).float()
 sample64=s['mean_fp64'].mean(0);token64=(s['mean_fp64']*lengths[:,None]).sum(0)/lengths.sum()
 sample_bf16=s['mean_bf16'].mean(0)
 scores=dict(frozen_token_fp32=token*s['weight_norm_fp32'],sample_fp32=sample*s['weight_norm_fp32'],
  token_fp64=token64*s['weight_norm_fp64'],sample_fp64=sample64*s['weight_norm_fp64'],
  sample_bf16activation_fp32score=sample_bf16.float()*s['weight_norm_fp32'],
  reference_sample_bf16=sample_bf16*s['weight_norm_bf16'])
 k=budget['ssu_columns'][name];old=saved['indices'][name].long();oldmask=torch.zeros(len(token),dtype=torch.bool);oldmask[old]=True
 for variant,score in scores.items():
  assert bool(torch.isfinite(score).all())
  order=torch.argsort(score,stable=True);idx=order[:k];mask=torch.zeros_like(oldmask);mask[idx]=True
  if variant=='frozen_token_fp32':assert torch.equal(mask,oldmask),(name,'cached frozen aggregation did not reproduce saved selection')
  chosen.setdefault(variant,{})[name]=idx
  boundary=score[order[k-1]];tie=score==boundary;lower=score<boundary
  need=k-int(lower.sum());ties=int(tie.sum());old_ties=int((oldmask&tie).sum());fixed=int((oldmask&lower).sum())
  overlap=int((mask&oldmask).sum());rows=shapes[name][0]
  record=dict(name=name,module='Attention' if '.self_attn.' in name else 'MLP',variant=variant,columns=len(score),selected_columns=k,
   common_selected_elements=rows*k,overlap_columns=overlap,overlap_elements=rows*overlap,overlap_fraction=overlap/k,
   selected_jaccard=overlap/(2*k-overlap),boundary_ties=ties,boundary_slots=need,
   tie_sensitive=ties>need,overlap_min_if_arbitrary_ties=fixed+max(0,need-(ties-old_ties)),
   overlap_max_if_arbitrary_ties=fixed+min(need,old_ties))
  results.append(record)
 for a,b in [('sample_fp32','sample_fp64'),('token_fp64','sample_fp64'),('sample_bf16activation_fp32score','reference_sample_bf16')]:
  first=set(chosen[a][name].tolist());second=set(chosen[b][name].tolist())
  summary.setdefault('direct_precision_pairs',{}).setdefault(a+' vs '+b,[]).append(dict(name=name,elements=shapes[name][0]*k,intersection=shapes[name][0]*len(first&second)))
aggregates={}
for variant in chosen:
 aggregates[variant]={}
 for group in ('All','Attention','MLP'):
  rr=[r for r in results if r['variant']==variant and (group=='All' or r['module']==group)]
  n=sum(r['common_selected_elements'] for r in rr);inter=sum(r['overlap_elements'] for r in rr)
  aggregates[variant][group]=dict(selected_elements=n,intersection_elements=inter,overlap_fraction=inter/n,
   selected_jaccard=inter/(2*n-inter),changed_elements=n-inter,
   matrices_with_different_selection=sum(r['overlap_columns']!=r['selected_columns'] for r in rr),
   matrices_with_boundary_ties=sum(r['tie_sensitive'] for r in rr))
for name,rows in summary['direct_precision_pairs'].items():
 summary['direct_precision_pairs'][name]=dict(selected_elements=sum(r['elements'] for r in rows),intersection_elements=sum(r['intersection'] for r in rows),overlap_fraction=sum(r['intersection'] for r in rows)/sum(r['elements'] for r in rows))
assert aggregates['frozen_token_fp32']['All']['selected_elements']==196608000
summary.update(passed=True,samples=128,token_min=min(cache['lengths']),token_max=max(cache['lengths']),
 current_selection_reproduced_exactly=True,aggregates=aggregates,
 reference_ties='Stable lowest-score order used to isolate differences. Saved reference freezes top(C-k), whose unstable ties may select different boundary columns; per-matrix lower/upper overlap bounds are recorded. No claim CPU topk tie ordering equals CUDA.',
 current_method_identity='SSU-token-weighted-FP32-score-budgetmatched',
 frozen_definition_status='Broad frozen description lowimportance columns does not specify sample-vs-token aggregation; implementation conforms to its source snapshot but this aggregation difference from the saved reference was not explicitly justified.',
 known_intentional_adaptations=['196 matrix scope, embedding/head/norm/bias frozen','Exact common column budget by floor/ceil allocation','Mask gradients before clipping, masked decay and writeback','Stable ties to meet exact quota'],
 unannounced_reference_differences=['Token-weighted instead of equal-sample activation mean','FP32 square and score with FP64 accumulation instead of BF16 activation square/means and BF16 norms/scores when model is BF16'],
 original_selection_sha256=sha(selection_path),cache_sha256=sha(RUN/'SSU_ACTIVATION_CACHE.pt'),checked_unix=time.time(),
 no_training_or_mask_changed=True)
with (REPORT/'SSU_MATRIX_COMPARISON.csv').open('w') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
torch.save(dict(indices=chosen,source_cache_sha256=summary['cache_sha256']),RUN/'SSU_COMPARISON_MASKS.pt')
write(REPORT/'SSU_AGGREGATION_IMPACT.json',summary)
print(json.dumps(dict(aggregates=aggregates,precision=summary['direct_precision_pairs']),indent=2))
