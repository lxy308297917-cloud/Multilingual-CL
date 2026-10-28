"""Use the saved reference's CUDA top-highest complement at the same quota."""
from pathlib import Path
import sys,os,argparse,json,time,csv
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import H,read,write,sha
REPORT=ROOT/'outputs/sw_fullbudget_baselines_20261028/audit_optimization_20261028';RUN=ROOT/'work/sw_fullbudget_baselines_20261028/audit_optimization_20261028'
ap=argparse.ArgumentParser();ap.add_argument('--uuid',required=True);a=ap.parse_args()
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
snap=q.snapshot();assert a.uuid in q.user_gpu_assignments(snap,q.other_queue_gpus())
assert len(q.user_gpu_assignments(snap,q.other_queue_gpus()))<=4
assert all(os.stat('/proc/'+str(p)).st_uid==os.getuid() for p in snap[a.uuid]['pids'])
os.environ['CUDA_VISIBLE_DEVICES']=a.uuid
import torch
torch.cuda.set_per_process_memory_fraction(.02);torch.set_num_threads(4)
cache=torch.load(RUN/'SSU_ACTIVATION_CACHE.pt',map_location='cpu',mmap=True,weights_only=False)
oldpath=H/'models/checkpoint/sw_fullbudget_baselines_20261028/SSU_first/selection.pt';oldhash=sha(oldpath)
saved=torch.load(oldpath,map_location='cpu',weights_only=False)['indices']
variants=torch.load(RUN/'SSU_COMPARISON_MASKS.pt',map_location='cpu',weights_only=False)['indices']
budget=read(ROOT/'outputs/sw_fullbudget_baselines_20261028/COMMON_BUDGET_20261028.json')
shapes=read(ROOT/'outputs/sw_fullbudget_baselines_20261028/DATA_FROZEN.json')['shapes'];rows=[];indices={}
for n,s in cache['statistics'].items():
 score=(s['mean_bf16'].mean(0)*s['weight_norm_bf16']).cuda();k=budget['ssu_columns'][n]
 frozen=torch.topk(score,len(score)-k,largest=True).indices
 train=torch.ones(len(score),dtype=torch.bool,device='cuda');train[frozen]=False
 chosen=torch.where(train)[0].cpu();assert len(chosen)==k;indices[n]=chosen
 actual=set(chosen.tolist());old=set(saved[n].tolist());stable=set(variants['reference_sample_bf16'][n].tolist())
 rows.append(dict(name=n,module='Attention' if '.self_attn.' in n else 'MLP',selected_columns=k,
  selected_elements=shapes[n][0]*k,overlap_with_current_columns=len(actual&old),overlap_with_current_elements=shapes[n][0]*len(actual&old),
  overlap_with_stable_reference_columns=len(actual&stable),overlap_with_stable_reference_elements=shapes[n][0]*len(actual&stable)))
aggregate={}
for group in ('All','Attention','MLP'):
 here=[r for r in rows if group=='All' or r['module']==group];total=sum(r['selected_elements'] for r in here)
 overlap=sum(r['overlap_with_current_elements'] for r in here);stable=sum(r['overlap_with_stable_reference_elements'] for r in here)
 aggregate[group]=dict(selected_elements=total,intersection_with_current=overlap,overlap_with_current=overlap/total,
  intersection_with_stable_reference=stable,overlap_with_stable_reference=stable/total)
assert sha(oldpath)==oldhash
with (REPORT/'SSU_REFERENCE_CUDA_TIES.csv').open('w') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
torch.save(dict(indices=indices,exact_common_budget=True,reference_cuda_top_highest_complement=True),RUN/'SSU_REFERENCE_CUDA_MASKS.pt')
write(REPORT/'SSU_REFERENCE_CUDA_TIES.json',dict(passed=True,aggregates=aggregate,old_selection_unchanged=True,
 only_quota_adjusted_to_match196608000=True,reference_aggregation_and_BF16_score=True,CUDA_topk_highest_complement=True,
 original_reference_default_fraction_not_claimed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),completed_unix=time.time()))
print(json.dumps(aggregate,indent=2))
