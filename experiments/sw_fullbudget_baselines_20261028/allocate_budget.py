"""Exact common element count while retaining HFT matrices and SSU columns."""
from settings import *
import math,torch
from selection import hft_masks
shapes=read(REPORT/'DATA_FROZEN.json')['shapes']
named={n:torch.empty(s,device='meta') for n,s in shapes.items()}
_,meta=hft_masks(named,.15,42,RUN/'HFT_BUDGET_SELECTION.pt')
N=sum(math.prod(s) for s in shapes.values());K=meta['selected_elements']
q={n:math.prod(s)*K//N for n,s in shapes.items()}
for n in sorted(shapes,key=lambda n:(-(math.prod(shapes[n])*K%N),n))[:K-sum(q.values())]:q[n]+=1
# Each column count lies at floor/ceil of its proportional target. Dynamic
# programming picks the exact global budget with minimum squared element error.
cols={n:s[1]*K//N for n,s in shapes.items()}
residual=K-sum(shapes[n][0]*v for n,v in cols.items())
unit=math.gcd(*(s[0] for s in shapes.values()));assert residual%unit==0
reachable={0:(0.,())}
for n,s in shapes.items():
 weight=s[0]//unit;ideal=s[1]*K/N;v=cols[n]
 cost=((v+1-ideal)**2-(v-ideal)**2)*s[0]**2;additions={}
 for total,(prior,chosen) in list(reachable.items()):
  new=total+weight
  if new>residual//unit:continue
  value=(prior+cost,chosen+(n,));old=reachable.get(new);other=additions.get(new)
  if (old is None or value[0]<old[0]) and (other is None or value[0]<other[0]):additions[new]=value
 for total,value in additions.items():
  if total not in reachable or value[0]<reachable[total][0]:reachable[total]=value
assert residual//unit in reachable
for n in reachable[residual//unit][1]:cols[n]+=1
assert sum(q.values())==sum(shapes[n][0]*v for n,v in cols.items())==K
result=dict(candidate_elements=N,target_fraction=.15,common_selected_elements=K,common_actual_fraction=K/N,
 previous=dict(Element=sum(math.floor(math.prod(s)*.15) for s in shapes.values()),
  Protected=sum(math.floor(math.prod(s)*.15) for s in shapes.values()),
  SSU=sum(s[0]*math.floor(s[1]*.15) for s in shapes.values()),HFT=K),
 element_quotas=q,ssu_columns=cols,hft_selected_names=meta['selected_parameters'],
 user_requirement='HFT/SSU/our methods exactly equal selected element count; closest feasible whole-matrix budget to15%',
 excluded_from_denominator='embedding,head,norm,bias; Dense not budgetmatched',
 MoFO='official threshold15% adaptive component; global decay separately reported',
 per_matrix_element_floor15_is_replaced_by_proportional_common_budget_integer_allocation=True,
 selection_granularities_preserved=True)
path=REPORT/'COMMON_BUDGET_20261028.json'
if path.exists():
 existing=read(path)
 for key,value in result.items():assert existing[key]==value,key
else:write(path,dict(result,created_unix=time.time()))
print(f'Exact matched support: {K}/{N} = {100*K/N:.9f}%')
