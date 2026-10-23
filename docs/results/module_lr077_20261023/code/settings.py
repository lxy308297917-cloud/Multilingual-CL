import pathlib,json,hashlib,time,sys,os
ROOT=pathlib.Path(__file__).resolve().parents[3]
REPORT=ROOT/'outputs/module_lr077_v1'
RUN=ROOT/'work/module_lr077_v1'
H=pathlib.Path('/data0/users/lixiaoyang')
BASE=H/'models/Qwen2.5-1.5B-Instruct'
OUT=H/'models/checkpoint/module_lr077_20261023'
OLD=ROOT/'outputs/Multilingual-CL-alignment-recovery-v3/experiments/representation_alignment_v1'
SPARSE=ROOT/'outputs/Multilingual-CL-paired-deadline/experiments/paired_capability_sparse_v1'
sys.path.insert(0,str(OLD))
from common import sha,write,rows,seed,pack,check_compute

def allocate(shapes,K):
 total=sum(a*b for a,b in shapes.values());q={n:K*(a*b)//total for n,(a,b) in shapes.items()};remaining=K-sum(q.values());order=sorted(shapes,key=lambda n:-(K*shapes[n][0]*shapes[n][1]%total))
 for n in order[:remaining]:q[n]+=1
 assert sum(q.values())==K and all(0<q[n]<=shapes[n][0]*shapes[n][1] for n in shapes)
 return q

REFERENCE=ROOT/'outputs/module_budget_sparse_v1'
REFERENCE_RUN=ROOT/'work/module_budget_sparse_v1'
REFERENCE_OUT=H/'models/checkpoint/module_budget_sparse_20261022'
STUDY_NAME='S-All-LR077'
LR_MULTIPLIER=0.7714
