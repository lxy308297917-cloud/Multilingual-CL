import pathlib,json,hashlib,time,sys,os
ROOT=pathlib.Path(__file__).resolve().parents[3]
REPORT=ROOT/'outputs/module_mix50_v1'
RUN=ROOT/'work/module_mix50_v1'
H=pathlib.Path('/srv/multilingual')
BASE=H/'models/Qwen2.5-1.5B-Instruct'
OUT=H/'models/checkpoint/module_mix50_20261023'
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
STUDY_NAME=os.environ.get('MIX50_JOB','preflight')
JOBS={'LR077-s43':{'group':'S-All','seed':43,'multiplier':.7714},'Attn-s43':{'group':'S-Attn','seed':43,'multiplier':1.},'Mix50-s42':{'group':'S-Mix50','seed':42,'multiplier':.7714},'Mix50-s43':{'group':'S-Mix50','seed':43,'multiplier':.7714}}
JOB=JOBS.get(STUDY_NAME,{'group':'S-Mix50','seed':42,'multiplier':.7714})
LR_MULTIPLIER=JOB['multiplier']

def budget_seconds():
 now=time.time();seconds=0.;active=0
 for p in RUN.glob('WORKER_*.json'):
  if p.name.startswith(('WORKER_DONE_','WORKER_FAILED_')):continue
  r=json.loads(p.read_text());suffix=p.name[len('WORKER_'):];done=RUN/('WORKER_DONE_'+suffix);failed=RUN/('WORKER_FAILED_'+suffix)
  end=json.loads(done.read_text())['finished_unix'] if done.exists() else json.loads(failed.read_text())['finished_unix'] if failed.exists() else now
  seconds+=end-r['started_unix'];active+=not (done.exists() or failed.exists())
 return seconds,active

def budget_guard():
 seconds,active=budget_seconds();assert seconds+active*60<16*3600,'Frozen total16GPUh ceiling/reserve; preserve state and no automatic more work'
