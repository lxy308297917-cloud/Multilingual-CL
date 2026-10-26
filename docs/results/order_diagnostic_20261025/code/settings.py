from pathlib import Path
import json,hashlib,time,os,sys
ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'outputs/order_diagnostic_v1'
RUN=ROOT/'work/order_diagnostic_v1'
PRIOR=ROOT/'outputs/module_mix50_v1'
PRUN=ROOT/'work/module_mix50_v1'
H=Path('/srv/multilingual')
PYTHON=H/'ssu-migration/envs/cl/bin/python'
BASE=H/'models/Qwen2.5-1.5B-Instruct'
MODELS={'42':H/'models/checkpoint/module_lr077_20261023/S-All','43':H/'models/checkpoint/module_mix50_20261023/LR077-s43'}
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):d.update(b)
 return d.hexdigest()
def usage():
 total=0;active=0
 for p in RUN.glob('WORKER_*.json'):
  if p.name.startswith(('WORKER_DONE_','WORKER_FAILED_')):continue
  r=read(p);end=RUN/('WORKER_DONE_'+p.name[7:]);fail=RUN/('WORKER_FAILED_'+p.name[7:]);receipt=end if end.exists() else fail if fail.exists() else None
  total+=(read(receipt)['finished_unix'] if receipt else time.time())-r['started_unix'];active+=receipt is None
 return total,active
def guard():
 used,n=usage();assert used+n*45<7200,'Frozen2GPUh budget with45second perworker reserve'
