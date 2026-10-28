from pathlib import Path
import hashlib,json,time,os
ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'outputs/sw_fullbudget_baselines_20261028'
RUN=ROOT/'work/sw_fullbudget_baselines_20261028'
H=Path('/data0/users/lixiaoyang')
BASE=H/'models/Qwen2.5-1.5B-Instruct'
PYTHON=H/'ssu-migration/envs/cl/bin/python'
EVALPY=H/'ssu-migration/envs/cl_eval_latest/bin/python'
METHODS=('Dense','MoFO','SSU','HFT15','Element','Protected')
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def evaluation_output(model,stage,task):
 path=RUN/'evaluation'/model/stage/task
 overrides=RUN/'EVALUATION_ATTEMPTS.json'
 if overrides.exists():
  attempt=read(overrides).get('/'.join((model,stage,task)))
  if attempt:path=path/attempt
 return path
