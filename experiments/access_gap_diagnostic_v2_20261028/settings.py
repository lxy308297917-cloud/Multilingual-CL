from pathlib import Path
import hashlib,json,time,os
ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'outputs/access_gap_diagnostic_v2_20261028'
RUN=ROOT/'work/access_gap_diagnostic_v2_20261028'
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

MODELS=("Base","LR077-s42","Historical-element_protected")
OLD_RUN=ROOT/"work/benchmark_direction_20261026/access_diagnostic"
OLD_REPORT=ROOT/"outputs/benchmark_direction_20261026/access_diagnostic"

RUN=RUN/"recovery01"
