import pathlib,json,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
RUN=ROOT/'work/evaluation_audit_v2'
REPORT=ROOT/'outputs/evaluation_audit_v2'
OLD=ROOT/'outputs/Multilingual-CL-alignment-recovery-v3/experiments'
H=pathlib.Path('/data0/users/lixiaoyang')
BASE=H/'models/Qwen2.5-1.5B-Instruct'
TRAIN=H/'models/checkpoint/representation_alignment_20261021'
MODELS={'Base':str(BASE),**{m:str(TRAIN/'comparison_v3'/m/'checkpoint_004293/model') for m in 'ABCD'}}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def rows(p):return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]
def write(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
