"""Versioned content scores from immutable raw answers; no regeneration or gold-aware extraction."""
from pathlib import Path
import sys,json,time,argparse,runpy
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'docs/results/representation_drift_functional_forgetting_20261028';WORK=ROOT/'work/representation_drift_functional_forgetting_20261028'
sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import read,write,sha
from math_content_v2 import parse
p=argparse.ArgumentParser();p.add_argument('--model',choices=['Base','Dense','Element','Protected'],required=True);a=p.parse_args()
protocol=read(REPORT/'MATH_CONTENT_V2_PROTOCOL.json')
assert sha(Path(__file__).parent/'math_content_v2.py')==protocol['parser_sha256']
runpy.run_path(str(Path(__file__).parent/'math_content_v2.py'),run_name='__main__')
out=WORK/a.model/'first';done=read(out/'DONE.json');raw=out/'PREDICTIONS.jsonl'
assert sha(raw)==done['files']['PREDICTIONS.jsonl']
destination=out/'PREDICTIONS_CONTENT_V2.jsonl';assert not destination.exists()
rows=[json.loads(line) for line in raw.open()];counts={};math=[]
with destination.open('x') as stream:
 for r in rows:
  if r['task']=='gsm8k':
   value,detail=parse(r['text'])
   r=dict(r,original_v1_status=r['status'],original_v1_correct=r['correct'],original_v1_extracted_answer=r['extracted_answer'],
    extracted_answer=str(value) if value is not None else None,content_parser=detail,status=detail['status'],
    correct=(value==Fraction(r['reference_final'])) if value is not None else None)
   math.append(r)
  stream.write(json.dumps(r,ensure_ascii=False)+'\n')
counts=dict(n=len(math),correct=sum(r['correct'] is True for r in math),incorrect_parsed=sum(r['correct'] is False for r in math),
 unresolved=sum(r['correct'] is None for r in math),strict_original_hash_format=sum(r['original_v1_status']=='parsed' for r in math))
write(out/'CONTENT_V2_ACCEPTANCE.json',dict(passed=True,model=a.model,raw_prediction_sha256=sha(raw),content_prediction_sha256=sha(destination),
 parser_sha256=protocol['parser_sha256'],protocol_sha256=sha(REPORT/'MATH_CONTENT_V2_PROTOCOL.json'),
 no_generation=True,no_raw_record_overwrite=True,numeric_content_summary=counts,unix=time.time()))
print(json.dumps(counts))
