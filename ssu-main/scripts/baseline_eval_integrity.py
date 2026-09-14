"""Fail-closed artifact and aggregation checks for the fixed baseline protocol."""
import json,math,statistics
from pathlib import Path

def require_scores(scores):
 if not scores or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in scores.values()):
  raise RuntimeError('Missing or non-finite task scores')

def aggregate(scores,base,groups):
 relative={}
 for key,value in scores.items():
  if key not in base:raise RuntimeError('Missing same-protocol Base metric: '+key)
  denominator=base[key]['mean']
  relative[key]=None if denominator==0 else (value['mean']/denominator-1)*100
 totals={name:None if any(relative.get(k) is None for k in members) else statistics.mean(relative[k] for k in members) for name,members in groups.items()}
 return relative,totals

def unique_completed(dest,expected,validate,identity):
 records=[]
 for job in expected:
  unit=Path(dest)/(job['name']+'-seed'+str(job['seed']))
  good=[d for d in unit.glob('attempt-*') if validate(d,dict(identity,job=job))]
  if len(good)!=1:raise RuntimeError('Expected exactly one valid completed attempt: '+str(unit))
  scores=json.loads((good[0]/'scores.json').read_text());require_scores(scores)
  records.append((job,good[0],scores))
 return records
