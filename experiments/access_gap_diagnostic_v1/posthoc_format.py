"""POST HOC format sensitivity; never replaces frozen protocol scores."""
from pathlib import Path
import sys,re,json,time,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common import *
import numpy as np
DECL=re.compile(r'\b(?:Final(?:\s+answer)?|Answer|Jibu(?:\s+la\s+mwisho)?)\s*:\s*([ABCD])\b',re.I)
CLAIM=re.compile(r'(?:correct\s+answer\s+is|jibu\s+sahihi\s+ni)\s*\(?([ABCD])\)?\b',re.I)
INITIAL=re.compile(r'^\s*([ABCD])\s*(?:[.)]|$)',re.I)
OPTIONS=re.compile(r'\b([ABCD])[.)]\s')
AMBIG=re.compile(r'^\s*(?:[,/&]\s*[ABCD]\b|(?:or|and|au|na)\s+[ABCD]\b)',re.I)
def extract(text):
 matches=list(DECL.finditer(text));labels={m.group(1).upper() for m in matches}|{x.upper() for x in CLAIM.findall(text)}
 if any(AMBIG.match(text[m.end():]) for m in matches):return None
 if len(labels)>1:return None
 if matches:return next(iter(labels))
 m=INITIAL.match(text)
 if m and len(set(OPTIONS.findall(text)))<=1:
  if labels and m.group(1).upper() not in labels:return None
  return m.group(1).upper()
 return None
cases=[('Final: B, kwa sababu ...','B'),('Final: B. The Soviets.','B'),('B. The Soviets.','B'),('B','B'),('A good sentence.',None),('A. one B. two C. three D. four',None),('Final: A or B',None),('Final: A, B',None),('Final: A\nAnswer: B',None),('Final: A\nFinal: A','A'),('Explanation.\nFinal: D','D'),('No clear answer.',None),('Jibu la mwisho: C','C')]
for text,answer in cases:assert extract(text)==answer,(text,extract(text),answer)
write(DEST/'POSTHOC_SCORER_ACCEPTANCE.json',dict(passed=True,controlled_cases=len(cases),posthoc=True,main_scores_unchanged=True,rule='Unique explicit final/answer declaration, or unambiguous initial choice; no option-text fuzzy match; contradictory declarations or option lists rejected.'))
assert read(DEST/'RAW_ARTIFACT_ACCEPTANCE.json')['passed']
def ci(delta,groups):
 keys=list(dict.fromkeys(groups));ix=np.array([keys.index(g) for g in groups]);s=np.bincount(ix,weights=delta,minlength=len(keys));n=np.bincount(ix,minlength=len(keys));rng=np.random.default_rng(42)
 w=rng.multinomial(len(keys),[1/len(keys)]*len(keys),size=10000);v=(w@s)/(w@n)
 return [float(x)*100 for x in np.quantile(v,[.025,.975])]
summary={};comparisons={};private=[]
for model in MODELS:
 rows=[json.loads(l) for l in (RUN/'generation'/model/'PREDICTIONS.jsonl').read_text().splitlines()]
 keyed={(r['id'],r['condition']):r for r in rows}
 for r in rows:
  a=extract(r['text']);private.append(dict(model=model,id=r['id'],condition=r['condition'],answer=a,correct=a==r['gold']))
 for task in ('belebele','knowledge'):
  ids=sorted({r['id'] for r in rows if r['task']==task})
  for c in 'ABCDE':
   rr=[keyed[q,c] for q in ids];answers=[extract(r['text']) for r in rr];correct=[a==r['gold'] for a,r in zip(answers,rr)]
   expected='en' if c in 'AC' else 'sw'
   summary[f'{model}/{task}/{c}']=dict(n=64,posthoc_correct=sum(correct),posthoc_ACC_percent=100*sum(correct)/64,posthoc_unparsed=sum(a is None for a in answers),
    frozen_correct=sum(r['scores']['correct'] for r in rr),frozen_extraction_failures=sum(r['scores']['extraction_failure'] for r in rr),
    previously_unparsed_now_explicitly_correct=sum(r['scores']['extraction_failure'] and ok for r,ok in zip(rr,correct)),
    explanation_language_matches_heuristic=sum(r['explanation_language']==expected for r in rr),explanation_language_undetermined=sum(r['explanation_language']=='und' for r in rr))
  for a,b in [('C','B'),('D','A'),('E','B')]:
   diff=np.array([float(extract(keyed[q,a]['text'])==keyed[q,a]['gold'])-float(extract(keyed[q,b]['text'])==keyed[q,b]['gold']) for q in ids])
   comparisons[f'{model}/{task}/{a}-{b}']=dict(difference_pp=float(diff.mean()*100),paired_95CI_pp=ci(diff,[keyed[q,a]['group'] for q in ids]),posthoc=True,source_cluster_bootstrap=10000)
write(DEST/'POSTHOC_FORMAT_SENSITIVITY.json',dict(posthoc=True,does_not_replace_main_protocol=True,no_new_generation=True,rule_sha256=sha(Path(__file__)),summary=summary,comparisons=comparisons,
 limitations=['Designed after examining actual format failures; exploratory diagnostic only','Unambiguous letter extraction is not manual verification of reasoning or Swahili semantics','Language compliance heuristic is unreliable for short or mixed explanations','No training or method selection allowed from these results'],finished_unix=time.time()))
write(RUN/'POSTHOC_PARSED_PRIVATE.json',private)
print('Posthoc format sensitivity complete; frozen main results preserved')
