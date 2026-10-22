"""Endpoint-only deltas, paired source-cluster bootstrap, no checkpoint search."""
import sys
from common import *
sys.path.insert(0,str(ROOT.parent/'paired_capability_sparse_v1'))
from development_scoring import summarize
from compare_dev import paired_ci

def main():
 rr={name:rows(OUT/'development'/name/'PREDICTIONS.jsonl') for name in ['Base']+list('ABCD')};reference=[(r['partition'],r['id']) for r in rr['Base']]
 for name,r in rr.items():assert [(x['partition'],x['id']) for x in r]==reference
 by={n:{task:[v for v in r if v['task']==task] for task in {v['task'] for v in r}} for n,r in rr.items()};pairs=[('A','Base'),('B','A'),('C','A'),('D','A'),('C','B'),('D','B')];deltas={a+'-'+b:{task:paired_ci(by[a][task],by[b][task]) for task in by[a]} for a,b in pairs}
 decision={}
 for name in 'CD':
  b=deltas[name+'-B'];a=deltas[name+'-A'];sw=b['en_to_sw']['paired95percentCI'][0]>0 and b['sw_reading_kenswquad']['paired95percentCI'][0]>=-2 and b['sw_to_en']['paired95percentCI'][0]>=-1;en=all(a[task]['paired95percentCI'][0]>=-2 for task in ['en_reading','en_math','en_instruction']);decision[name]={'SW_benefit_and_noninferiority':sw,'EN_retention':en,'exploratory_screen_passed':sw and en,'long_training':'not released; independent confirmation/seed/shuffled-pair control required'}
 result={'decision':decision,'paired_deltas':deltas,'scores':{n:summarize(r) for n,r in rr.items()},'limits':'One seed and a single developer slice; paired intervals are item sampling uncertainty only, not training-seed uncertainty. No formal scores.'};write(OUT/'COMPARISON.json',result)
 # Preserve fully paired error table for downstream review; no aggregate mixed-task average.
 with (OUT/'PAIRED_OUTPUTS.jsonl').open('w') as f:
  for i,k in enumerate(reference):f.write(json.dumps({'partition':k[0],'id':k[1],'outputs':{name:{z:rr[name][i][z] for z in ('text','scores','language_diagnostic','capped')} for name in rr}},ensure_ascii=False)+'\n')
 import shutil
 destination=ROOT.parents[3]/'outputs/representation_alignment_report';destination.mkdir(parents=True,exist_ok=True)
 for name in ('COMPARISON.json','PAIRED_OUTPUTS.jsonl','BUDGET.json'):shutil.copy2(OUT/name,destination/name)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
