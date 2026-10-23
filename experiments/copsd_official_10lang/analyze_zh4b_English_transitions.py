"""Describe English score changes for the accepted ZH4B seed42 pair.

Keep official scores. A truncated response is an observed budget outcome,
not proof of mathematical inability or of recovery with a longer budget.
"""
import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();code=Path(__file__).parent
    report_path=code/'zh4b_followup200_completed_20261023.PUBLIC.json';report=read(report_path);frozen=read(code/'zh4b_seed43_eval_frozen_20261023.PUBLIC.json')
    folders=[a.root/frozen['cached_Base_English']['relative'],a.root/'logs/copsd_official_10lang/zh4b_followup_development200'/report['languages']['en']['conditions']['ZH20']['source']['run_id']]
    records=[]
    for case,folder in zip(['Base','ZH20'],folders):
        expected=report['languages']['en']['conditions'][case]['source'];assert sha(folder/'ITEMS.jsonl')==expected['items_sha256']
        records.append([json.loads(x) for x in (folder/'ITEMS.jsonl').read_text().splitlines()])
    b,m=records;assert len(b)==len(m)==200
    for x,y in zip(b,m):assert all(x[k]==y[k] for k in ['id','question','answer','prompt'])
    changes={}
    for name,condition in [('gained',lambda x,y:not x and y),('lost',lambda x,y:x and not y)]:
        indices=[i for i,(x,y) in enumerate(zip(b,m)) if condition(x['samples'][0]['correct'],y['samples'][0]['correct'])]
        changes[name]={'count':len(indices),'indices_zero_based':indices,'finish_transitions':dict(Counter(b[i]['samples'][0]['finish_reason']+'->'+m[i]['samples'][0]['finish_reason'] for i in indices)),
            'trained_no_box':sum(m[i]['samples'][0]['boxed_answer'] is None for i in indices)}
        assert changes[name]['count']==report['languages']['en']['paired'][name]
    normal=[i for i in changes['lost']['indices_zero_based'] if m[i]['samples'][0]['finish_reason']=='stop'];assert normal==[65]
    i=65;assert m[i]['samples'][0]['boxed_answer']=='1902' and b[i]['samples'][0]['boxed_answer']=='12017'
    # Exact coordinate solution independently checks the one normal-termination loss.
    A=(F(0),F(6));E=(F(42,17),F(66,17));P=(F(-2),F(2));H=(F(0),F(7,2))
    circle=lambda p:p[0]**2+p[1]**2-4*p[0]+F(5,2)*p[1]-21
    assert all(circle(p)==0 for p in [E,P,H,(F(-3),F(0)),(F(7),F(0))])
    assert E[1]==6-F(6,7)*E[0] and P[1]==2*P[0]+6
    area=abs((E[0]-A[0])*(P[1]-A[1])-(E[1]-A[1])*(P[0]-A[0]))/2
    assert area==F(120,17) and 100*area.numerator+area.denominator==12017
    wrong_E=(F(0),F(-7,2));assert wrong_E[1]!=6-F(6,7)*wrong_E[0]
    out={'completed_at':datetime.now().isoformat(),'source_report_sha256':sha(report_path),'analyzer_sha256':sha(Path(__file__)),'English_correct_of200':{'Base':166,'ZH20':161},'changes':changes,
        'normal_termination_loss_review':{'index_zero_based':65,'question_ID':b[i]['id'],'category':'invalid_geometric_construction','finding':'The trained answer identifies E as reflection of H across BC. Its E=(0,-7/2) is not on the required AC line y=6-6x/7. Exact circle-line intersection gives area120/17 and requested12017; trained answer1902 is mathematically wrong, not extraction-only.','independent_fraction_check_passed':True},
        'interpretation':'11 of12 newlyincorrect responses hit8192-tokenlimit with no extractedbox; one normaltermination contains a real geometricerror. This diagnoses observed English loss as primarily failedtermination under fixedbudget, not a proof of underlying ability loss or recovery by extending budget. Do not change scores or attribute causality to clipping/teacher without intervention. One fixedseedpair only.','new_GPU_generation':False}
    with a.output.open('x') as stream:stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(changes))


if __name__=='__main__':main()
