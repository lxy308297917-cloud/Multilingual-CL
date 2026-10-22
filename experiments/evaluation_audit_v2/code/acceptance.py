"""Freeze v2 scorer only after independent-source controlled acceptance."""
import time
from settings import *
from scoring_v2 import *

def main():
    manifest=json.loads((RUN/'DATA_FROZEN.json').read_text());info=manifest['partitions']['acceptance'];assert sha(info['path'])==info['sha256']
    cases=[]
    def add(row,text,expected,kind):
        got=score(row,text);metric='content_f1' if row['task']=='sw_reading' else 'content_correct'
        value=got[metric];passed=(value==1 if expected else value==0)
        cases.append({'source_id':row['id'],'task':row['task'],'text':text,'expected_full_correct':expected,'kind':kind,'score':got,'passed':passed})
    for row in rows(info['path']):
        if row['task']=='en_reading':
            g=row['reference'];bad=next(x for x in 'ABCD' if x!=g);opt=row['options'][ord(g)-65]
            for text in (g,g+'. '+opt,'The answer is '+g+'.',opt):add(row,text,True,'correct_format_variant')
            for text in (bad,bad+'. '+row['options'][ord(bad)-65],g+'. '+row['options'][ord(bad)-65],g+' or '+bad,'Answer: '+g+', but actually '+bad,'Answer: '+g+'\nAnswer: '+bad,'\n'.join(x+'. '+y for x,y in zip('ABCD',row['options']))):add(row,text,False,'wrong_or_contradictory')
        elif row['task']=='en_math':
            g=row['reference'];v,_=numeric(g);bad=str(v+1)
            for text in (g,'The answer is '+g+'.',r'\boxed{'+g+'}',row['source_worked_answer']):add(row,text,True,'correct_natural_or_format_variant')
            for text in (r'\boxed{'+bad+'}','Answer: '+g+'\nAnswer: '+bad,'We considered '+g+' and '+bad+' in the calculation.','Answer: '+g+' or '+bad):add(row,text,False,'wrong_or_ambiguous')
        else:
            for text in (row['reference'],'Jibu: '+row['reference'],row['reference'].upper()+'!'):add(row,text,True,'correct_format_variant')
            add(row,'QZXV',False,'unrelated_wrong_content')
    # Exact numeric equivalence and unsupported/ambiguous expressions are explicit.
    synthetic=[('1/2','0.5',True),(r'\boxed{\frac{1}{2}}','0.5',True),('-0.25','-1/4',True),('1,000','1000',True),('Answer: 7 dollars','$7',True),('Answer: 700 cents','$7',True),('Answer: 7 cents','$7',False),('Answer: 60 minutes','1 hour',True),('Answer: 1 meter','100 centimeters',True),('Answer: 25%','25%',True),('Answer: 0.25','25%',False),('Answer: 2+3','5',False),('Answer: 7 or 8','7',False),('Step 1: 7. Step 2: 8.','8',False)]
    for i,(text,gold,expect) in enumerate(synthetic):add({'id':f'controlled:{i}','task':'en_math','reference':gold},text,expect,'controlled_equivalence')
    failed=[c for c in cases if not c['passed']]
    write(RUN/'ACCEPTANCE_CASES.json',cases)
    report={'total_cases':len(cases),'failed_cases':len(failed),'false_positive':sum(not c['expected_full_correct'] and c['score'].get('content_correct',c['score'].get('content_f1',0))>0 for c in cases),'missed_correct':sum(c['expected_full_correct'] and not c['passed'] for c in cases),'source_groups':36,'cases_sha256':sha(RUN/'ACCEPTANCE_CASES.json'),'scorer_sha256':sha(pathlib.Path(__file__).parent/'scoring_v2.py'),'data_manifest_sha256':sha(RUN/'DATA_FROZEN.json'),'frozen_unix':time.time(),'limits':['Controlled acceptance is finite, not a human gold evaluation of all future model answers','Assistant-reviewed test design; no independent human reviewer','Numeric-only reference cannot verify unit semantics; unit availability is separately recorded'],'failed_case_kinds':[c['kind'] for c in failed]}
    write(REPORT/'ACCEPTANCE_SUMMARY.json',report)
    if failed:
        print(json.dumps(failed[:12],ensure_ascii=False,indent=2));raise RuntimeError(f'{len(failed)} scorer acceptance failures; do not run model comparison')
    assert not (RUN/'ACCEPTANCE_FROZEN.json').exists()
    write(RUN/'ACCEPTANCE_FROZEN.json',report);print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
