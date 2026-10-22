"""Persist explicit assistant adjudication of an already inspected packet range."""
import argparse,time
from settings import *
p=argparse.ArgumentParser();p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True);p.add_argument('--correct',default='');p.add_argument('--unclear',default='');a=p.parse_args()
source=[r for r in rows(RUN/'comparison.jsonl') if r['task']=='en_math'];ids={r['id'] for r in source[a.start-1:a.end]};packet=json.loads((RUN/'POSTHOC_MATH_ANONYMOUS_PACKET.json').read_text());cases=[r for r in packet if r['source_id'] in ids];correct=set(a.correct.split()) if a.correct else set();unclear=set(a.unclear.split()) if a.unclear else set();allcodes={r['code'] for r in cases};assert correct|unclear<=allcodes and not correct&unclear
path=RUN/'POSTHOC_MATH_ASSISTANT_LABELS.json';labels=json.loads(path.read_text()) if path.exists() else {};assert not allcodes&labels.keys(),'Do not silently revise already adjudicated responses'
for r in cases:
    code=r['code'];label='correct' if code in correct else 'unclear' if code in unclear else 'wrong'
    labels[code]={'label':label,'reviewed_unix':time.time(),'source_id':r['source_id'],'reviewer':'assistant, final requested-answer review; not independent human gold','basis':'Inspected question and final response text; clear final target matches source-QA gold' if label=='correct' else 'Insufficient or conflicting requested answer' if label=='unclear' else 'Inspected response gives incorrect target or clear erroneous requested quantity','long_output_review':'Last1400 characters initially; full output inspected when ending/context ambiguous; this does not validate all reasoning steps'}
write(path,labels);print('Saved',len(cases),'unique adjudications; total',len(labels))
