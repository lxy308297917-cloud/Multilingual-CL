"""Fixed paired cluster bootstrap; no mixed-task average or endpoint selection."""
import argparse,json
from pathlib import Path
import numpy as np
from calibration import ROOT,sha,write
from development_scoring import summarize
PRIMARY={'en_math':'numeric_equivalence','en_instruction':'prompt_strict','en_reading':'strict_letter','sw_reading_kenswquad':'token_f1'}

def paired_ci(left,right,*,resamples=10000,seed=42):
    if [r['id'] for r in left]!=[r['id'] for r in right]:raise ValueError('Different ordered sources')
    if any(a['group_id']!=b['group_id'] or a['task']!=b['task'] for a,b in zip(left,right)):raise ValueError('Cluster/task mismatch')
    if len({r['task'] for r in left})!=1:raise ValueError('No mixed-task effect')
    groups={}
    for i,r in enumerate(left):groups.setdefault(r['group_id'],[]).append(i)
    clusters=list(groups.values());rng=np.random.default_rng(seed);task=left[0]['task']
    if 'chrfpp_stats' in left[0]['scores']:
        from sacrebleu.metrics import CHRF
        metric=CHRF(char_order=6,word_order=2,beta=2)
        l=np.array([r['scores']['chrfpp_stats'] for r in left]);r=np.array([v['scores']['chrfpp_stats'] for v in right])
        l=np.array([l[ix].sum(axis=0) for ix in clusters]);r=np.array([r[ix].sum(axis=0) for ix in clusters])
        def measure(a,b):return metric._compute_score_from_stats(a.tolist()).score-metric._compute_score_from_stats(b.tolist()).score
        point=measure(l.sum(axis=0),r.sum(axis=0));delta=[]
        for _ in range(resamples):
            ix=rng.integers(len(clusters),size=len(clusters));delta.append(measure(l[ix].sum(axis=0),r[ix].sum(axis=0)))
        metric_name='corpus_chrfpp'
    else:
        key=PRIMARY[task];l=np.array([v['scores'][key] for v in left]);r=np.array([v['scores'][key] for v in right]);sums=np.array([(l[ix]-r[ix]).sum() for ix in clusters]);sizes=np.array([len(ix) for ix in clusters]);point=100*(l-r).mean();delta=[]
        for _ in range(resamples):
            ix=rng.integers(len(clusters),size=len(clusters));delta.append(100*sums[ix].sum()/sizes[ix].sum())
        metric_name=key
    low,high=np.quantile(delta,[.025,.975]);return {'metric':metric_name,'delta_left_minus_right':point,'paired95percentCI':[float(low),float(high)],'resamples':resamples,'seed':seed,'clusters':len(clusters),'n':len(left),'cluster_unit':'frozen source group_id; FLORES documentURL hash'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--endpoint',action='append',required=True,help='label=completed development directory');ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('No report overwrite')
    protocol=json.loads((ROOT/'comparison_protocol.v1.json').read_text());by={};receipts={}
    for arg in args.endpoint:
        label,path=arg.split('=',1);folder=Path(path);done=json.loads((folder/'DONE.json').read_text())
        if done['label']!=label or done['protocol_sha256']!=sha(ROOT/'comparison_protocol.v1.json') or sha(folder/'PREDICTIONS.jsonl')!=done['predictions_sha256']:raise ValueError('Endpoint receipt changed')
        rows=[json.loads(s) for s in (folder/'PREDICTIONS.jsonl').read_text().splitlines()]
        by[label]={task:[r for r in rows if r['task']==task] for task in {r['task'] for r in rows}};receipts[label]={'done_sha256':sha(folder/'DONE.json'),'scores':done['scores']}
    required=['Base','A2048','D2048','R2048']
    if any(k not in by for k in required):raise ValueError('Incomplete mandatory matched endpoints; no efficacy decision')
    pairs=[('D2048','A2048'),('D2048','R2048'),('A2048','Base'),('D2048','Base'),('R2048','Base')]
    optional=all(k in by for k in ('E2048','F2048','S2048'))
    if optional:pairs += [('F2048',k) for k in ('E2048','S2048','R2048')]
    deltas={a+'-'+b:{task:paired_ci(by[a][task],by[b][task],resamples=protocol['development']['bootstrap']['resamples'],seed=protocol['development']['bootstrap']['seed']) for task in by[a]} for a,b in pairs}
    required_sw={'sw_reading_kenswquad':-2.,'sw_summary':-1.,'en_to_sw':-1.,'sw_to_en':-1.}
    def passes(a,controls):
        retention=any(all(deltas[a+'-'+b][task]['paired95percentCI'][0]>0 for b in controls) for task in protocol['decision']['retention_primaries'])
        sw=all(deltas[a+'-'+b][task]['paired95percentCI'][0]>=limit for b in controls for task,limit in required_sw.items())
        return {'retention_same_primary_beats_all_controls':retention,'all_SW_noninferiority_bounds_pass':sw,'screening_pass':retention and sw}
    decision={'D':passes('D2048',['A2048','R2048']),'pairing':'not tested' if not optional else passes('F2048',['E2048','S2048','R2048']),'long_training':'not released: strong matched baseline and unused confirmation required even if screening passes','negative_action':'stop expansion when screen fails; report all task effects'}
    write(out,{'protocol_sha256':sha(ROOT/'comparison_protocol.v1.json'),'endpoints':receipts,'paired_deltas':deltas,'decision':decision,'formal_test_used':False,'no_positive_claim_from_small_dev_alone':True})
if __name__=='__main__':main()
