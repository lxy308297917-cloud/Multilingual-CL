"""Full historical step-log coverage. Gradient loss magnitudes are not strengths."""
import numpy as np
from settings import *
def stats(x):
    x=np.asarray(x,dtype=float)
    return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max()),'q10':float(np.quantile(x,.1)),'q90':float(np.quantile(x,.9))}
result={}
for method in 'ABCD':
    path=TRAIN/'comparison_v3'/method/'STEPS.jsonl';rr=rows(path);assert [r['step'] for r in rr]==list(range(1,4294));aux=[r for r in rr if r['step']%4==0];other=[r for r in rr if r['step']%4]
    assert method=='A' or all(r['aux_loss']>0 for r in aux)
    config=json.loads((path.parent/'CONFIG.json').read_text());assert sha(OLD/'representation_alignment_v1/train.py')==config['code_sha256'];assert sha(OLD/'representation_alignment_v1/common.py')==config['common_sha256']
    result[method]={'steps':len(rr),'aux_events':0 if method=='A' else len(aux),'log_sha256':sha(path),'code_matches_executed_config':True,'aux_loss_quarters':[stats([r['aux_loss'] for r in aux if lo<r['step']<=hi]) for lo,hi in [(0,1073),(1073,2146),(2146,3219),(3219,4293)]],'global_grad_norm_aux_steps':stats([r['grad_norm'] for r in aux]),'global_grad_norm_non_aux':stats([r['grad_norm'] for r in other]),'global_clip_active_fraction':sum(r['grad_norm']>1 for r in rr)/len(rr),'cumulative_counts':rr[-1]['counts']}
write(REPORT/'HISTORICAL_LOG_AUDIT.json',{'methods':result,'limits':['Logged auxiliary value is unweighted; same0.1 does not equalize gradient or effective update','Quarter losses sample different sentence pairs and learning states; not paired same-input evidence','BF16 full-trajectory auxiliary survival cannot be established from step logs alone']})
print(json.dumps({m:{'aux_events':v['aux_events'],'clip_fraction':v['global_clip_active_fraction'],'loss_quarters':[x['mean'] for x in v['aux_loss_quarters']]} for m,v in result.items()}))
