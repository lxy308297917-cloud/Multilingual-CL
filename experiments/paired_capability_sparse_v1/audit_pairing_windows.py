"""Check pairing identifiability in the actual two-pair calibration windows.

Global mixed mastery strata do not imply that per-window normalized E/F differ.
This is a necessary algebraic gate, not evidence that actual QP updates differ.
"""
import argparse,json
from pathlib import Path
import torch
from core import pair_weights
from calibration import ROOT,sha,write

def audit_windows(en,sw,permutation):
    outputs={name:pair_weights(en,sw,'paired' if name!='E' else 'independent',permutation if name=='S' else None) for name in ('E','F','S')}
    rows=[];size=len(en)
    for first in range(size):
        ix=[first,(first+1)%size];weights={}
        for method,(a,b) in outputs.items():
            weights[method]=[]
            for w in (a,b):
                x=w[ix].float();weights[method].append(x/x.sum().clamp_min(1.))
        equal=lambda a,b:all(torch.equal(x,y) for x,y in zip(weights[a],weights[b]))
        rows.append({'indices':ix,'E_F_equal':equal('E','F'),'F_S_equal':equal('F','S'),'normalized_weights':{k:[x.tolist() for x in v] for k,v in weights.items()}})
    return {'windows':rows,'E_F_different_windows':sum(not r['E_F_equal'] for r in rows),'F_S_different_windows':sum(not r['F_S_equal'] for r in rows),'necessary_weight_identifiability':any(not r['E_F_equal'] for r in rows) and any(not r['F_S_equal'] for r in rows),'limits':'Nonzero weight difference does not establish distinct gradients or active constraints or actual BF16 updates'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Do not overwrite gate evidence')
    state=json.loads(Path(args.cache).read_text());data=[json.loads(s) for s in (ROOT/'data/calibration_pairs.v1.jsonl').read_text().splitlines()];protocol=json.loads((ROOT/'comparison_protocol.v1.json').read_text())
    en=[state['records'][r['id']+':en']['known'] for r in data];sw=[state['records'][r['id']+':sw']['known'] for r in data];result=audit_windows(en,sw,protocol['pairing']['shuffled_partner_permutation'])
    write(out,{'cache_sha256':sha(args.cache),'protocol_sha256':sha(ROOT/'comparison_protocol.v1.json'),'joint_known':sum(a and b for a,b in zip(en,sw)),'EN_only':sum(a and not b for a,b in zip(en,sw)),'actual_training_window':'two adjacent pair IDs, starting(2*step)mod29; all29 starts occur','no_dev_scores_used':True,**result})
if __name__=='__main__':main()
