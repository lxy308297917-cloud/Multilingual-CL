"""CPU audit of saved valid-token lengths and illustrative reduction weights.

Contiguous groups of4 illustrate releasedmicro4 normalization, not reconstruction
of the released four-rank sampler or a parameter-gradient/efficacy measurement.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    code=Path(__file__).parent;tokenizer=AutoTokenizer.from_pretrained(a.root/'models/Qwen3-4B',local_files_only=True);results={}
    for label,name in [('Clip20','bn4b_server20_frozen_20261023.PUBLIC.json'),('Unclip20','bn4b_unclip20_frozen_20261023.PUBLIC.json')]:
        f=read(code/name);folder=a.root/f['checkpoint_relative'];path=folder/'SERVER_ROLLOUTS.PRIVATE.jsonl'
        with path.open() as stream:rr=[json.loads(s) for s in stream]
        assert len(rr)==640 and [r['global_step'] for r in rr]==[i//32 for i in range(640)]
        length=np.array([sum(t!=tokenizer.pad_token_id for t in r['completion_ids']) for r in rr]);assert (length>0).all()
        proof=read(folder/'SERVER_PILOT20_UPDATES_DONE.json');assert int(length.sum())==proof['cost']['valid_response_loss_positions']
        ratio=[];tv=[]
        for start in range(0,640,32):
            ll=length[start:start+32];group_weights=(ll.reshape(8,4)/ll.reshape(8,4).sum(1,keepdims=True)/8).flatten()
            ratio.extend((np.full(32,1/32)/group_weights).tolist());tv.append(float(np.abs(np.full(32,1/32)-group_weights).sum()/2))
        results[label]={'rows':640,'valid_loss_tokens':int(length.sum()),'full2048_valid_tokens':int(sum(length==2048)),'shorter_than2048':int(sum(length<2048)),
            'length_quantiles_min_p10_median_p90_max':np.quantile(length,[0,.1,.5,.9,1]).tolist(),
            'illustrative_micro1_to_contiguous_group4_sample_mean_weight_ratio_quantiles':np.quantile(ratio,[0,.1,.5,.9,1]).tolist(),
            'illustrative_per_update_sample_weight_total_variation_max':max(tv),'illustrative_per_update_sample_weight_total_variation_mean':float(np.mean(tv)),
            'sample_weight_ratio_outside_0_9_to_1_1':sum(x<.9 or x>1.1 for x in ratio),'first32_valid_lengths':length[:32].tolist(),'source_rollouts_sha256':sha(path)}
    report={'updated_at':datetime.datetime.now().isoformat(),'protocol':'BN4B_saved640_response_denominator_audit','conditions':results,
        'official_source_sha256':sha(a.root/'COPSD-official-20261015/multilingual_opsd_trainer.py'),
        'released_launcher_sha256':sha(a.root/'COPSD-official-20261015/multilingual_scripts/run_all_opsd_4b_3000.sh'),
        'released_micro_batch':4,'released_world_size':4,'released_gradient_accumulation':2,'project_micro_batch':1,'project_world_size':1,'project_gradient_accumulation':32,
        'same_effective_batch_not_same_per_example_weights':True,'interpretation':'Mostresponses haveidentical2048valid lengths, so the analytic reduction difference is localized to groups containingshort responses. This doesnot prove causality or exactreconstruction of releasedrank-group sampler. No correctedtraining or downstream effect claimed. Length2048 is hittingbudget, not byitself proof of missingfinalanswer.',
        'analysis_sha256':sha(Path(__file__))}
    with a.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{x:v[x] for x in ['full2048_valid_tokens','shorter_than2048','illustrative_per_update_sample_weight_total_variation_max','sample_weight_ratio_outside_0_9_to_1_1']} for k,v in results.items()}))


if __name__=='__main__':main()
