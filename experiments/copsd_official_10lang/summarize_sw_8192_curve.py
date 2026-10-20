"""Audit one historical SW run at0/20/50/100 using the same8192 dev protocol."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

def read(p):
    return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    import torch
    configs=[];rows=[];metrics=[];training_args=[];gpu_hours=0.
    historical=a.root/'models/checkpoint/copsd_official_17b_single_v1/sw_17b_project2900_seed42_20261015_v1'
    for step in [0,20,50,100]:
        q='copsd-sw-base-dev8192-v1' if step==0 else f'copsd-sw-old{step}-dev8192-v1'
        state=read(a.root/'gpu-queue/queues'/q/'state.json')
        assert state['phase']=='completed' and state['job']['exit_code']==0
        j=state['job']
        assert len(j['gpu_uuids'])==1 and j['config']['max_user_gpus']==4
        assert hashlib.sha256(Path(j['script']).read_bytes()).hexdigest()==j['script_sha256']
        name='sw_base_dev100_8192_20261020_v1' if step==0 else f'sw_old{step}_dev100_8192_20261020_v1'
        d=a.root/'logs/copsd_official_10lang/development_vllm8192'/name
        c,m=read(d/'CONFIG.json'),read(d/'DONE.json')
        assert c['language']=='sw' and c['max_new_tokens']==8192 and c['samples']==1
        assert c['protocol']==m['protocol']=='project_dev_single_vllm8192_v2'
        assert c['max_num_seqs']==m['actual_max_num_seqs']==16
        assert hashlib.sha256(Path(__file__).with_name('evaluate_dev8192.py').read_bytes()).hexdigest()==c['evaluator_sha256']
        if step:
            cp=historical/f'checkpoint-{step}'
            assert Path(c['adapter']).resolve()==cp.resolve() and c['checkpoint_step']==step
            s=read(cp/'trainer_state.json');assert s['global_step']==step and s['max_steps']==100
            training_args.append(torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False))
            gpu_hours+=(datetime.datetime.fromisoformat(j['finished_at'])-datetime.datetime.fromisoformat(j['started_at'])).total_seconds()/3600
        else:
            assert c['adapter'] is None and c['checkpoint_step'] is None
        rr=[json.loads(x) for x in (d/'ITEMS.jsonl').read_text().splitlines()]
        assert len(rr)==len({r['id'] for r in rr})==m['items']==100
        assert not m['formal_test'] and m['n']==1
        assert all(len(r['samples'])==1 for r in rr)
        assert sum(r['samples'][0]['correct'] for r in rr)==m['single_correct']
        assert sum(len(r['samples'][0]['token_ids']) for r in rr)==m['generated_tokens']
        configs.append(c);metrics.append(m);rows.append(rr)
    canonical=[{k:v for k,v in c.items() if k not in ['adapter','checkpoint_step','output']} for c in configs]
    assert all(c==canonical[0] for c in canonical)
    for quartet in zip(*rows):
        for key in ['id','question','answer','prompt']:
            assert len({r[key] for r in quartet})==1
        assert len({r['samples'][0]['seed'] for r in quartet})==1
    fields=['seed','data_seed','learning_rate','max_steps','gradient_accumulation_steps','per_device_train_batch_size','optim','lr_scheduler_type','max_completion_length','max_length','beta','temperature']
    for key in fields:
        assert all(getattr(x,key)==getattr(training_args[0],key) for x in training_args),key
    scores={str(s):m['single_correct'] for s,m in zip([0,20,50,100],metrics)}
    pairs={}
    for i,j in [(0,1),(0,2),(0,3),(1,2),(2,3)]:
        x,y=[[r['samples'][0]['correct'] for r in rows[k]] for k in [i,j]]
        first,second=[[0,20,50,100][k] for k in [i,j]]
        pairs[f'{first}_to_{second}']={'gained':sum(not u and v for u,v in zip(x,y)),'lost':sum(u and not v for u,v in zip(x,y))}
    out={'observed_at':datetime.datetime.now().isoformat(),'protocol':'project_dev_single_vllm8192_v2','same_historical_SW_training_run':True,'correct_out_of100':scores,'metrics':dict(zip(['0','20','50','100'],metrics)),'paired_changes':pairs,'actual_parents_exit0_configs_inputs_prompts_seeds_and_raw_counts_verified':True,'saved_training_scientific_fields_match':fields,'cost':{'new_responses':300,'new_generated_tokens':sum(m['generated_tokens'] for m in metrics[1:]),'new_generation_seconds':sum(m['generation_seconds'] for m in metrics[1:]),'new_allocated_GPU_hours':gpu_hours,'cached_Base_excluded':True,'new_training':False},'scope':'Exploratory independent-development100, one historical training seed.0/20/50/100 are checkpoints of the same historical run, not the fresh SW20 loss-control run. Projectn1, not officialpass12/formal. No stable three-language selection.'}
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(scores))

if __name__=='__main__':
    main()
