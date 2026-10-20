"""Audit target/English8192 development for the preselected saved ZH20/FR20."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def load_eval(root, queue, run, lang, adapter):
    s=read(root/'gpu-queue/queues'/queue/'state.json')
    assert s['phase']=='completed' and s['job']['exit_code']==0,queue
    j=s['job']
    assert len(j['gpu_uuids'])==1 and j['config']['max_user_gpus']==4
    assert hashlib.sha256(Path(j['script']).read_bytes()).hexdigest()==j['script_sha256']
    d=root/'logs/copsd_official_10lang/development_vllm8192'/run
    c,m=read(d/'CONFIG.json'),read(d/'DONE.json')
    assert c['protocol']==m['protocol']=='project_dev_single_vllm8192_v2'
    assert c['language']==lang and c['max_new_tokens']==8192 and c['samples']==m['n']==1
    assert c['max_num_seqs']==m['actual_max_num_seqs']==16
    assert hashlib.sha256(Path(__file__).with_name('evaluate_dev8192.py').read_bytes()).hexdigest()==c['evaluator_sha256']
    if adapter is None:
        assert c['adapter'] is None and c['checkpoint_step'] is None
    else:
        assert Path(c['adapter']).resolve()==adapter.resolve() and c['checkpoint_step']==20
        state=read(adapter/'trainer_state.json')
        assert state['global_step']==20 and state['max_steps']==100
    rr=[json.loads(x) for x in (d/'ITEMS.jsonl').read_text().splitlines()]
    assert len(rr)==len({x['id'] for x in rr})==m['items']==100 and not m['formal_test']
    assert all(len(x['samples'])==1 for x in rr)
    assert sum(x['samples'][0]['correct'] for x in rr)==m['single_correct']
    assert sum(len(x['samples'][0]['token_ids']) for x in rr)==m['generated_tokens']
    assert all(len(x['samples'][0]['token_ids'])==x['samples'][0]['tokens'] for x in rr)
    gpu_hours=(datetime.datetime.fromisoformat(j['finished_at'])-datetime.datetime.fromisoformat(j['started_at'])).total_seconds()/3600
    return c,m,rr,gpu_hours


def pair(first,second):
    c0,m0,r0,_=first;c1,m1,r1,_=second
    configs=[{k:v for k,v in c.items() if k not in ['adapter','checkpoint_step','output']} for c in [c0,c1]]
    assert configs[0]==configs[1]
    for x,y in zip(r0,r1):
        assert all(x[k]==y[k] for k in ['id','question','answer','prompt'])
        assert x['samples'][0]['seed']==y['samples'][0]['seed']
    x,y=[[r['samples'][0]['correct'] for r in rr] for rr in [r0,r1]]
    return {'base_correct':m0['single_correct'],'adapter20_correct':m1['single_correct'],
        'gained':sum(not a and b for a,b in zip(x,y)),'lost':sum(a and not b for a,b in zip(x,y)),
        'delta_correct':sum(y)-sum(x)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--language',choices=['zh','fr'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();root,lang=a.root,a.language
    frozen=read(Path(__file__).with_name('zh_fr_saved_early20_dev8192_frozen_20261020.PUBLIC.json'))
    cp=root/f'models/checkpoint/copsd_vllm_seeded_cost_single_v2/{lang}_17b_project2900_vllm_seeded_cost_seed42_20261020_v2/checkpoint-20'
    runs={}
    for label in ['base','early20','early20_en']:
        name=f'{lang}_{label}_candidate_dev100_8192_20261020_v1'
        queue=f'copsd-{lang}-{label.replace("_","-")}-candidate-dev8192-v1'
        assert any(j['queue']==queue and j['run_id']==name for j in frozen['jobs'])
        runs[label]=load_eval(root,queue,name,'en' if label.endswith('_en') else lang,None if label=='base' else cp)
    runs['English_Base']=load_eval(root,'copsd-en-sw-base-dev8192-v1','en_sw_base_dev100_8192_20261020_v1','en',None)
    for quartet in zip(*[v[2] for v in runs.values()]):
        assert len({r['id'] for r in quartet})==len({r['answer'] for r in quartet})==1
    pairs={'target':pair(runs['base'],runs['early20']), 'English':pair(runs['English_Base'],runs['early20_en'])}
    costs={k:v for k,v in runs.items() if k!='English_Base'}
    out={'observed_at':datetime.datetime.now().isoformat(),'protocol':'project_dev_single_vllm8192_v2',
        'training_language':lang,'checkpoint_step':20,'paired_development100':pairs,
        'metrics':{k:v[1] for k,v in runs.items()},
        'actual_parents_exit0_fixed_checkpoints_raw_counts_configs_prompts_and_seeds_verified':True,
        'common_original_IDs_gold_across_target_English_verified':True,
        'cost':{'new_responses':300,'new_generated_tokens':sum(v[1]['generated_tokens'] for v in costs.values()),
            'new_generation_seconds':sum(v[1]['generation_seconds'] for v in costs.values()),
            'new_allocated_GPU_hours':sum(v[3] for v in costs.values()),'cached_English_Base_excluded':True,'new_training':False},
        'scope':'Exploratory budget correction of previously development-selected saved20-update checkpoints; one training seed. Not final three-language selection, formal test, officialpass12, or sequential evidence.'}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(pairs))


if __name__=='__main__':
    main()
