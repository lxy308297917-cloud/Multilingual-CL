"""Screen existing20-update checkpoints at one fixed target/English dev budget.

No training or formal-test selection. Every child uses the unchanged frozen
8192 evaluator and retains its own CONFIG, raw ITEMS and cost records.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--language',choices=['de','es','ru','th','te'],required=True)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();root,lang=a.root,a.language;directory=Path(__file__).parent
    frozen=read(directory/'remaining_saved20_screen8192_frozen_20261020.PUBLIC.json')
    for name,sha in frozen['code_sha256'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==sha,name
    selected=frozen['languages'][lang]
    cp=root/'models/checkpoint'/selected['checkpoint_relative']
    assert cp.resolve().is_relative_to((root/'models/checkpoint').resolve())
    state=read(cp/'trainer_state.json')
    assert state['global_step']==20 and state['max_steps']==100
    assert hashlib.sha256((cp/'adapter_model.safetensors').read_bytes()).hexdigest()==selected['adapter_sha256']
    data=root/f'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_{lang}.json'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==selected['development_sha256']
    rows=read(data);english=read(root/'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json')
    assert len(rows)==len({x['problem_id'] for x in rows})==100
    assert all(all(x[k]==y[k] for k in ['problem_id','problem','Answer']) for x,y in zip(rows,english))
    python=root/'envs/copsd-vllm011-20261018/bin/python'
    jobs=[]
    for role in ['base','target20','English20']:
        language='en' if role=='English20' else lang
        run=selected['run_ids'][role]
        args=['--root',str(root),'--language',language,'--run-id',run]
        if role!='base':args+=['--adapter',str(cp),'--expected-step','20']
        subprocess.run([str(python),str(directory/'evaluate_dev8192.py'),*args,'--prepare-only'],check=True)
        jobs.append((role,language,run))
    if a.prepare_only:
        print(f'{lang}: all source/data/adapter/output identities verified; no GPU initialized.',flush=True)
        return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT')=='1'
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    assert visible.startswith('GPU-') and ',' not in visible
    for role,language,run in jobs:
        print('PIPELINE_PHASE',lang,role,flush=True)
        args=['bash',str(directory/'run_dev8192.sh'),language,run]
        if role!='base':args+=[str(cp),'20']
        subprocess.run(args,check=True)
    print('SCREEN20_TARGET_EN_DONE',lang,'Parent must separately exit0.',flush=True)


if __name__=='__main__':
    main()
