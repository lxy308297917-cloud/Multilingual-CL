"""Accept only a terminal, complete dedicated-server4B resource run, not efficacy."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import torch


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();assert not a.output.exists()
    code=Path(__file__).parent;f=read(code/'sw4b_server_resource_v2_frozen_20261022.PUBLIC.json')
    state=read(a.root/'gpu-queue/queues/copsd-sw4b-server-resource-v2/state.json')
    assert state['phase']==state['job']['phase']=='completed' and state['job']['exit_code']==0
    output=a.root/f['checkpoint_relative'];proof=read(output/'SERVER_RESOURCE_UPDATES_DONE.json');parent=read(output/'SERVER_PARENT_DONE.json')
    assert parent['two_role_job_completed'] and parent['training_exit_code']==0 and parent['server_cleanup_completed']
    assert proof['global_step']==proof['completed_updates']==2 and proof['micro_steps']==64
    assert proof['base_unchanged_bitwise'] and proof['all_HF_parameters_on_CUDA'] and len(proof['syncs'])==3
    assert len({r['adapter_hash'] for r in proof['syncs']})==3
    assert proof['memory_reduction']['position_reduction_calls']==64
    cost=proof['cost'];assert cost['loss_calls']==cost['teacher_forward_calls']==cost['student_forward_calls']==64
    receipts=a.root/'logs/copsd_official_10lang'/(f['run_id']+'.remote_weights.PRIVATE.jsonl')
    rows=[json.loads(line) for line in receipts.read_text().splitlines()]
    assert len(rows)==sum(r['parameter_count'] for r in proof['syncs']) and all(r['original_loader_returned'] for r in rows)
    at=0
    for sync in proof['syncs']:
        selected=rows[at:at+sync['parameter_count']];at+=sync['parameter_count']
        assert hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()==sync['export_and_remote_loaded_sha256']
    rollouts=[json.loads(line) for line in (output/'SERVER_ROLLOUTS.PRIVATE.jsonl').read_text().splitlines()]
    prepared=read(code/'sw4b_resource2_prepared_20261022.PUBLIC.json');mapping=read(a.root/prepared['prompt_identity_relative'])
    assert len(rollouts)==len({r['student_prompt_sha256'] for r in rollouts})==64
    assert {mapping[r['student_prompt_sha256']]['id'] for r in rollouts}==set(prepared['selected_IDs'])
    assert set(prepared['stress_IDs'])<={mapping[r['student_prompt_sha256']]['id'] for r in rollouts}
    lengths=[len(r['completion_ids']) for r in rollouts];assert max(lengths)==2048 and sum(lengths)==cost['valid_response_loss_positions']
    checkpoint=[]
    for step in [1,2]:
        cp=output/f'checkpoint-{step}';assert read(cp/'trainer_state.json')['global_step']==step
        assert torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)['last_epoch']==step
        assert {int(v['step']) for v in torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)['state'].values()}=={step}
        assert (cp/'rng_state.pth').stat().st_size>0
        args=torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False)
        assert args.max_steps==100 and args.seed==args.data_seed==42 and args.gradient_accumulation_steps==32
        assert args.learning_rate==5e-6 and 'adamw' in str(args.optim).lower()
        checkpoint.append({'step':step,'optimizer_scheduler_RNG_present':True,'adapter_sha256':sha(cp/'adapter_model.safetensors')})
    memories=[]
    for i,uuid in enumerate(state['job']['gpu_uuids']):
        path=a.root/'logs/copsd_official_10lang'/(f['run_id']+f'.gpu_role{i}.csv')
        peak=max(float(line.split(',')[1]) for line in path.read_text().splitlines() if line.strip())
        total=float(subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
        assert peak<total
        memories.append({'role':'training_and_teacher' if i==0 else 'student_rollout_server','physical_peak_MiB':peak,'sampled_headroom_MiB':total-peak,'telemetry_sha256':sha(path)})
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb=a.root/'logs/copsd_official_10lang/tensorboard'/f['run_id'];acc=EventAccumulator(str(tb)).Reload()
    loss=acc.Scalars('train/loss');assert {1,2}<={r.step for r in loss}
    start=datetime.datetime.fromisoformat(state['job']['started_at']);end=datetime.datetime.fromisoformat(state['job']['finished_at'])
    result={'protocol':'SW4B_official_server_resource2_v2_completed','engineering_completed':True,'completed_at':state['job']['finished_at'],
        'frozen_sha256':sha(code/'sw4b_server_resource_v2_frozen_20261022.PUBLIC.json'),'model_revision':f['model_revision'],
        'completed_updates':2,'actual_exposures':64,'scheduler_horizon':100,'base_unchanged_bitwise':True,
        'teacher_all_on_training_GPU':True,'actual_remote_load_receipts_match_all_exported_parameters':True,'weight_sync_count':3,
        'three_distinct_student_adapters':True,'generated_tokens':sum(lengths),'responses_at2048':lengths.count(2048),'cost':cost,
        'batch_generation':proof['batch_generation'],'checkpoint_proofs':checkpoint,'memory_per_role':memories,
        'train_seconds_including_diagnostic_hashing':proof['train_seconds_including_diagnostic_hashing'],
        'allocated_parent_GPU_hours':(end-start).total_seconds()*2/3600,'TensorBoard_train_loss_steps':[r.step for r in loss],
        'checkpoint_actual_resume_tested':False,'fused_layout_full_logit_equivalence_tested':False,'downstream_4B_mathematical_effect_tested':False,
        'scope':'Engineering gate only. Full64 stresspool and2048 retained; not100 updates or efficacy. AnySW20 must start fresh fromoriginal4B with a new protocol and matching4BBase, not debugadapter or failed singleGPU pilot.'}
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
