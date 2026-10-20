"""Audit the completed one-factor clip control, without selecting test samples."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.home())
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    assert not a.output.exists()
    import torch
    queues=['copsd-sw-prefix20-match-v1','copsd-sw-clip-ablation20-v1']
    names=['sw_17b_prefix20_horizon100_seed42_20261020_v1',
           'sw_17b_prefix20_horizon100_unclipped_seed42_20261020_v1']
    tracks=['copsd_prefix20_replicates_v1','copsd_clip_ablation_v1']
    states=[read(a.root/'gpu-queue/queues'/q/'state.json') for q in queues]
    for s in states:
        assert s['phase']=='completed' and s['job']['exit_code']==0
        j=s['job'];assert len(j['gpu_uuids'])==1 and j['config']['max_user_gpus']==4
        assert hashlib.sha256(Path(j['script']).read_bytes()).hexdigest()==j['script_sha256']
    training=[];proofs=[];args=[]
    for track,name,s in zip(tracks,names,states):
        p=a.root/'models/checkpoint'/track/name;cp=p/'checkpoint-20'
        proof=read(p/'COLLOCATE_TRAIN_UPDATES_DONE.json');cost=read(p/'COST_SEGMENT_DONE.json')
        assert proof['global_step']==20 and proof['scheduler_horizon']==100
        assert proof['base_unchanged_bitwise'] and proof['micro_steps']==cost['loss_calls']==640
        assert len(proof['syncs'])==21
        assert read(cp/'trainer_state.json')['global_step']==20
        assert read(cp/'trainer_state.json')['max_steps']==100
        assert read(p/'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
        groups=read(p/'UPDATE_BATCH_GENERATION.json')['groups']
        assert len(groups)==20 and [g['global_step'] for g in groups]==list(range(20))
        assert sum(g['rollouts'] for g in groups)==640
        assert sum(g['generated_tokens'] for g in groups)==cost['valid_response_loss_positions']
        for f in ['optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors']:
            assert (cp/f).stat().st_size>0
        args.append(torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False))
        proofs.append(proof)
        job=s['job'];start=datetime.datetime.fromisoformat(job['started_at'])
        end=datetime.datetime.fromisoformat(job['finished_at'])
        training.append({'updates':20,'scheduler_horizon':100,'question_exposures':640,
            'generated_tokens':sum(g['generated_tokens'] for g in groups),
            'valid_response_loss_positions':cost['valid_response_loss_positions'],
            'training_clock_seconds':proof['train_seconds_including_diagnostic_hashing'],
            'pipeline_GPU_hours_including_startup_training_development':(end-start).total_seconds()/3600,
            'teacher_forward_cuda_seconds':cost['teacher_forward_cuda_seconds'],
            'student_forward_cuda_seconds':cost['student_forward_cuda_seconds']})
    assert proofs[0]['syncs'][0]['adapter_hash']==proofs[1]['syncs'][0]['adapter_hash']
    assert proofs[0]['syncs'][0]['engine_hash']==proofs[1]['syncs'][0]['engine_hash']
    fields=['learning_rate','max_grad_norm','per_device_train_batch_size',
        'gradient_accumulation_steps','max_steps','seed','data_seed','optim',
        'lr_scheduler_type','lr_scheduler_kwargs','warmup_steps','warmup_ratio',
        'weight_decay','adam_beta1','adam_beta2','adam_epsilon','gradient_checkpointing',
        'gradient_checkpointing_kwargs','bf16','fp16','max_length','max_completion_length']
    for field in fields:
        assert getattr(args[0],field)==getattr(args[1],field),field
    actual=read(a.root/'models/checkpoint'/tracks[1]/names[1]/'ABLATION_ACTUAL.json')
    assert actual['actual_component_clip'] is None and not actual['core_loss_replaced']
    root=a.root/'logs/copsd_official_10lang/development_vllm'
    devnames=['sw_17b_dev100_vllm_base_20261020_v1',
              'sw_17b_dev100_vllm_prefix20_seed42_20261020_v1',
              'sw_17b_dev100_vllm_prefix20_unclipped_seed42_20261020_v1']
    configs=[read(root/n/'CONFIG.json') for n in devnames]
    canonical=[{k:v for k,v in c.items() if k not in ['adapter','output']} for c in configs]
    assert canonical[0]==canonical[1]==canonical[2]
    assert configs[0]['adapter'] is None
    assert all(Path(c['adapter']).resolve()==(a.root/'models/checkpoint'/t/n/'checkpoint-20').resolve()
               for c,t,n in zip(configs[1:],tracks,names))
    rows=[[json.loads(l) for l in (root/n/'ITEMS.jsonl').read_text().splitlines()] for n in devnames]
    metrics=[read(root/n/'DONE.json') for n in devnames]
    for rr,m in zip(rows,metrics):
        assert len(rr)==len({r['id'] for r in rr})==m['items']==100
        assert not m['formal_test'] and m['n']==1
        assert sum(r['samples'][0]['correct'] for r in rr)==m['single_correct']
        assert sum(len(r['samples'][0]['token_ids']) for r in rr)==m['generated_tokens']
    for triplet in zip(*rows):
        assert all(all(r[k]==triplet[0][k] for k in ['id','question','answer','prompt']) for r in triplet)
        assert len({r['samples'][0]['seed'] for r in triplet})==1
    correct=[[r['samples'][0]['correct'] for r in rr] for rr in rows]
    b,c,u=correct
    out={'observed_at':datetime.datetime.now().isoformat(),'protocol':'project_SW_component_clip_one_factor_v1',
        'paired_development_correct_out_of100':dict(zip(['Base','clip0.05_20','unclipped_20'],[sum(x) for x in correct])),
        'unclipped_vs_clipped':{'gained':sum(not x and y for x,y in zip(c,u)),
            'lost':sum(x and not y for x,y in zip(c,u)),'delta_correct':sum(u)-sum(c)},
        'metrics':dict(zip(['Base','clipped','unclipped'],metrics)),
        'training_cost':dict(zip(['clipped','unclipped'],training)),
        'source_script_and_completed_exit0_verified':True,
        'initial_adapter_and_inference_weights_match':True,
        'scientific_training_args_compared':fields,
        'development_inputs_prompts_seeds_configs_match':True,
        'new_core_loss_implementation':False,'clip_only_scientific_factor':True,
        'scope':'One training seed, independent development100, prespecified20-update budget. '
            'Not a stable multilingual effect, formal-test selection or sequential forgetting result. '
            'Online trajectories and generated token costs may differ after policies diverge. Cached Base costs excluded.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'scores':out['paired_development_correct_out_of100'],
                      'pair':out['unclipped_vs_clipped']}),flush=True)


if __name__=='__main__':main()
