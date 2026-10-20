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
    queues=['copsd-bn-clip20-dev8192-v1','copsd-bn-unclip20-dev8192-v1']
    names=['bn_17b_clip20_horizon100_seed42_20261020_v1',
           'bn_17b_unclip20_horizon100_seed42_20261020_v1']
    tracks=['copsd_bn_clip_control_v1','copsd_bn_clip_control_v1']
    states=[read(a.root/'gpu-queue/queues'/q/'state.json') for q in queues]
    for s in states:
        assert s['phase']=='completed' and s['job']['exit_code']==0
        j=s['job'];assert len(j['gpu_uuids'])==1 and j['config']['max_user_gpus']==4
        assert hashlib.sha256(Path(j['script']).read_bytes()).hexdigest()==j['script_sha256']
    training=[];proofs=[];args=[];adapter_configs=[]
    for track,name,s in zip(tracks,names,states):
        p=a.root/'models/checkpoint'/track/name;cp=p/'checkpoint-20'
        proof=read(p/'COLLOCATE_TRAIN_UPDATES_DONE.json');cost=read(p/'COST_SEGMENT_DONE.json')
        assert proof['global_step']==20 and proof['scheduler_horizon']==100
        assert proof['base_unchanged_bitwise'] and proof['micro_steps']==cost['loss_calls']==640
        assert proof['max_completion_length']==2048 and proof['effective_batch']==32
        assert cost['loss_temperature']==1.1 and cost['seed_applied_before_model_and_PEFT']==42
        assert len(proof['syncs'])==21
        assert read(cp/'trainer_state.json')['global_step']==20
        assert read(cp/'trainer_state.json')['max_steps']==100
        assert read(p/'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
        groups=read(p/'UPDATE_BATCH_GENERATION.json')['groups']
        assert len(groups)==20 and [g['global_step'] for g in groups]==list(range(20))
        assert sum(g['rollouts'] for g in groups)==640
        generated_tokens=sum(g['generated_tokens'] for g in groups)
        assert 0<cost['valid_response_loss_positions']<=generated_tokens
        for f in ['optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors']:
            assert (cp/f).stat().st_size>0
        args.append(torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False))
        adapter_configs.append(read(cp/'adapter_config.json'))
        proofs.append(proof)
        job=s['job'];start=datetime.datetime.fromisoformat(job['started_at'])
        end=datetime.datetime.fromisoformat(job['finished_at'])
        training.append({'updates':20,'scheduler_horizon':100,'question_exposures':640,
            'generated_tokens':sum(g['generated_tokens'] for g in groups),
            'valid_response_loss_positions':cost['valid_response_loss_positions'],
            'generation_tokens_minus_valid_loss_positions':generated_tokens-cost['valid_response_loss_positions'],
            'training_clock_seconds':proof['train_seconds_including_diagnostic_hashing'],
            'pipeline_GPU_hours_including_startup_training_development':(end-start).total_seconds()/3600,
            'teacher_forward_cuda_seconds':cost['teacher_forward_cuda_seconds'],
            'student_forward_cuda_seconds':cost['student_forward_cuda_seconds']})
    assert proofs[0]['syncs'][0]['adapter_hash']==proofs[1]['syncs'][0]['adapter_hash']
    assert proofs[0]['syncs'][0]['engine_hash']==proofs[1]['syncs'][0]['engine_hash']
    # PEFT serializes target_modules from a set; JSON list order can differ.
    adapter_serialization_differences=[k for k in adapter_configs[0]
        if adapter_configs[0].get(k)!=adapter_configs[1].get(k)]
    canonical_adapters=[]
    for config in adapter_configs:
        canonical=dict(config)
        for field in ['target_modules','exclude_modules','modules_to_save']:
            if isinstance(canonical.get(field),list):canonical[field]=sorted(canonical[field])
        canonical_adapters.append(canonical)
    assert canonical_adapters[0]==canonical_adapters[1]
    fields=['learning_rate','max_grad_norm','per_device_train_batch_size',
        'gradient_accumulation_steps','max_steps','seed','data_seed','optim',
        'lr_scheduler_type','lr_scheduler_kwargs','warmup_steps','warmup_ratio',
        'weight_decay','adam_beta1','adam_beta2','adam_epsilon','gradient_checkpointing',
        'gradient_checkpointing_kwargs','bf16','fp16','max_length','max_completion_length']
    for field in fields:
        assert getattr(args[0],field)==getattr(args[1],field),field
    for i in [0,1]:
        actual=read(a.root/'models/checkpoint'/tracks[i]/names[i]/'ABLATION_ACTUAL.json')
        assert actual['actual_component_clip']==(.05 if i==0 else None) and actual['beta']==0
        assert not actual['core_loss_replaced']
    base_queue=read(a.root/'gpu-queue/queues/copsd-bn-base-dev8192-v1/state.json')
    assert base_queue['phase']=='completed' and base_queue['job']['exit_code']==0
    base_job=base_queue['job']
    assert len(base_job['gpu_uuids'])==1 and base_job['config']['max_user_gpus']==4
    base_hours=(datetime.datetime.fromisoformat(base_job['finished_at'])-datetime.datetime.fromisoformat(base_job['started_at'])).total_seconds()/3600
    root=a.root/'logs/copsd_official_10lang/development_vllm8192'
    devnames=['bn_base_dev100_8192_20261020_v1', *[n+'_dev8192' for n in names]]
    configs=[read(root/n/'CONFIG.json') for n in devnames]
    canonical=[{k:v for k,v in c.items() if k not in ['adapter','output','checkpoint_step']} for c in configs]
    assert canonical[0]==canonical[1]==canonical[2]
    assert configs[0]['adapter'] is None and configs[0]['checkpoint_step'] is None
    assert all(c['checkpoint_step']==20 for c in configs[1:])
    for cfg in configs:
        assert cfg['language']=='bn' and cfg['max_new_tokens']==8192 and cfg['max_num_seqs']==16
        assert cfg['protocol']=='project_dev_single_vllm8192_v2'
    assert all(Path(c['adapter']).resolve()==(a.root/'models/checkpoint'/t/n/'checkpoint-20').resolve()
               for c,t,n in zip(configs[1:],tracks,names))
    rows=[[json.loads(l) for l in (root/n/'ITEMS.jsonl').read_text().splitlines()] for n in devnames]
    metrics=[read(root/n/'DONE.json') for n in devnames]
    for rr,m in zip(rows,metrics):
        assert len(rr)==len({r['id'] for r in rr})==m['items']==100
        assert not m['formal_test'] and m['n']==1 and m['actual_max_num_seqs']==16
        assert sum(r['samples'][0]['correct'] for r in rr)==m['single_correct']
        assert sum(len(r['samples'][0]['token_ids']) for r in rr)==m['generated_tokens']
    for triplet in zip(*rows):
        assert all(all(r[k]==triplet[0][k] for k in ['id','question','answer','prompt']) for r in triplet)
        assert len({r['samples'][0]['seed'] for r in triplet})==1
    correct=[[r['samples'][0]['correct'] for r in rr] for rr in rows]
    b,c,u=correct
    out={'observed_at':datetime.datetime.now().isoformat(),'protocol':'project_BN_component_clip_matched20_v1',
        'paired_development_correct_out_of100':dict(zip(['Base','clip0.05_20','unclipped_20'],[sum(x) for x in correct])),
        'unclipped_vs_clipped':{'gained':sum(not x and y for x,y in zip(c,u)),
            'lost':sum(x and not y for x,y in zip(c,u)),'delta_correct':sum(u)-sum(c)},
        'metrics':dict(zip(['Base','clipped','unclipped'],metrics)),
        'training_cost':dict(zip(['clipped','unclipped'],training)),
        'target_evaluation_cost':{'new_responses':300,'new_generated_tokens':sum(m['generated_tokens'] for m in metrics),
            'new_generation_seconds':sum(m['generation_seconds'] for m in metrics),'fresh_Base_allocated_GPU_hours':base_hours,
            'train_target_pipelines_plus_Base_GPU_hours':base_hours+sum(x['pipeline_GPU_hours_including_startup_training_development'] for x in training),
            'English_supplement_not_included':True,
            'accounting_caveat':'Same two frozen training runs as earlier completion reports; do not double-count overlapping records. Base here is newly evaluated at8192, not a cached older2048 run.'},
        'source_script_and_completed_exit0_verified':True,
        'initial_adapter_and_inference_weights_match':True,
        'saved_adapter_configs_loss_temperature_and_initialization_seed_match':True,
        'adapter_serialization_order_differences':adapter_serialization_differences,
        'scientific_training_args_compared':fields,
        'development_inputs_prompts_seeds_configs_match':True,
        'new_core_loss_implementation':False,'clip_only_scientific_factor':True,
        'scope':'One training seed, independent development100, prespecified20-update budget. '
            'Not a stable multilingual effect, formal-test selection or sequential forgetting result. '
            'Online trajectories and generated token costs may differ after policies diverge. Fresh8192 Base costs are included separately; English supplement is not included.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'scores':out['paired_development_correct_out_of100'],
                      'pair':out['unclipped_vs_clipped']}),flush=True)


if __name__=='__main__':main()
