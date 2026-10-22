"""Audit fresh paired SW learning-rate runs and their fixed originaldev100 outputs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from summarize_saved_candidates8192 import load_eval,pair,read


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--language-report',type=Path,nargs='+');a=p.parse_args();assert not a.output.exists()
    root=a.root;code=Path(__file__).parent;freeze=code/'sw_lr20_frozen_20261022.PUBLIC.json';f=read(freeze)
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    manifest=root/f['training_manifest_relative'];assert sha(manifest)==f['training_manifest_sha256'];m=read(manifest)
    data=root/m['data_relative'];assert sha(data)==m['data_sha256']
    base={'sw':load_eval(root,'copsd-sw-base-dev8192-v1','sw_base_dev100_8192_20261020_v1','sw',None),
          'en':load_eval(root,'copsd-en-sw-base-dev8192-v1','en_sw_base_dev100_8192_20261020_v1','en',None)}
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    evaluations={};training={};orders={};arguments={}
    for condition,spec in f['conditions'].items():
        state=read(root/'gpu-queue/queues'/spec['queue']/'state.json');job=state.get('job') or {}
        assert state['phase']=='completed' and job['exit_code']==0
        assert job['script_sha256']==f['code_sha256']['run_sw_lr20.sh'] and job['config']['job']['args']==[condition]
        output=root/spec['checkpoint_relative'];cp=output/'checkpoint-20'
        assert read(cp/'trainer_state.json')['global_step']==20
        args=torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False)
        assert args.learning_rate==spec['learning_rate'] and args.seed==args.data_seed==42 and args.max_steps==100
        assert torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)['last_epoch']==20
        assert {int(v['step']) for v in torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)['state'].values()}=={20}
        torch.load(cp/'rng_state.pth',map_location='cpu',weights_only=False)
        actual=read(output/'LR_DIAGNOSTIC_ACTUAL.json');assert actual['learning_rate']==args.learning_rate and actual['frozen_sha256']==sha(freeze) and actual['training_data_sha256']==sha(data)
        proof=read(output/'COLLOCATE_TRAIN_UPDATES_DONE.json');assert proof['global_step']==20 and proof['micro_steps']==640 and proof['base_unchanged_bitwise'] and len(proof['syncs'])==21
        assert read(output/'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
        orders[condition]=read(output/'LR_EXPOSURE_PROMPTS.PRIVATE.json');assert len(orders[condition])==len({tuple(sorted(r.items())) for r in orders[condition]})==640
        cost=read(output/'COST_SEGMENT_DONE.json');assert cost['loss_calls']==640
        generation=read(output/'UPDATE_BATCH_GENERATION.json')['groups'];assert len(generation)==20 and sum(r['rollouts'] for r in generation)==640
        for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth']:assert (cp/name).stat().st_size>0
        keys=['max_steps','seed','data_seed','per_device_train_batch_size','gradient_accumulation_steps','optim','lr_scheduler_type','warmup_ratio','warmup_steps','weight_decay','max_grad_norm','adam_beta1','adam_beta2','adam_epsilon','bf16','fp16','gradient_checkpointing']
        arguments[condition]={key:str(getattr(args,key)) for key in keys}
        tb=root/'logs/copsd_official_10lang/tensorboard'/spec['run_id'];acc=EventAccumulator(str(tb));acc.Reload();assert acc.Scalars('train/loss')[-1].step==20
        training[condition]={'learning_rate':args.learning_rate,'initial_sync':proof['syncs'][0],
            'actual640_prompt_sequence_sha256':sha(output/'LR_EXPOSURE_PROMPTS.PRIVATE.json'),'adapter_sha256':sha(cp/'adapter_model.safetensors'),
            'training_generated_tokens':sum(r['generated_tokens'] for r in generation),'cost':cost,'train_seconds_including_diagnostic_hashing':proof['train_seconds_including_diagnostic_hashing'],
            'parent_GPU_hours_counted_once':(datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()/3600,
            'TensorBoard_training_loss_step20_verified':True}
        evaluations[condition]={lang:load_eval(root,spec['queue'],name,lang,cp) for lang,name in spec['evaluation_run_ids'].items()}
    assert arguments['reference']==arguments['lower_lr'] and orders['reference']==orders['lower_lr']
    for key in ['adapter_hash','engine_hash']:assert training['reference']['initial_sync'][key]==training['lower_lr']['initial_sync'][key]
    comparisons={cond:{lang:pair(base[lang],run) for lang,run in rr.items()} for cond,rr in evaluations.items()}
    direct={lang:pair(evaluations['reference'][lang],evaluations['lower_lr'][lang]) for lang in ['sw','en']}
    deltas={lang:direct[lang]['delta_correct'] for lang in ['sw','en']}
    passes=comparisons['lower_lr']['sw']['delta_correct']>0 and min(deltas.values())>=0 and max(deltas.values())>0
    from torch.utils.tensorboard import SummaryWriter
    directory=root/'logs/copsd_official_10lang/tensorboard/sw_lr20_development_20261022_v1';assert not directory.exists()
    with SummaryWriter(str(directory)) as writer:
        for condition,rr in evaluations.items():
            for lang,run in rr.items():writer.add_scalar(condition+'/'+lang+'_accuracy',run[1]['single_correct']/100,20)
            writer.add_scalar(condition+'/unique_parent_GPU_hours',training[condition]['parent_GPU_hours_counted_once'],20)
    verification=EventAccumulator(str(directory));verification.Reload()
    for condition,rr in evaluations.items():
        for lang,run in rr.items():
            event=verification.Scalars(condition+'/'+lang+'_accuracy')[-1]
            assert event.step==20 and abs(event.value-run[1]['single_correct']/100)<1e-6
        event=verification.Scalars(condition+'/unique_parent_GPU_hours')[-1]
        assert event.step==20 and abs(event.value-training[condition]['parent_GPU_hours_counted_once'])<1e-6
    out={'observed_at':datetime.datetime.now().isoformat(),'protocol':f['protocol'],'frozen_sha256':sha(freeze),
        'both_actual_parents_completed_exit0':True,'actual_initial_adapter_engine_and640_ordered_prompt_pairs_equal':True,
        'matched_scientific_arguments_except_learning_rate_verified':True,'training':training,'Base_vs_conditions':comparisons,'reference_vs_lower_lr':direct,
        'metrics':{cond:{lang:v[1] for lang,v in rr.items()} for cond,rr in evaluations.items()},'predeclared_joint_development_screen_passes':passes,
        'cost':{'new_eval_responses':400,'new_eval_generated_tokens':sum(v[1]['generated_tokens'] for rr in evaluations.values() for v in rr.values()),
            'new_training_responses':1280,'training_generated_tokens':sum(v['training_generated_tokens'] for v in training.values()),
            'unique_parent_GPU_hours_including_training_and_both_development_languages_once':sum(v['parent_GPU_hours_counted_once'] for v in training.values()),
            'shared_cached_Base_not_charged_again':True,'scope':'Two unique fresh parents. Counts include training, startup, CPU, saving and both development evals once; do not add per-language parent clocks again.'},
        'TensorBoard_development_relative':str(directory.relative_to(root)),'TensorBoard_all4_eval_accuracy_and2_unique_parent_cost_series_read_back_verified':True,'analysis_source_sha256':sha(Path(__file__)),
        'final_three_selected':False,'scope':f['scope'],'selection_caveat':f['selection_rule']}
    if a.language_report:
        language={}
        for path in a.language_report:
            report=read(path);lang={'__label__swh_Latn':'sw','__label__eng_Latn':'en'}[report['target_label']]
            assert lang not in language and sha(report['model'])==report['model_sha256']
            summaries={}
            for condition,run in [('Base',base[lang]),*[(cond,rr[lang]) for cond,rr in evaluations.items()]]:
                cfg,metrics,rows,_=run;folder=Path(cfg['output']);record=report['runs'][folder.name]
                assert record['items_sha256']==sha(folder/'ITEMS.jsonl')
                assert [r['id'] for r in record['predictions']]==[r['id'] for r in rows]
                assert sum(r['correct'] for r in record['predictions'])==metrics['single_correct']
                summaries[condition]=record['summary']
            language[lang]={'target_label':report['target_label'],'detector_model_sha256':report['model_sha256'],'summaries':summaries}
        assert set(language)=={'sw','en'}
        out['language_supplement']={'languages':language,'scope':'GlotLID after math stripping is not fluency or per-token purity; all100 denominators retained, no filtering or regrading.'}
    with a.output.open('x') as stream:json.dump(out,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'Base_vs_conditions':comparisons,'direct':direct,'screen':passes}))


if __name__=='__main__':main()
