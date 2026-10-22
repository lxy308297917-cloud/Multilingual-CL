"""Pair completed same4B Base/SW20 development results and account unique parents.

No new inference and no checkpoint selection. Originaldev100 is exploratory;
passing this screen cannot establish seed stability or select three languages.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import torch


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def terminal(root,name):
    p=root/'gpu-queue/queues'/name/'state.json';s=read(p)
    assert s['phase']==s['job']['phase']=='completed' and s['job']['exit_code']==0,name
    start=datetime.datetime.fromisoformat(s['job']['started_at']);end=datetime.datetime.fromisoformat(s['job']['finished_at'])
    count=len(s['job']['gpu_uuids'])
    return {'queue':name,'started_at':s['job']['started_at'],'finished_at':s['job']['finished_at'],
        'allocated_GPUs':count,'allocated_parent_GPU_hours':(end-start).total_seconds()*count/3600,
        'script_sha256':s['job']['script_sha256'],'state_sha256':sha(p)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    root=a.root;code=Path(__file__).parent;freeze=code/'sw4b_server_pilot20_frozen_20261022.PUBLIC.json';f=read(freeze)
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    parent_names=['copsd-sw4b-server-pilot20-v1','copsd-sw4b-server-base-dev-v1','copsd-sw4b-server-checkpoint-dev-v1']
    parents=[terminal(root,n) for n in parent_names]
    for record,name in zip(parents,['run_sw4b_server_pilot20.sh','run_sw4b_server_base_dev8192.sh','run_sw4b_server_checkpoint_dev8192.sh']):assert record['script_sha256']==f['code_sha256'][name]
    directory=root/f['checkpoint_relative'];proof=read(directory/'SERVER_PILOT20_UPDATES_DONE.json');cp=directory/'checkpoint-20'
    assert proof['completed_updates']==20 and proof['micro_steps']==640 and proof['base_unchanged_bitwise'] and len(proof['syncs'])==21
    assert proof['cost']['loss_calls']==proof['cost']['student_forward_calls']==proof['cost']['teacher_forward_calls']==640
    assert read(directory/'SERVER_PARENT_DONE.json')['two_role_job_completed']
    assert read(cp/'trainer_state.json')['global_step']==20
    args=torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False)
    assert args.max_steps==100 and args.seed==args.data_seed==42 and args.learning_rate==5e-6
    assert torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)['last_epoch']==20
    assert {int(v['step']) for v in torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)['state'].values()}=={20}
    assert (cp/'rng_state.pth').stat().st_size>0
    exposure=read(directory/'EXPOSURE_PROMPTS.PRIVATE.json');assert len(exposure)==640
    prior=read(code/'sw_lr20_frozen_20261022.PUBLIC.json')
    reference=read(root/prior['conditions']['reference']['checkpoint_relative']/'LR_EXPOSURE_PROMPTS.PRIVATE.json')
    matched_inputs=exposure==reference
    cases={};total_tokens=0
    for language in ['sw','en']:
        conditions={}
        for condition,key in [('Base','base_evaluation_run_ids'),('SW20','evaluation_run_ids')]:
            path=root/'logs/copsd_official_10lang/development_vllm8192'/f[key][language]
            config=read(path/'CONFIG.json');done=read(path/'DONE.json')
            rows=[json.loads(line) for line in (path/'ITEMS.jsonl').read_text().splitlines()]
            assert config['protocol']==done['protocol']=='project_dev_single_vllm8192_4b_serverpilot_v1'
            assert config['model_revision']==f['model_revision'] and config['evaluator_sha256']==f['code_sha256']['evaluate_sw4b_server_dev8192.py']
            assert config['official_source_sha256']==f['official_evaluator_sha256']
            assert len(rows)==len({str(r['id']) for r in rows})==done['items']==100
            assert config['samples']==done['n']==1 and config['max_new_tokens']==8192 and config['max_num_seqs']==16
            assert (config['temperature'],config['top_p'],config['top_k'],config['seed'])==(1.,.95,-1,42)
            assert config['checkpoint_step']==(None if condition=='Base' else 20)
            assert config['adapter']==(None if condition=='Base' else str(cp))
            samples=[r['samples'][0] for r in rows];assert all(len(r['samples'])==1 for r in rows)
            correct=sum(bool(s['correct']) for s in samples);tokens=sum(s['tokens'] for s in samples)
            assert correct==done['single_correct'] and tokens==done['generated_tokens']
            assert sum(s['finish_reason']=='stop' for s in samples)==done['normal_terminations']
            assert sum(s['finish_reason']=='length' for s in samples)==done['truncated']
            assert sum(s['boxed_answer'] is None for s in samples)==done['answer_extraction_failures']
            assert all(s['tokens']==len(s['token_ids'])<=8192 for s in samples)
            total_tokens+=tokens
            conditions[condition]={'correct':correct,'items':100,'accuracy':correct/100,
                'normal_terminations':done['normal_terminations'],'truncated':done['truncated'],
                'answer_extraction_failures':done['answer_extraction_failures'],'generated_tokens':tokens,
                'other_terminations':100-done['normal_terminations']-done['truncated'],
                'DONE_sha256':sha(path/'DONE.json'),'ITEMS_sha256':sha(path/'ITEMS.jsonl'),
                '_rows':rows}
        base=conditions['Base']['_rows'];trained=conditions['SW20']['_rows']
        for b,t in zip(base,trained):
            assert (b['id'],b['question'],b['answer'],b['prompt'],b['samples'][0]['seed'])==(t['id'],t['question'],t['answer'],t['prompt'],t['samples'][0]['seed'])
        gains=[str(b['id']) for b,t in zip(base,trained) if not b['samples'][0]['correct'] and t['samples'][0]['correct']]
        losses=[str(b['id']) for b,t in zip(base,trained) if b['samples'][0]['correct'] and not t['samples'][0]['correct']]
        for values in conditions.values():del values['_rows']
        cases[language]={'conditions':conditions,'paired_gains':gains,'paired_losses':losses,
            'delta_correct':len(gains)-len(losses)}
    screen=cases['sw']['delta_correct']>0 and cases['en']['delta_correct']>=0
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb=root/'logs/copsd_official_10lang/tensorboard'/f['run_id']/'paired_originaldev100'
    writer=SummaryWriter(str(tb))
    for lang,values in cases.items():
        for condition,stats in values['conditions'].items():writer.add_scalar(lang+'/'+condition+'_accuracy',stats['accuracy'],20)
    hours=sum(r['allocated_parent_GPU_hours'] for r in parents)
    writer.add_scalar('cost/unique_parents_allocated_GPU_hours',hours,20);writer.add_scalar('cost/new_evaluation_generated_tokens',total_tokens,20);writer.close()
    acc=EventAccumulator(str(tb)).Reload()
    for lang in cases:
        for condition in ['Base','SW20']:assert acc.Scalars(lang+'/'+condition+'_accuracy')[-1].value==torch.tensor(cases[lang]['conditions'][condition]['accuracy'],dtype=torch.float32).item()
    result={'protocol':f['protocol'],'completed_at':datetime.datetime.now().isoformat(),'model_revision':f['model_revision'],
        'summarizer_sha256':sha(Path(__file__)),'pilot_frozen_sha256':sha(freeze),'checkpoint20_adapter_sha256':sha(cp/'adapter_model.safetensors'),
        'training_completed':True,'updates':20,'exposures':640,'training_cost':proof['cost'],'training_generation':proof['batch_generation'],
        'frozen_Base_bitwise_unchanged':True,'actual_inputs_match_fresh1_7B_reference':matched_inputs,
        'matched_input_caveat':'Matching prompts/exposure does not equate modelsize, generation trajectory or numerical backend. Cohorts remain separate.',
        'development':cases,'screen_target_improves_English_not_lower':screen,
        'selection_limit':'Exploratory originaldev100 singletrainingseed only; no independent followup200 or seed confirmation, no3-language selection.',
        'output_language_review_completed':False,'formal_test_rerun':False,'unique_parent_costs':parents,
        'unique_parents_allocated_GPU_hours':hours,'new_evaluation_generated_tokens':total_tokens,
        'TensorBoard_four_accuracies_and_two_costs_read_back':True,
        'next_stage':'If screenpasses, inspect output language/termination and plan independent confirmation; if fails, preserve result and no automatic hyperparameter grid,100updates or ten-language expansion.'}
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
