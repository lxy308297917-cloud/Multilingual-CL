"""Accept fixed TE4B20 target/English development pairs; cached EnglishBase is not regenerated."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from evaluate_vllm import question_seed


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def audit_case(root, code, frozen, freeze_sha, case):
    spec = dict(frozen['cases'][case])
    if spec['adapter_relative']:
        release=read(root/frozen['checkpoint_release_relative'])
        assert release['evaluation_freeze_sha256']==freeze_sha and release['training_freeze_sha256']==frozen['training_freeze_sha256']
        spec['adapter_sha256']=release['adapter_sha256']
    assert case=='TE20R'
    queue = 'copsd-te4b-resume5-eval-v1'
    state = read(root / 'gpu-queue/queues' / queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0, queue
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert job['config']['job']['args'] == [case]
    assert job['script_sha256'] == frozen['runtime_code_sha256']['run_te4b_resume5_followup200.sh'] == sha(Path(job['script']))
    adapter = root / spec['adapter_relative'] if spec['adapter_relative'] else None
    if adapter:
        assert sha(adapter / 'adapter_model.safetensors') == spec['adapter_sha256']
        assert read(adapter / 'trainer_state.json')['global_step'] == 20
    evaluations = {}
    for lang in spec['evaluation_languages']:
        name = f'te4b_resume5_followup200_{case}_{lang}_20261024'
        folder = root / 'logs/copsd_official_10lang/te4b_resume5_followup_development200' / name
        cfg, metrics = read(folder / 'CONFIG.json'), read(folder / 'DONE.json')
        assert cfg['protocol'] == metrics['protocol'] == frozen['protocol']
        assert cfg['case'] == case and cfg['language'] == lang
        assert cfg['model_revision'] == frozen['model_revision'] and cfg['cohort'] == 'independent_4B'
        for field in ['items', 'samples', 'max_new_tokens', 'max_num_seqs', 'temperature', 'top_p', 'top_k', 'seed']:
            expected = frozen['items_each'] if field == 'items' else frozen[field]
            assert cfg[field] == expected, (case, lang, field)
        assert cfg['data_freeze_sha256'] == freeze_sha
        assert cfg['input_sha256'] == frozen['inputs'][lang]['sha256'] == sha(Path(cfg['input']))
        assert cfg['official_source_sha256'] == frozen['official_evaluation_source_sha256'] == sha(Path(cfg['source']))
        assert cfg['evaluator_sha256'] == frozen['runtime_code_sha256']['evaluate_te4b_resume5_followup200.py']
        assert Path(cfg['model']).resolve() == (root / 'models/Qwen3-4B').resolve()
        if adapter:
            assert Path(cfg['adapter']).resolve() == adapter.resolve() and cfg['checkpoint_step'] == 20
            assert cfg['adapter_sha256'] == spec['adapter_sha256']
        else:
            assert cfg['adapter'] is cfg['checkpoint_step'] is cfg['adapter_sha256'] is None
        rows = read_jsonl(folder / 'ITEMS.jsonl')
        assert [r['id'] for r in rows] == frozen['selected_IDs']
        assert len(rows) == len({r['id'] for r in rows}) == metrics['items'] == metrics['generation_count'] == 200
        assert not metrics['formal_test'] and metrics['n'] == 1 and metrics['actual_max_num_seqs'] == 16
        inputs = read(Path(cfg['input']))
        for row, original in zip(rows, inputs):
            assert row['id'] == str(original['problem_id']) and row['answer'] == original['Answer']
            assert row['question'] == original['problem' if lang == 'en' else 'problem_' + lang]
            assert len(row['samples']) == 1
            sample = row['samples'][0]
            assert type(sample['correct']) is bool
            assert sample['seed'] == question_seed(42, lang, row['id'])
            assert 0 < len(sample['token_ids']) == sample['tokens'] <= 8192
            assert sample['finish_reason'] in ['stop', 'length']
            if sample['correct']: assert sample['boxed_answer'] is not None
        samples = [r['samples'][0] for r in rows]
        for field, values in {
            'single_correct': [s['correct'] for s in samples],
            'generated_tokens': [len(s['token_ids']) for s in samples],
            'truncated': [s['finish_reason'] == 'length' for s in samples],
            'normal_terminations': [s['finish_reason'] == 'stop' for s in samples],
            'answer_extraction_failures': [s['boxed_answer'] is None for s in samples],
        }.items(): assert sum(values) == metrics[field], (case, lang, field)
        assert metrics['single_accuracy'] == metrics['single_correct'] / 200
        evaluations[lang] = {'config': cfg, 'metrics': metrics, 'rows': rows,
            'source': {'run_id': name, 'items_sha256': sha(folder / 'ITEMS.jsonl'), 'config_sha256': sha(folder / 'CONFIG.json')}}
    hours = (datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds() / 3600
    return evaluations, {'queue': queue, 'parent_GPU_hours_counted_once': hours,
                         'evaluation_cases': len(evaluations), 'training_updates': 0}


def paired(base, model):
    assert model['config']['protocol']=='project_TE4B_resume5_followup_dev200_single8192_c16_v1'
    assert base['config']['protocol'] in ['project_TE4B_followup_dev200_single8192_c16_v1','project_SW4B_followup_dev200_single8192_c16_v1']
    expected='project_TE4B_followup_dev200_single8192_c16_v1' if model['config']['language']=='te' else 'project_SW4B_followup_dev200_single8192_c16_v1'
    assert base['config']['protocol']==expected
    for key in ['language', 'model', 'items', 'samples', 'max_new_tokens',
                'max_num_seqs', 'temperature', 'top_p', 'top_k', 'seed', 'input_sha256',
                'official_source_sha256', 'model_revision', 'packages']:
        assert base['config'][key] == model['config'][key], key
    for x, y in zip(base['rows'], model['rows']):
        assert all(x[k] == y[k] for k in ['id', 'question', 'answer', 'prompt'])
        assert x['samples'][0]['seed'] == y['samples'][0]['seed']
    b, m = [[r['samples'][0]['correct'] for r in run['rows']] for run in [base, model]]
    return {'Base_correct': sum(b), 'checkpoint_correct': sum(m),
            'gained': sum(not x and y for x, y in zip(b, m)),
            'lost': sum(x and not y for x, y in zip(b, m)), 'delta_correct': sum(m) - sum(b)}



def parent_cost(root,queue,GPUs,phase='completed',exit_code=0):
    state=read(root/'gpu-queue/queues'/queue/'state.json');assert state['phase']==phase and state['job']['exit_code']==exit_code,queue;job=state['job'];assert len(job['gpu_uuids'])==GPUs and job['config']['max_user_gpus']==4
    seconds=(datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()
    return {'queue':queue,'phase':phase,'exit_code':exit_code,'parent_GPU_hours_counted_once':seconds*GPUs/3600}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();code=Path(__file__).parent;fp=code/'te4b_resume5_eval_frozen_20261024.PUBLIC.json';f=read(fp)
    for n,h in f['runtime_code_sha256'].items():assert sha(code/n)==h,n
    model,model_cost=audit_case(a.root,code,f,sha(fp),'TE20R');base={}
    for lang,key in [('te','cached_Base_TE'),('en','cached_Base_English')]:
        cache=f[key];folder=a.root/cache['relative'];assert sha(folder/'ITEMS.jsonl')==cache['ITEMS_sha256'] and sha(folder/'CONFIG.json')==cache['CONFIG_sha256']
        if lang=='te':
            accepted_path=code/cache['accepted_report'];assert sha(accepted_path)==cache['accepted_report_sha256'];accepted=read(accepted_path)
        else:accepted=read(code/'sw4b_followup200_completed_20261022.PUBLIC.json')['languages']['en']['conditions']['Base']
        assert accepted['source']['items_sha256']==cache['ITEMS_sha256'] and accepted['source']['config_sha256']==cache['CONFIG_sha256']
        base[lang]={'config':read(folder/'CONFIG.json'),'metrics':accepted['metrics'],'source':accepted['source'],'rows':read_jsonl(folder/'ITEMS.jsonl')}
        assert [x['id'] for x in base[lang]['rows']]==f['selected_IDs'] and len(base[lang]['rows'])==200
        assert sum(x['samples'][0]['correct'] for x in base[lang]['rows'])==base[lang]['metrics']['single_correct']
    languages={}
    for lang in ['te','en']:
        comparison=paired(base[lang],model[lang]);delta=np.array([int(y['samples'][0]['correct'])-int(x['samples'][0]['correct']) for x,y in zip(base[lang]['rows'],model[lang]['rows'])]);rng=np.random.default_rng(20261024);boot=delta[rng.integers(0,200,size=(10000,200))].mean(1)*100
        languages[lang]={'paired':comparison,'paired_question_bootstrap95_CI_pp':np.quantile(boot,[.025,.975]).tolist(),'conditions':{k:{'metrics':v['metrics'],'source':v['source']} for k,v in [('Base',base[lang]),('TE20R',model[lang])]},'uncertainty_scope':'Questions for onefixedtrainedmodel; not seedpopulation, blindtest or multiplicityadjusted inference.'}
    train=read(code/'te4b_resume5_frozen_20261024.PUBLIC.json');proof=read(a.root/train['checkpoint_relative']/'SERVER_PILOT20_UPDATES_DONE.json');assert proof['resumed_from_global_step']==5 and proof['micro_steps']==480 and proof['logical_total_exposures']==640 and proof['base_unchanged_bitwise'];retained=proof['retained_checkpoint5_cost'];assert retained['loss_calls']==retained['teacher_forward_calls']==160
    state=read(a.root/'gpu-queue/queues/copsd-te4b-resume5-v1/state.json');assert state['job']['script_sha256']==train['code_sha256']['run_te4b_resume5.sh']
    recovery=parent_cost(a.root,'copsd-te4b-resume5-v1',2);failed=parent_cost(a.root,'copsd-te4b-server20-v1',2,'failed',1);base_cost=parent_cost(a.root,'copsd-te4b-followup-base-v1',1)
    probes=[parent_cost(a.root,'copsd-te4b-oom-check-v'+str(i),1,'completed' if i==3 else 'failed',0 if i==3 else 1) for i in [1,2,3]]
    previous=read(code/'te4b_server20_frozen_20261024.PUBLIC.json');rollout=read_jsonl(a.root/previous['checkpoint_relative']/'SERVER_ROLLOUTS.PRIVATE.jsonl');assert len(rollout)==256 and [r['global_step'] for r in rollout]==[i//32 for i in range(256)]
    from transformers import AutoTokenizer
    pad=AutoTokenizer.from_pretrained(a.root/'models/Qwen3-4B',local_files_only=True).pad_token_id
    valid=lambda r:sum(t!=pad for t in r['completion_ids']);assert sum(map(valid,rollout[:160]))==retained['valid_response_loss_positions']
    new_tokens=sum(x['generated_tokens'] for x in proof['batch_generation']);assert sum(x['rollouts'] for x in proof['batch_generation'])==480
    parents=[failed,recovery,base_cost,model_cost]+probes
    cost={'unique_parent_records':parents,'candidate_total_GPU_hours_including_failed_work_and_probes':sum(x['parent_GPU_hours_counted_once'] for x in parents),'failed_training_parent_GPU_hours':failed['parent_GPU_hours_counted_once'],'successful_recovery_parent_GPU_hours':recovery['parent_GPU_hours_counted_once'],'new_model_evaluation_parent_GPU_hours':model_cost['parent_GPU_hours_counted_once'],'cached_Base_TE_cost_included_once':True,'cached_Base_EN_cost_recounted':False,'new_model_evaluation_generated_tokens':sum(v['metrics']['generated_tokens'] for v in model.values()),'Base_TE_generated_tokens_previously_recorded':base['te']['metrics']['generated_tokens'],'logical_training_exposures':640,'retained_checkpoint5_exposures':160,'new_recovery_exposures':480,'actual_generated_training_trajectories_including_discarded':736,'actual_generated_training_tokens_including_discarded':sum(len(x['completion_ids']) for x in rollout)+new_tokens,'logical_retained_training_generated_tokens':sum(len(x['completion_ids']) for x in rollout[:160])+new_tokens,'logical_valid_loss_positions':retained['valid_response_loss_positions']+proof['cost']['valid_response_loss_positions'],'actual_completed_finite_loss_calls_including_discarded':228+480,'actual_completed_finite_loss_positions_including_discarded':sum(map(valid,rollout[:228]))+proof['cost']['valid_response_loss_positions'],'logical_teacher_forward_calls':640,'actual_teacher_forward_calls_reconstructed':229+proof['cost']['teacher_forward_calls'],'teacher_counter_scope':'Failedtrace enteredKL on micro229 after228finite completions, so229teacherforwards reconstructed from sourcepath; no lostCUDAseconds reconstructed. Recovery counters measured. Fullfailedgeneration batches counted even when unconsumed. Groupcost includes previouslyrecordedBaseTE once, not new spending on it.'}
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb=a.root/'logs/copsd_official_10lang/tensorboard/te4b_resume5_followup200_20261024';scalars={}
    for lang,data in languages.items():
        for case,run in data['conditions'].items():scalars[lang+'/'+case+'_accuracy']=run['metrics']['single_accuracy']
    scalars['cost/group_parent_GPU_hours']=cost['candidate_total_GPU_hours_including_failed_work_and_probes'];scalars['cost/new_model_evaluation_tokens']=cost['new_model_evaluation_generated_tokens']
    with SummaryWriter(str(tb)) as writer:
        for k,v in scalars.items():writer.add_scalar(k,v,20)
    acc=EventAccumulator(str(tb)).Reload();assert all(abs(acc.Scalars(k)[-1].value-v)<max(1e-7,abs(v)*1e-7) for k,v in scalars.items())
    report={'completed_at':datetime.datetime.now().isoformat(),'protocol':f['protocol'],'languages':languages,'cost':cost,'TensorBoard_four_accuracies_two_costs_read_back':True,'frozen_sha256':sha(fp),'collector_sha256':sha(Path(__file__)),'scope':'SameTE4B candidate completedglobal20 by retaining originalcheckpoint5 then15updates; originalBase teacher staysfrozen. Response-viewstorage compaction preserves loss but engineering change and restartedvLLMRNG are explicit. BothacceptedBases reused. Failed/discardedwork and3GPUrepairprobes included. Not officialpass12, a newseed, selectedlanguage or sequentialexperiment.'}
    with a.output.open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({lang:v['paired'] for lang,v in languages.items()}))


if __name__=='__main__':main()
