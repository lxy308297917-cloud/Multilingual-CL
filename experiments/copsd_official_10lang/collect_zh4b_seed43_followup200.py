"""Accept fixed ZH4B20 target/English development pairs; cached EnglishBase is not regenerated."""
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
    assert case == 'ZH43'
    queue = 'copsd-zh4b-seed43-eval-v1'
    state = read(root / 'gpu-queue/queues' / queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0, queue
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert job['config']['job']['args'] == [case]
    assert job['script_sha256'] == frozen['runtime_code_sha256']['run_zh4b_seed43_followup200.sh'] == sha(Path(job['script']))
    adapter = root / spec['adapter_relative'] if spec['adapter_relative'] else None
    if adapter:
        assert sha(adapter / 'adapter_model.safetensors') == spec['adapter_sha256']
        assert read(adapter / 'trainer_state.json')['global_step'] == 20
    evaluations = {}
    for lang in spec['evaluation_languages']:
        name = f'zh4b_seed43_followup200_{case}_{lang}_20261023'
        folder = root / 'logs/copsd_official_10lang/zh4b_seed43_followup_development200' / name
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
        assert cfg['evaluator_sha256'] == frozen['runtime_code_sha256']['evaluate_zh4b_seed43_followup200.py']
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
    assert model['config']['protocol']=='project_ZH4B_seed43_followup_dev200_single8192_c16_v1'
    assert base['config']['protocol'] in ['project_ZH4B_followup_dev200_single8192_c16_v1','project_SW4B_followup_dev200_single8192_c16_v1']
    expected='project_ZH4B_followup_dev200_single8192_c16_v1' if model['config']['language']=='zh' else 'project_SW4B_followup_dev200_single8192_c16_v1'
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




def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();code=Path(__file__).parent
    fp=code/'zh4b_seed43_eval_frozen_20261023.PUBLIC.json';f=read(fp)
    for name,digest in f['runtime_code_sha256'].items():assert sha(code/name)==digest,name
    model,model_cost=audit_case(a.root,code,f,sha(fp),'ZH43');base={}
    for lang,key in [('zh','cached_Base_ZH'),('en','cached_Base_English')]:
        spec=f[key];folder=a.root/spec['relative'];assert sha(folder/'ITEMS.jsonl')==spec['ITEMS_sha256'] and sha(folder/'CONFIG.json')==spec['CONFIG_sha256']
        accepted=code/spec['accepted_report_relative'];assert sha(accepted)==spec['accepted_report_sha256'];record=read(accepted)
        accepted_case=record if lang=='zh' else record['languages']['en']['conditions']['Base']
        assert accepted_case['source']['items_sha256']==spec['ITEMS_sha256'] and accepted_case['source']['config_sha256']==spec['CONFIG_sha256']
        base[lang]={'config':read(folder/'CONFIG.json'),'rows':read_jsonl(folder/'ITEMS.jsonl'),'metrics':accepted_case['metrics'],'source':accepted_case['source']}
        assert [x['id'] for x in base[lang]['rows']]==f['selected_IDs'] and len(base[lang]['rows'])==200
        assert sum(x['samples'][0]['correct'] for x in base[lang]['rows'])==base[lang]['metrics']['single_correct']
    languages={}
    for lang in ['zh','en']:
        comparison=paired(base[lang],model[lang]);changes=np.array([int(y['samples'][0]['correct'])-int(x['samples'][0]['correct']) for x,y in zip(base[lang]['rows'],model[lang]['rows'])]);rng=np.random.default_rng(20261023)
        boot=changes[rng.integers(0,200,size=(10000,200))].mean(1)*100
        languages[lang]={'paired':comparison,'paired_question_bootstrap95_CI_pp':np.quantile(boot,[.025,.975]).tolist(),'conditions':{k:{'metrics':v['metrics'],'source':v['source']} for k,v in [('Base',base[lang]),('ZH43',model[lang])]},'uncertainty_scope':'Questions for one fixedmodelpair; not trainingseedpopulation or multiplicityadjusted inference.'}
    train=read(code/'zh4b_seed43_frozen_20261023.PUBLIC.json');state=read(a.root/'gpu-queue/queues/copsd-zh4b-seed43-v1/state.json');assert state['phase']=='completed' and state['job']['exit_code']==0
    job=state['job'];assert len(job['gpu_uuids'])==2 and job['config']['max_user_gpus']==4 and job['script_sha256']==train['code_sha256']['run_zh4b_seed43.sh']
    proof=read(a.root/train['checkpoint_relative']/'SERVER_PILOT20_UPDATES_DONE.json');assert proof['completed_updates']==20 and proof['micro_steps']==640 and proof['base_unchanged_bitwise']
    hours=(datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()*2/3600
    cost={'new_training_parent_GPU_hours':hours,'new_evaluation_parent_GPU_hours':model_cost['parent_GPU_hours_counted_once'],'new_evaluation_generated_tokens':sum(x['metrics']['generated_tokens'] for x in model.values()),'training_exposures':640,'training_generated_tokens':sum(x['generated_tokens'] for x in proof['batch_generation']),'training_valid_response_loss_positions':proof['cost']['valid_response_loss_positions'],'teacher_forward_calls':proof['cost']['teacher_forward_calls'],'cached_Bases_and_seed42_GPU_cost_recounted':False};cost['candidate_total_GPU_hours']=hours+cost['new_evaluation_parent_GPU_hours']
    ref=code/'zh4b_followup200_completed_20261023.PUBLIC.json';assert sha(ref)==f['seed42_reference_report_sha256'];reference=read(ref)
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb=a.root/'logs/copsd_official_10lang/tensorboard/zh4b_seed43_followup200_20261023';expected={}
    for lang,v in languages.items():
        for case,record in v['conditions'].items():expected[lang+'/'+case+'_accuracy']=record['metrics']['single_accuracy']
    expected['cost/new_parent_GPU_hours']=cost['candidate_total_GPU_hours'];expected['cost/new_evaluation_tokens']=cost['new_evaluation_generated_tokens']
    with SummaryWriter(str(tb)) as writer:
        for key,value in expected.items():writer.add_scalar(key,value,20)
    acc=EventAccumulator(str(tb)).Reload();assert all(abs(acc.Scalars(k)[-1].value-v)<max(1e-7,abs(v)*1e-7) for k,v in expected.items())
    report={'completed_at':datetime.datetime.now().isoformat(),'protocol':f['protocol'],'training_seed':43,'data_seed':43,'evaluation_question_seed':42,'languages':languages,'cost':cost,'unique_evaluation_parent_costs':[model_cost],'seed42_reference_correct_of200':{lang:reference['languages'][lang]['conditions']['ZH20']['metrics']['single_correct'] for lang in ['zh','en']},'TensorBoard_four_accuracies_two_costs_read_back':True,'frozen_sha256':sha(fp),'collector_sha256':sha(Path(__file__)),'scope':'OnefreshBase confirmation with training/data seed43, same2700pool/budget, differentpossible640exposures and onlinepolicy trajectories. BothverifiedBases reused, not regenerated/recounted. Native16 samequestionseed42/prompt/source/packages compared. Not formaltest, pureLoRAinitialization intervention, guaranteedvLLMRNGseed43, three-language selection or sequentialtraining.'}
    with a.output.open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({lang:v['paired'] for lang,v in languages.items()}))


if __name__=='__main__':main()
