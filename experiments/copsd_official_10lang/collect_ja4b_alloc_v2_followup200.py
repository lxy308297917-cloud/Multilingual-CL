"""Accept fixed JA4B20 target/English development pairs; cached EnglishBase is not regenerated."""
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
    queue = 'copsd-ja4b-followup-ja20-alloc-v2'
    state = read(root / 'gpu-queue/queues' / queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0, queue
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert job['config']['job']['args'] == [case]
    assert job['script_sha256'] == frozen['runtime_code_sha256']['run_ja4b_alloc_v2_followup200.sh'] == sha(Path(job['script']))
    adapter = root / spec['adapter_relative'] if spec['adapter_relative'] else None
    if adapter:
        assert sha(adapter / 'adapter_model.safetensors') == spec['adapter_sha256']
        assert read(adapter / 'trainer_state.json')['global_step'] == 20
    evaluations = {}
    for lang in spec['evaluation_languages']:
        name = f'ja4b_alloc_v2_followup200_{case}_{lang}_20261023'
        folder = root / 'logs/copsd_official_10lang/ja4b_alloc_v2_followup_development200' / name
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
        assert cfg['evaluator_sha256'] == frozen['runtime_code_sha256']['evaluate_ja4b_alloc_v2_followup200.py']
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
    assert model['config']['protocol']=='project_JA4B_alloc_v2_followup_dev200_single8192_c16_v1'
    assert base['config']['protocol'] in ['project_JA4B_followup_dev200_single8192_c16_v1','project_SW4B_followup_dev200_single8192_c16_v1']
    expected='project_JA4B_followup_dev200_single8192_c16_v1' if model['config']['language']=='ja' else 'project_SW4B_followup_dev200_single8192_c16_v1'
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
    freeze=code/'ja4b_alloc_v2_eval_frozen_20261023.PUBLIC.json';f=read(freeze)
    for name,digest in f['runtime_code_sha256'].items():assert sha(code/name)==digest,name
    base={};cacheJA=f['cached_Base_JA'];folderJA=a.root/cacheJA['relative']
    acceptedJApath=code/'ja4b_Base200_completed_20261023.PUBLIC.json';assert sha(acceptedJApath)==cacheJA['accepted_report_sha256']
    acceptedJA=read(acceptedJApath)
    assert sha(folderJA/'ITEMS.jsonl')==cacheJA['ITEMS_sha256']==acceptedJA['source']['items_sha256']
    assert sha(folderJA/'CONFIG.json')==cacheJA['CONFIG_sha256']==acceptedJA['source']['config_sha256']
    base['ja']={'config':read(folderJA/'CONFIG.json'),'rows':read_jsonl(folderJA/'ITEMS.jsonl'),'metrics':acceptedJA['metrics'],'source':acceptedJA['source']}
    stateJA=read(a.root/'gpu-queue/queues'/cacheJA['queue']/'state.json');assert stateJA['phase']=='completed' and stateJA['job']['exit_code']==0
    jobJA=stateJA['job'];baseJA_hours=(datetime.datetime.fromisoformat(jobJA['finished_at'])-datetime.datetime.fromisoformat(jobJA['started_at'])).total_seconds()/3600
    base_cost={'queue':cacheJA['queue'],'parent_GPU_hours_counted_once':baseJA_hours,'evaluation_cases':1,'training_updates':0,'scope':'Priorpartialreportdeferredthisparentcost; countedonce inthiscompletecandidate report.'}
    model,model_cost=audit_case(a.root,code,f,sha(freeze),'JA20')
    cache=f['cached_Base_English'];folder=a.root/cache['relative']
    assert sha(folder/'ITEMS.jsonl')==cache['ITEMS_sha256'] and sha(folder/'CONFIG.json')==cache['CONFIG_sha256']
    accepted=read(code/'sw4b_followup200_completed_20261022.PUBLIC.json')['languages']['en']['conditions']['Base']
    assert accepted['source']['items_sha256']==cache['ITEMS_sha256'] and accepted['source']['config_sha256']==cache['CONFIG_sha256']
    base['en']={'config':read(folder/'CONFIG.json'),'rows':read_jsonl(folder/'ITEMS.jsonl'),'metrics':accepted['metrics'],'source':accepted['source']}
    assert [x['id'] for x in base['en']['rows']]==f['selected_IDs']
    assert sum(x['samples'][0]['correct'] for x in base['en']['rows'])==base['en']['metrics']['single_correct']
    languages={}
    for lang in ['ja','en']:
        delta=paired(base[lang],model[lang]);flags=lambda run:np.array([x['samples'][0]['correct'] for x in run['rows']],dtype=float)
        changes=flags(model[lang])-flags(base[lang]);rng=np.random.default_rng(20261023)
        values=changes[rng.integers(0,200,size=(10000,200))].mean(axis=1)*100
        languages[lang]={'paired':delta,'paired_question_bootstrap95_CI_pp':np.quantile(values,[.025,.975]).tolist(),
            'conditions':{key:{'metrics':run['metrics'],'source':run['source']} for key,run in [('Base',base[lang]),('JA20',model[lang])]},
            'uncertainty_scope':'Questions for onefixedmodelpair, not trainingseedpopulation or multiplicityadjusted inference.'}
    train=read(code/'ja4b_server20_alloc_v2_frozen_20261023.PUBLIC.json');proof=read(a.root/train['checkpoint_relative']/'SERVER_PILOT20_UPDATES_DONE.json')
    state=read(a.root/'gpu-queue/queues/copsd-ja4b-server20-alloc-v2/state.json');assert state['phase']=='completed' and state['job']['exit_code']==0
    job=state['job'];assert len(job['gpu_uuids'])==2
    train_hours=(datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()*2/3600
    cost={'new_training_parent_GPU_hours':train_hours,'new_evaluation_parent_GPU_hours':sum(x['parent_GPU_hours_counted_once'] for x in [base_cost,model_cost]),
        'new_evaluation_generated_tokens':base['ja']['metrics']['generated_tokens']+sum(v['metrics']['generated_tokens'] for v in model.values()),
        'training_exposures':640,'training_generated_tokens':sum(x['generated_tokens'] for x in proof['batch_generation']),
        'training_valid_response_loss_positions':proof['cost']['valid_response_loss_positions'],'teacher_forward_calls':proof['cost']['teacher_forward_calls'],
        'cached_Base_English_GPU_cost_recounted':False}
    failure=read(code/'ja4b_server20_failure_20261023.PUBLIC.json')
    cost['prior_failed_training_parent_GPU_hours']=failure['parent_GPU_hours']
    cost['candidate_total_including_failed_attempt_GPU_hours']=train_hours+cost['new_evaluation_parent_GPU_hours']+failure['parent_GPU_hours']
    cost['reused_Base_JA_generation_tokens_included_once']=base['ja']['metrics']['generated_tokens']
    cost['new_v2_trained_evaluation_tokens']=sum(v['metrics']['generated_tokens'] for v in model.values())
    cost['Base_JA_cost_deferred_in_prior_partial_report_now_counted_once']=True
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb=a.root/'logs/copsd_official_10lang/tensorboard/ja4b_alloc_v2_followup200_20261023';writer=SummaryWriter(str(tb))
    for lang,data in languages.items():
        for condition,run in data['conditions'].items():writer.add_scalar(lang+'/'+condition+'_accuracy',run['metrics']['single_accuracy'],20)
    writer.add_scalar('cost/new_parent_GPU_hours',train_hours+cost['new_evaluation_parent_GPU_hours'],20);writer.add_scalar('cost/new_evaluation_tokens',cost['new_evaluation_generated_tokens'],20);writer.close()
    acc=EventAccumulator(str(tb)).Reload();assert len(acc.Tags()['scalars'])==6
    for lang,data in languages.items():
        for condition,run in data['conditions'].items():assert abs(acc.Scalars(lang+'/'+condition+'_accuracy')[-1].value-run['metrics']['single_accuracy'])<1e-7
    result={'protocol':f['protocol'],'completed_at':datetime.datetime.now().isoformat(),'languages':languages,'cost':cost,'unique_evaluation_parent_costs':[base_cost,model_cost],
        'TensorBoard_four_accuracies_two_costs_read_back':True,'frozen_sha256':sha(freeze),'summarizer_sha256':sha(Path(__file__)),
        'source_metadata_difference':'Cached Japanese/EnglishBase use priorJA4B/SW4B protocol names/outputwrappers, explicitly bound cache digests. Actual inputs/prompts/questionseeds/modelrevision/packages/sampling/length/native16 verified equal; no engine ormethod mixing.',
        'scope':'One allocator-repaired JA4B20 independentdevelopmentcandidate onpre-existing200 withheldfrom2700train, not formaltest, officialpass12, seedconfirmation orselected3languages. Rawnegative results retained. Cached EnglishBase notregenerated orrecounted; cachedJA Basepartialdeferredcost nowincludedonce. Priorfailedtraining cost separatelylisted.'}
    with a.output.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v['paired'] for k,v in languages.items()}))


if __name__=='__main__':main()
