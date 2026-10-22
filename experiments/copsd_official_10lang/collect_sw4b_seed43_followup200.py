"""Accept one seed43 candidate against cachedBase/seed42 fixed200, without recounting old costs."""
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


def audit_case(root, code, frozen, freeze_sha, case):
    spec = dict(frozen['cases'][case])
    release=read(root/frozen['checkpoint_release_relative'])
    assert release['evaluation_freeze_sha256']==freeze_sha and release['training_freeze_sha256']==frozen['training_freeze_sha256']
    spec['adapter_sha256']=release['adapter_sha256']
    queue = 'copsd-sw4b-followup-' + case.lower() + '-v1'
    state = read(root / 'gpu-queue/queues' / queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0, queue
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert job['config']['job']['args'] == []
    assert job['script_sha256'] == frozen['runtime_code_sha256']['run_sw4b_seed43_followup200.sh'] == sha(Path(job['script']))
    adapter = root / spec['adapter_relative'] if spec['adapter_relative'] else None
    if adapter:
        assert sha(adapter / 'adapter_model.safetensors') == spec['adapter_sha256']
        assert read(adapter / 'trainer_state.json')['global_step'] == 20
    evaluations = {}
    for lang in spec['evaluation_languages']:
        name = f'sw4b_followup200_{case}_{lang}_20261022_v1'
        folder = root / 'logs/copsd_official_10lang/sw4b_seed43_followup_development200_v1' / name
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
        assert cfg['evaluator_sha256'] == frozen['runtime_code_sha256']['evaluate_sw4b_seed43_followup200.py']
        assert Path(cfg['model']).resolve() == (root / 'models/Qwen3-4B').resolve()
        if adapter:
            assert Path(cfg['adapter']).resolve() == adapter.resolve() and cfg['checkpoint_step'] == 20
            assert cfg['adapter_sha256'] == spec['adapter_sha256']
        else:
            assert cfg['adapter'] is cfg['checkpoint_step'] is cfg['adapter_sha256'] is None
        rows = [json.loads(s) for s in (folder / 'ITEMS.jsonl').read_text().splitlines()]
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
    for key in ['protocol', 'language', 'model', 'items', 'samples', 'max_new_tokens',
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
    freeze_path=code/'sw4b_seed43_eval_frozen_20261022.PUBLIC.json';f=read(freeze_path)
    for name,digest in f['runtime_code_sha256'].items():assert sha(code/name)==digest,name
    model,eval_cost=audit_case(a.root,code,f,sha(freeze_path),'SW43')
    accepted=read(code/'sw4b_followup200_completed_20261022.PUBLIC.json')
    cache={};seed42={}
    for lang in ['sw','en']:
        cache[lang]={}
        for condition,target in [('Base',cache),('SW20',seed42)]:
            saved=accepted['languages'][lang]['conditions'][condition]
            folder=a.root/'logs/copsd_official_10lang/sw4b_followup_development200_v1'/saved['source']['run_id']
            assert sha(folder/'ITEMS.jsonl')==saved['source']['items_sha256'] and sha(folder/'CONFIG.json')==saved['source']['config_sha256']
            target[lang]={'config':read(folder/'CONFIG.json'),'rows':[json.loads(x) for x in (folder/'ITEMS.jsonl').read_text().splitlines()],'metrics':saved['metrics']}
        assert f['Base_cache'][lang]==accepted['languages'][lang]['conditions']['Base']['source']
    languages={}
    for lang in ['sw','en']:
        delta43=paired(cache[lang],model[lang]);delta42=paired(cache[lang],seed42[lang])
        flags=lambda x:np.array([r['samples'][0]['correct'] for r in x['rows']],dtype=float)
        delta=(flags(model[lang])+flags(seed42[lang]))/2-flags(cache[lang])
        rng=np.random.default_rng(20261022);values=delta[rng.integers(0,200,size=(10000,200))].mean(axis=1)*100
        languages[lang]={'Base_correct':int(flags(cache[lang]).sum()),'seed42_correct':int(flags(seed42[lang]).sum()),'seed43_correct':int(flags(model[lang]).sum()),
            'seed42_pair':delta42,'seed43_pair':delta43,'fixed_two_model_mean_delta_pp':float(delta.mean()*100),
            'paired_question_bootstrap95_CI_mean_delta_pp':np.quantile(values,[.025,.975]).tolist(),
            'uncertainty_scope':'Questions for twofixedmodels, not trainingseedpopulation or multiplicityadjusted inference.',
            'seed43_metrics':model[lang]['metrics'],'seed43_source':model[lang]['source']}
    train=read(code/'sw4b_server_seed43_frozen_20261022.PUBLIC.json');folder=a.root/train['checkpoint_relative'];proof=read(folder/'SERVER_PILOT20_UPDATES_DONE.json')
    state=read(a.root/'gpu-queue/queues/copsd-sw4b-server-seed43-v1/state.json');assert state['phase']=='completed' and state['job']['exit_code']==0
    job=state['job'];train_hours=(datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()*2/3600
    cost={'new_training_parent_GPU_hours':train_hours,'new_evaluation_parent_GPU_hours':eval_cost['parent_GPU_hours_counted_once'],
        'new_evaluation_generated_tokens':sum(v['metrics']['generated_tokens'] for v in model.values()),
        'training_exposures':640,'training_generated_tokens':sum(v['generated_tokens'] for v in proof['batch_generation']),
        'training_valid_response_loss_positions':proof['cost']['valid_response_loss_positions'],
        'teacher_forward_calls':proof['cost']['teacher_forward_calls'],
        'cached_Base_and_seed42_GPU_cost_recounted':False}
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb=a.root/'logs/copsd_official_10lang/tensorboard/sw4b_seed43_confirmation_20261022';writer=SummaryWriter(str(tb))
    for lang,v in languages.items():
        for c in ['Base','seed42','seed43']:writer.add_scalar(lang+'/'+c+'_accuracy',v[c+'_correct']/200,20)
    writer.add_scalar('cost/new_parent_GPU_hours',train_hours+eval_cost['parent_GPU_hours_counted_once'],20);writer.close()
    acc=EventAccumulator(str(tb)).Reload();assert len(acc.Tags()['scalars'])==7
    for lang,v in languages.items():
        for c in ['Base','seed42','seed43']:assert abs(acc.Scalars(lang+'/'+c+'_accuracy')[-1].value-v[c+'_correct']/200)<1e-7
    result={'protocol':f['protocol'],'completed_at':datetime.datetime.now().isoformat(),'languages':languages,'cost':cost,
        'evaluation_entry_sha256':f['runtime_code_sha256']['evaluate_sw4b_seed43_followup200.py'],
        'cached_entry_sha256':read(code/'sw4b_followup200_frozen_20261022.PUBLIC.json')['runtime_code_sha256']['evaluate_sw4b_followup200.py'],
        'source_metadata_difference':'Releaseguard, frozenmanifest/outputroot andrecording metadata are declared adaptations; matchedactualprompts/input/questionseeds/parameters/packages are verified. No silent method/backend mixing.',
        'TensorBoard_six_accuracies_one_cost_read_back':True,'scope':'One seed43 confirmation againstfixedBase/seed42; not officialpass12, a newlyblindtest, three-language selection or formaltestdriven checkpointselection. Negative effects retained.',
        'summarizer_sha256':sha(Path(__file__))}
    with a.output.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ['Base_correct','seed42_correct','seed43_correct']} for k,v in languages.items()}))


if __name__=='__main__':main()
