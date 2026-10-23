"""Accept the BN4B component-clipping intervention against matched native16 references.

CPU only. Existing Base and clipped-model costs are not counted again.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_vllm import question_seed


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def completed_job(root, queue, script_hash, arguments, devices):
    state = read(root/'gpu-queue/queues'/queue/'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0, queue
    job = state['job']
    assert len(job['gpu_uuids']) == devices and job['config']['max_user_gpus'] == 4
    assert job['config']['job']['args'] == arguments
    assert job['script_sha256'] == script_hash == sha(Path(job['script']))
    hours = (datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()*devices/3600
    return {'queue': queue, 'parent_GPU_hours': hours}


def load_run(folder, frozen, lang):
    config, metrics = read(folder/'CONFIG.json'), read(folder/'DONE.json')
    items = rows(folder/'ITEMS.jsonl')
    assert len(items) == metrics['items'] == metrics['generation_count'] == 200
    assert [x['id'] for x in items] == frozen['selected_IDs']
    assert config['language'] == lang and config['model_revision'] == frozen['model_revision']
    assert not metrics['formal_test'] and metrics['n'] == 1 and metrics['actual_max_num_seqs'] == 16
    for key in ['samples', 'max_new_tokens', 'max_num_seqs', 'temperature', 'top_p', 'top_k', 'seed']:
        assert config[key] == frozen[key], key
    assert config['input_sha256'] == frozen['inputs'][lang]['sha256']
    assert config['official_source_sha256'] == frozen['official_evaluation_source_sha256']
    for row in items:
        assert len(row['samples']) == 1
        sample = row['samples'][0]
        assert type(sample['correct']) is bool
        assert sample['seed'] == question_seed(42, lang, row['id'])
        assert 0 < sample['tokens'] == len(sample['token_ids']) <= 8192
        assert sample['finish_reason'] in ['stop', 'length']
    samples = [r['samples'][0] for r in items]
    totals = {'single_correct': sum(s['correct'] for s in samples),
              'generated_tokens': sum(s['tokens'] for s in samples),
              'truncated': sum(s['finish_reason'] == 'length' for s in samples),
              'normal_terminations': sum(s['finish_reason'] == 'stop' for s in samples),
              'answer_extraction_failures': sum(s['boxed_answer'] is None for s in samples)}
    assert all(metrics[k] == v for k,v in totals.items())
    assert metrics['single_accuracy'] == totals['single_correct']/200
    return {'config': config, 'rows': items, 'metrics': metrics,
            'source': {'items_sha256': sha(folder/'ITEMS.jsonl'), 'config_sha256': sha(folder/'CONFIG.json')}}


def compare(reference, intervention):
    for key in ['language', 'model', 'model_revision', 'packages', 'samples', 'max_new_tokens', 'max_num_seqs', 'temperature', 'top_p', 'top_k', 'seed', 'input_sha256', 'official_source_sha256']:
        assert reference['config'][key] == intervention['config'][key], key
    for x,y in zip(reference['rows'], intervention['rows'], strict=True):
        assert all(x[k] == y[k] for k in ['id', 'question', 'answer', 'prompt'])
        assert x['samples'][0]['seed'] == y['samples'][0]['seed']
    b,m = [np.array([x['samples'][0]['correct'] for x in run['rows']], dtype=int) for run in [reference, intervention]]
    differences = m-b
    rng = np.random.default_rng(20261023)
    bootstrap = differences[rng.integers(0,200,size=(10000,200))].mean(axis=1)*100
    return {'reference_correct': int(b.sum()), 'intervention_correct': int(m.sum()),
            'gained': int(((b == 0)&(m == 1)).sum()), 'lost': int(((b == 1)&(m == 0)).sum()),
            'delta_correct': int(differences.sum()),
            'paired_question_bootstrap95_CI_pp': np.quantile(bootstrap,[.025,.975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, code = args.root, Path(__file__).parent
    fp = code/'bn4b_unclip20_eval_frozen_20261023.PUBLIC.json'
    frozen = read(fp)
    for name,digest in frozen['runtime_code_sha256'].items():
        assert sha(code/name) == digest, name
    train_path = code/'bn4b_unclip20_frozen_20261023.PUBLIC.json'
    train = read(train_path)
    assert sha(train_path) == frozen['training_freeze_sha256']
    for name,digest in train['code_sha256'].items():
        assert sha(code/name) == digest, name
    release_path = root/frozen['checkpoint_release_relative']
    release = read(release_path)
    assert release['training_freeze_sha256'] == sha(train_path) and release['evaluation_freeze_sha256'] == sha(fp)
    assert release['actual_clip_None_asserted_in_frozen_runtime'] and release['exposure_order_matches_clipped_reference']
    adapter = root/frozen['cases']['Unclip20']['adapter_relative']
    assert release['global_step'] == 20 and sha(adapter/'adapter_model.safetensors') == release['adapter_sha256']
    costs = [completed_job(root, 'copsd-bn4b-unclip20-v1', train['code_sha256']['run_bn4b_unclip20.sh'], [], 2),
             completed_job(root, 'copsd-bn4b-unclip20-eval-v1', frozen['runtime_code_sha256']['run_bn4b_unclip20_followup200.sh'], ['Unclip20'], 1)]
    accepted = read(code/'bn4b_followup200_completed_20261023.PUBLIC.json')
    languages = {}
    for lang in ['bn','en']:
        base_spec = frozen['cached_Base_BN' if lang == 'bn' else 'cached_Base_English']
        base_folder = root/base_spec['relative']
        assert sha(base_folder/'ITEMS.jsonl') == base_spec['ITEMS_sha256'] and sha(base_folder/'CONFIG.json') == base_spec['CONFIG_sha256']
        base = load_run(base_folder, frozen, lang)
        clipped_folder = root/'logs/copsd_official_10lang/bn4b_followup_development200'/f'bn4b_followup200_BN20_{lang}_20261023'
        clipped = load_run(clipped_folder, frozen, lang)
        prior = accepted['languages'][lang]['conditions']['BN20']
        assert clipped['source']['items_sha256'] == prior['source']['items_sha256'] and clipped['metrics'] == prior['metrics']
        folder = root/'logs/copsd_official_10lang/bn4b_unclip20_followup_development200'/f'bn4b_unclip20_followup200_Unclip20_{lang}_20261023'
        unclip = load_run(folder, frozen, lang)
        cfg = unclip['config']
        assert cfg['protocol'] == unclip['metrics']['protocol'] == frozen['protocol']
        assert cfg['data_freeze_sha256'] == sha(fp) and cfg['checkpoint_release_sha256'] == sha(release_path)
        assert cfg['evaluator_sha256'] == frozen['runtime_code_sha256']['evaluate_bn4b_unclip20_followup200.py']
        assert cfg['adapter_sha256'] == release['adapter_sha256'] and cfg['checkpoint_step'] == 20
        assert Path(cfg['adapter']).resolve() == adapter.resolve()
        languages[lang] = {'conditions': {name: {'metrics': run['metrics'], 'source': run['source']} for name,run in [('Base',base),('Clip20',clipped),('Unclip20',unclip)]},
                           'Unclip20_vs_Base': compare(base,unclip), 'Unclip20_vs_Clip20': compare(clipped,unclip)}
    proof = read(root/train['checkpoint_relative']/'SERVER_PILOT20_UPDATES_DONE.json')
    assert proof['micro_steps'] == 640 and proof['completed_updates'] == 20 and proof['base_unchanged_bitwise']
    report = {'completed_at': datetime.datetime.now().isoformat(), 'protocol': frozen['protocol'], 'languages': languages,
              'new_parent_costs': costs, 'new_parent_GPU_hours': sum(x['parent_GPU_hours'] for x in costs),
              'training_exposures': 640, 'training_valid_response_loss_tokens': proof['cost']['valid_response_loss_positions'],
              'new_evaluation_generated_tokens': sum(x['conditions']['Unclip20']['metrics']['generated_tokens'] for x in languages.values()),
              'cached_Base_and_clipped_cost_recounted': False, 'collector_sha256': sha(Path(__file__)),
              'scope': 'One-factor component clipping intervention from fresh4BBase with fixedseed42, exactmatched640inputexposureorder; online student trajectories differ. Native16 paired development only, not mergedbackend, formaltest, seedpopulation confirmation or3-language selection. QuestionCIs unadjusted for multiple comparisons. Original official scores retained.'}
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb = root/'logs/copsd_official_10lang/tensorboard/bn4b_clip_unclip20_comparison_20261023'
    writer = SummaryWriter(str(tb))
    expected = {}
    for lang,data in languages.items():
        for name,run in data['conditions'].items():
            expected[lang+'/'+name+'_accuracy'] = run['metrics']['single_accuracy']
    expected['cost/new_parent_GPU_hours'] = report['new_parent_GPU_hours']
    expected['cost/new_evaluation_tokens'] = report['new_evaluation_generated_tokens']
    for key,value in expected.items(): writer.add_scalar(key,value,20)
    writer.close()
    accumulator = EventAccumulator(str(tb)).Reload()
    assert all(abs(accumulator.Scalars(k)[-1].value-v) < max(1e-7,abs(v)*1e-7) for k,v in expected.items())
    report['TensorBoard_six_accuracies_two_costs_read_back'] = True
    with args.output.open('x') as stream:
        stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({lang: data['Unclip20_vs_Clip20'] for lang,data in languages.items()}))


if __name__ == '__main__':
    main()
