"""Audit the two fixed SW4B followup200 parents; report one-model question uncertainty, not seed stability."""
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
    spec = frozen['cases'][case]
    queue = 'copsd-sw4b-followup-' + case.lower() + '-v1'
    state = read(root / 'gpu-queue/queues' / queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0, queue
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert job['config']['job']['args'] == [case]
    assert job['script_sha256'] == frozen['runtime_code_sha256']['run_sw4b_followup200.sh'] == sha(Path(job['script']))
    adapter = root / spec['adapter_relative'] if spec['adapter_relative'] else None
    if adapter:
        assert sha(adapter / 'adapter_model.safetensors') == spec['adapter_sha256']
        assert read(adapter / 'trainer_state.json')['global_step'] == 20
    evaluations = {}
    for lang in spec['evaluation_languages']:
        name = f'sw4b_followup200_{case}_{lang}_20261022_v1'
        folder = root / 'logs/copsd_official_10lang/sw4b_followup_development200_v1' / name
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
        assert cfg['evaluator_sha256'] == frozen['runtime_code_sha256']['evaluate_sw4b_followup200.py']
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
                'official_source_sha256', 'data_freeze_sha256', 'evaluator_sha256', 'packages']:
        assert base['config'][key] == model['config'][key], key
    for x, y in zip(base['rows'], model['rows']):
        assert all(x[k] == y[k] for k in ['id', 'question', 'answer', 'prompt'])
        assert x['samples'][0]['seed'] == y['samples'][0]['seed']
    b, m = [[r['samples'][0]['correct'] for r in run['rows']] for run in [base, model]]
    return {'Base_correct': sum(b), 'checkpoint_correct': sum(m),
            'gained': sum(not x and y for x, y in zip(b, m)),
            'lost': sum(x and not y for x, y in zip(b, m)), 'delta_correct': sum(m) - sum(b)}



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    code = Path(__file__).parent
    freeze_path = code/'sw4b_followup200_frozen_20261022.PUBLIC.json'
    frozen = read(freeze_path)
    for name, digest in frozen['runtime_code_sha256'].items(): assert sha(code/name) == digest, name
    base, base_cost = audit_case(a.root, code, frozen, sha(freeze_path), 'Base')
    trained, trained_cost = audit_case(a.root, code, frozen, sha(freeze_path), 'SW20')
    languages = {}
    for language in ['sw', 'en']:
        comparison = paired(base[language], trained[language])
        delta = np.array([int(y['samples'][0]['correct'])-int(x['samples'][0]['correct'])
                          for x,y in zip(base[language]['rows'], trained[language]['rows'])])
        rng = np.random.default_rng(20261022)
        samples = delta[rng.integers(0, 200, size=(10000,200))].mean(axis=1)*100
        comparison['paired_question_bootstrap95_CI_pp'] = np.quantile(samples, [.025,.975]).tolist()
        comparison['bootstrap_seed'] = 20261022
        comparison['bootstrap_replicates'] = 10000
        comparison['uncertainty_scope'] = 'Questions for this one fixed model pair only, not trainingseed population or multiplicityadjusted inference.'
        languages[language] = {'paired': comparison, 'conditions': {}}
        for condition, data in [('Base',base),('SW20',trained)]:
            run = data[language]
            languages[language]['conditions'][condition] = {'metrics':run['metrics'],'source':run['source']}
    costs = [base_cost, trained_cost]
    hours = sum(c['parent_GPU_hours_counted_once'] for c in costs)
    tokens = sum(run['metrics']['generated_tokens'] for case in [base,trained] for run in case.values())
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb = a.root/'logs/copsd_official_10lang/tensorboard/sw4b_followup200_20261022'
    writer = SummaryWriter(str(tb))
    for language in languages:
        for condition in ['Base','SW20']:
            writer.add_scalar(language+'/'+condition+'_accuracy',languages[language]['conditions'][condition]['metrics']['single_accuracy'],20)
    writer.add_scalar('cost/unique_parent_GPU_hours',hours,20)
    writer.add_scalar('cost/generated_tokens',tokens,20)
    writer.close()
    acc = EventAccumulator(str(tb)).Reload()
    for language in languages:
        for condition in ['Base','SW20']:
            assert abs(acc.Scalars(language+'/'+condition+'_accuracy')[-1].value-languages[language]['conditions'][condition]['metrics']['single_accuracy']) < 1e-7
    result = {'protocol':frozen['protocol'],'completed_at':datetime.datetime.now().isoformat(),
              'model_revision':frozen['model_revision'],'data_freeze_sha256':sha(freeze_path),
              'languages':languages,'unique_parent_costs':costs,'parent_GPU_hours':hours,
              'new_generated_tokens':tokens,'TensorBoard_four_accuracies_two_costs_read_back':True,
              'scope':'Fixed200 excluded from4B2700 training, used previously for1.7B analysis; not newlyblind or formaltests. One4Btrainingseed, no3-language selection. Same native languageprompts forBase/SW20, English-prefix controls remainoriginal100 only.',
              'summarizer_sha256':sha(Path(__file__))}
    with a.output.open('x') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v['paired'] for k,v in languages.items()},ensure_ascii=False))


if __name__ == '__main__':main()
