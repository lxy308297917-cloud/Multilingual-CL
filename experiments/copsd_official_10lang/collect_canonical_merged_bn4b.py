"""Accept the separate canonical merged Base/BN20 development evaluation.

CPU postprocessing only. Never launches, stops, or retries a GPU job.
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


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def audit(root, code, frozen, freeze_sha, case):
    spec = frozen['cases'][case]
    queue = 'copsd-canon-bn4b-' + ('base' if case == 'Base' else 'bn20') + '-v1'
    state = read(root / 'gpu-queue/queues' / queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0, queue
    job = state['job']
    assert len(job['gpu_uuids']) == 1 and job['config']['max_user_gpus'] == 4
    assert job['config']['job']['args'] == [case]
    assert job['script_sha256'] == frozen['runtime_code_sha256']['run_canonical_merged_bn4b_eval.sh'] == sha(Path(job['script']))
    folder = root / spec['output_relative']
    cfg, done = read(folder / 'CONFIG.json'), read(folder / 'DONE.json')
    assert cfg['case'] == done['case'] == case
    assert cfg['protocol'] == done['protocol'] == frozen['protocol']
    assert cfg['freeze_sha256'] == freeze_sha
    assert cfg['model_revision'] == frozen['model_revision']
    assert Path(cfg['model']).resolve() == (root / 'models/Qwen3-4B').resolve()
    assert cfg['evaluator_sha256'] == frozen['runtime_code_sha256']['evaluate_canonical_merged_bn4b.py']
    assert cfg['official_source_sha256'] == frozen['official_source_sha256']
    assert cfg['source_inputs'] == frozen['inputs']
    for key in ['samples', 'max_new_tokens', 'max_num_seqs', 'temperature', 'top_p', 'top_k']:
        assert cfg[key] == frozen[key], key
    assert cfg['seed'] == frozen['question_seed'] == 42
    assert not cfg['formal_test'] and not done['formal_test'] and done['no_new_training']
    assert not cfg['old_native16_scores_or_release_modified']
    assert cfg['items'] == done['items'] == 400
    assert cfg['adapter_sha256'] == spec['adapter_sha256']
    if spec['adapter_relative']:
        adapter = root / spec['adapter_relative']
        assert Path(cfg['adapter']).resolve() == adapter.resolve()
        assert sha(adapter / 'adapter_model.safetensors') == spec['adapter_sha256']
        release_path = root / spec['training_release_relative']
        assert sha(release_path) == spec['training_release_sha256']
        assert read(release_path)['adapter_sha256'] == spec['adapter_sha256']
    else:
        assert cfg['adapter'] is None
    init = cfg['initialization']
    assert init['all398_reference_weights_match'] and init['all_weights_loaded'] == 398
    assert init['adapted_weights_loaded'] == (252 if case == 'BN20' else 0)
    assert init['LoRA_tensors'] == (504 if case == 'BN20' else 0)
    assert not init['merge_TF32_enabled'] and init['previous_TF32_restored']
    assert cfg['canonical_reference_sha256'] == spec['canonical_reference_sha256']
    canonical = root / spec['canonical_reference_relative']
    assert sha(canonical) == spec['canonical_reference_sha256']
    assert len(rows(canonical)) == spec['canonical_reference_rows']
    cache_path = folder / 'GENERATED.PRIVATE.jsonl'
    generation = read(folder / 'GENERATION_DONE.PRIVATE.json')
    assert generation['sha256'] == sha(cache_path) and generation['freeze_sha256'] == freeze_sha
    assert generation['items'] == 400 and generation['generation_seconds'] == done['generation_seconds']
    cached = rows(cache_path)
    assert [(x['language'], x['id']) for x in cached] == [(lang, i) for lang in frozen['languages'] for i in frozen['selected_IDs']]
    receipt = root / 'logs/copsd_official_10lang' / (spec['run_id'] + '.question_seeds.PRIVATE.jsonl')
    assert generation['seed_receipt_sha256'] == done['seed_receipt_sha256'] == sha(receipt)
    receipts = rows(receipt)
    assert len(receipts) == 1 and receipts[0]['per_question_params_reached_LLM_generate']
    metadata = receipts[0]['records']
    assert len(metadata) == 400
    for row, meta in zip(cached, metadata, strict=True):
        assert meta['prompt_sha256'] == hashlib.sha256(row['prompt'].encode()).hexdigest()
        assert meta['seed'] == row['seed'] == question_seed(42, row['language'], row['id'])
        assert meta['token_ids_sha256'] == hashlib.sha256(json.dumps(row['token_ids']).encode()).hexdigest()
        assert meta['generated_tokens'] == len(row['token_ids'])
        assert meta['finish_reason'] == row['finish_reason']
    result = {}
    for index, lang in enumerate(frozen['languages']):
        path = folder / lang / 'ITEMS.jsonl'
        items, metrics = rows(path), read(folder / lang / 'DONE.json')
        assert metrics == done['languages'][lang] and metrics['ITEMS_sha256'] == sha(path)
        assert metrics['protocol'] == frozen['protocol'] and metrics['n'] == 1 and not metrics['formal_test']
        assert len(items) == metrics['items'] == 200
        input_path = root / frozen['inputs'][lang]['relative']
        assert sha(input_path) == frozen['inputs'][lang]['sha256']
        for row, source, raw in zip(items, read(input_path), cached[index*200:(index+1)*200], strict=True):
            assert row['id'] == str(source['problem_id']) and row['language'] == lang
            assert row['answer'] == source['Answer']
            assert row['question'] == source['problem' if lang == 'en' else 'problem_bn']
            assert all(row[k] == raw[k] for k in ['id', 'language', 'answer', 'question', 'prompt'])
            assert len(row['samples']) == 1
            sample = row['samples'][0]
            assert all(sample[k] == raw[k] for k in ['seed', 'token_ids', 'finish_reason', 'raw_output'])
            assert type(sample['correct']) is bool
            assert 0 < sample['tokens'] == len(sample['token_ids']) <= 8192
            assert sample['finish_reason'] in ['stop', 'length']
        samples = [r['samples'][0] for r in items]
        totals = {'single_correct': sum(s['correct'] for s in samples),
                  'generated_tokens': sum(s['tokens'] for s in samples),
                  'truncated': sum(s['finish_reason'] == 'length' for s in samples),
                  'normal_terminations': sum(s['finish_reason'] == 'stop' for s in samples),
                  'answer_extraction_failures': sum(s['boxed_answer'] is None for s in samples)}
        assert all(metrics[k] == v for k, v in totals.items())
        assert metrics['single_accuracy'] == totals['single_correct'] / 200
        result[lang] = {'rows': items, 'metrics': metrics}
    assert done['generated_tokens'] == sum(x['metrics']['generated_tokens'] for x in result.values())
    hours = (datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds()/3600
    return result, cfg, {'case': case, 'queue': queue, 'parent_GPU_hours': hours,
                        'generated_tokens': done['generated_tokens'], 'generation_seconds': done['generation_seconds']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).parent
    fp = code / 'canonical_merged_bn4b_eval_frozen_20261023.PUBLIC.json'
    frozen = read(fp)
    for name, digest in frozen['runtime_code_sha256'].items():
        assert sha(code / name) == digest, name
    cases = {case: audit(args.root, code, frozen, sha(fp), case) for case in ['Base', 'BN20']}
    base, model = cases['Base'], cases['BN20']
    for key in ['protocol', 'model', 'model_revision', 'packages', 'source_inputs', 'samples', 'max_new_tokens', 'max_num_seqs', 'temperature', 'top_p', 'top_k', 'seed', 'official_source_sha256']:
        assert base[1][key] == model[1][key], key
    languages = {}
    for lang in frozen['languages']:
        b, m = base[0][lang], model[0][lang]
        for x, y in zip(b['rows'], m['rows'], strict=True):
            assert all(x[k] == y[k] for k in ['id', 'language', 'question', 'answer', 'prompt'])
            assert x['samples'][0]['seed'] == y['samples'][0]['seed']
        bv, mv = [np.array([r['samples'][0]['correct'] for r in run['rows']], dtype=int) for run in [b, m]]
        delta = mv-bv
        rng = np.random.default_rng(20261023)
        bootstrap = delta[rng.integers(0, 200, size=(10000, 200))].mean(axis=1)*100
        languages[lang] = {'Base': b['metrics'], 'BN20': m['metrics'],
                           'paired': {'gained': int(((bv == 0) & (mv == 1)).sum()), 'lost': int(((bv == 1) & (mv == 0)).sum()), 'delta_correct': int(delta.sum())},
                           'paired_question_bootstrap95_CI_pp': np.quantile(bootstrap, [.025, .975]).tolist()}
    costs = [value[2] for value in cases.values()]
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb = args.root / 'logs/copsd_official_10lang/tensorboard/canonical_merged_bn4b_paired_20261023'
    writer = SummaryWriter(str(tb))
    expected = {}
    for lang, data in languages.items():
        for case in ['Base', 'BN20']:
            expected[lang+'/'+case+'_accuracy'] = data[case]['single_accuracy']
    expected['cost/new_evaluation_parent_GPU_hours'] = sum(x['parent_GPU_hours'] for x in costs)
    expected['cost/new_evaluation_tokens'] = sum(x['generated_tokens'] for x in costs)
    for key, value in expected.items(): writer.add_scalar(key, value, 20)
    writer.close()
    accumulator = EventAccumulator(str(tb)).Reload()
    assert all(abs(accumulator.Scalars(k)[-1].value-v) < max(1e-7, abs(v)*1e-7) for k,v in expected.items())
    report = {'completed_at': datetime.datetime.now().isoformat(), 'protocol': frozen['protocol'],
              'languages': languages, 'unique_parent_costs': costs,
              'new_evaluation_parent_GPU_hours': expected['cost/new_evaluation_parent_GPU_hours'],
              'new_evaluation_tokens': expected['cost/new_evaluation_tokens'],
              'TensorBoard_four_accuracies_two_costs_read_back': True, 'freeze_sha256': sha(fp),
              'collector_sha256': sha(Path(__file__)),
              'scope': 'New merged-serving paired development protocol only. Both Base and BN20 regenerated; no mixing with historical native16 scores or recounting old costs. One fixed training seed; confidence intervals over questions, not seed population. No new training or three-language selection.'}
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({lang: data['paired'] for lang,data in languages.items()}))


if __name__ == '__main__':
    main()
