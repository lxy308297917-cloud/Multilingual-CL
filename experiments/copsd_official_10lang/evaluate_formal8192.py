"""Full project math suites at8192/n1, separate from official pass@12.

Uses the pinned official loader, language prefixes, extraction and grading.
No training or development inputs are used. Does not change frozen evaluators.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

from evaluate_transformers import load_items
from evaluate_vllm import official_api, question_seed, write_json

PROTOCOL = 'project_formal_single_vllm8192_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--benchmark', choices=['mgsm_rev2', 'polymath', 'gsm8k'], required=True)
    p.add_argument('--language', choices='sw te th bn zh ja ru de fr es en'.split(), required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--expected-step', type=int)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    root, lang = a.root.resolve(), a.language
    assert re.fullmatch(r'[A-Za-z0-9_-]+', a.run_id)
    assert (a.benchmark == 'gsm8k') == (lang == 'en')
    source = root / 'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    release = source.with_name('run_eval_4b_all_checkpoints.sh')
    assert re.search(r'(?m)^MAX_NEW_TOKENS=8192\s*$', release.read_text())
    if a.benchmark == 'mgsm_rev2':
        data = root / f'data/math_single_10lang_20261015/mgsm_rev2/MGSM-Rev2/mgsm_{lang}.tsv'
        kind, count = 'mgsm_rev2', 250
    elif a.benchmark == 'polymath':
        data = root / f'data/copsd_official_20261015/polymath_project_eval_v1/polymath_all500_{lang}.json'
        kind, count = 'polymath_project', 500
    else:
        data = root / 'data/copsd_official_20261015/gsm8k_project_en_zeroshot_v1/gsm8k_en_test_1319.json'
        kind, count = 'polymath_project', 1319
    lock = json.loads((root / 'logs/copsd_official_10lang/eval_vllm_runtime_20261019.CONFIG.json').read_text())
    assert lock['official_commit'] == '4f4410730e9b6775a30d20f4b6e141cba2e87590'
    assert digest(source) == lock['source_sha256']['polymath_eval/evaluate_math.py']
    assert digest(data) == lock['input_sha256'][str(data.relative_to(root))]
    assert digest(Path(lock['model_lock']['path'])) == lock['model_lock']['sha256']
    for name in ['evaluate_transformers.py', 'evaluate_vllm.py', 'compat/vllm011_uuid/sitecustomize.py']:
        assert digest(Path(__file__).parent / name) == lock['code_sha256'][name], name
    items = list(load_items(data, kind, lang))
    assert len(items) == len({r['id'] for r in items}) == count
    difficulties = {}
    if a.benchmark == 'polymath':
        difficulties = {r['id']: r['split'] for r in json.loads(data.read_text())}
        assert len(difficulties) == 500 and sorted(Counter(difficulties.values()).values()) == [125] * 4
    output = root / 'logs/copsd_official_10lang/evaluation_vllm8192' / a.run_id
    assert not output.exists()
    if a.adapter:
        assert a.adapter.resolve().is_relative_to((root / 'models/checkpoint').resolve())
        assert a.expected_step is not None and a.expected_step > 0
        assert json.loads((a.adapter / 'trainer_state.json').read_text())['global_step'] == a.expected_step
        assert (a.adapter / 'adapter_model.safetensors').stat().st_size > 0
    else:
        assert a.expected_step is None
    config = {'protocol': PROTOCOL, 'benchmark': a.benchmark, 'language': lang,
        'model': str(root / 'models/Qwen3-1.7B'),
        'model_revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
        'adapter': str(a.adapter) if a.adapter else None, 'checkpoint_step': a.expected_step,
        'adapter_sha256': digest(a.adapter / 'adapter_model.safetensors') if a.adapter else None,
        'input': str(data), 'input_sha256': digest(data), 'items': count, 'samples': 1,
        'max_new_tokens': 8192, 'temperature': 1., 'top_p': .95, 'top_k': -1,
        'seed': 42, 'seed_policy': 'SHA256(seed:language:id:0), per request',
        'max_num_seqs': 16, 'model_context_limit': 40960,
        'official_source_sha256': digest(source), 'released_length_script_sha256': digest(release),
        'evaluator_sha256': digest(Path(__file__)),
        'scope': 'Full project formal single-answer report at released length, not officialpass12. Configuration/checkpoint selection must use development only.'}
    if a.prepare_only:
        print(json.dumps({'prepare_only': True, 'protocol': PROTOCOL, 'benchmark': a.benchmark,
                          'language': lang, 'items': count, 'input_sha256': config['input_sha256'],
                          'adapter_provided': a.adapter is not None, 'GPU_initialized': False}))
        return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    def resource_configured_llm(**kwargs):
        assert 'max_num_seqs' not in kwargs
        return LLM(**kwargs, max_num_seqs=16)
    api = official_api(source, {'LLM': resource_configured_llm, 'SamplingParams': SamplingParams})
    config['packages'] = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'vllm', 'math-verify']}
    output.mkdir(parents=True)
    write_json(output / 'CONFIG.json', config)
    write_json(output / 'PHASE.json', {'phase': 'model_load'})
    tick = time.monotonic()
    llm, tokenizer = api['load_vllm_model'](config['model'], config['adapter'],
        gpu_memory_utilization=.9, tensor_parallel_size=1, max_model_len=40960, enable_thinking=True)
    load_seconds = time.monotonic() - tick
    actual_seqs = llm.llm_engine.vllm_config.scheduler_config.max_num_seqs
    assert actual_seqs == 16
    prompts = [api['build_qwen3_prompt'](tokenizer, r['question'], lang.upper(), True) for r in items]
    params = [SamplingParams(n=1, temperature=1., top_p=.95, top_k=-1, min_p=0., presence_penalty=0.,
        max_tokens=8192, seed=question_seed(42, lang, r['id'])) for r in items]
    request = LoRARequest('checkpoint_lora', 1, str(a.adapter)) if a.adapter else None
    write_json(output / 'PHASE.json', {'phase': 'generation', 'actual_max_num_seqs': actual_seqs})
    tick = time.monotonic()
    outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=True)
    seconds = time.monotonic() - tick
    assert len(outputs) == count
    totals = Counter(items=count)
    by_difficulty = {}
    with (output / 'ITEMS.jsonl').open('x') as stream:
        for item, prompt, out in zip(items, prompts, outputs):
            assert out.prompt == prompt and len(out.outputs) == 1
            response = out.outputs[0]
            ids = list(response.token_ids)
            assert len(ids) <= 8192
            answer = api['extract_boxed_answer'](response.text)
            correct = bool(api['grade_answer'](answer, item['answer']))
            sample = {'seed': question_seed(42, lang, item['id']), 'correct': correct, 'boxed_answer': answer,
                'token_ids': ids, 'tokens': len(ids), 'finish_reason': response.finish_reason,
                'raw_output': tokenizer.decode(ids, skip_special_tokens=False)}
            totals['single_correct'] += correct
            totals['generated_tokens'] += len(ids)
            totals['truncated'] += response.finish_reason == 'length'
            totals['normal_terminations'] += response.finish_reason == 'stop'
            totals['answer_extraction_failures'] += answer is None
            row = {**item, 'prompt': prompt, 'samples': [sample]}
            if difficulties:
                level = difficulties[item['id']]
                row['split'] = level
                d = by_difficulty.setdefault(level, Counter())
                d['items'] += 1
                d['single_correct'] += correct
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = {**dict(totals), 'protocol': PROTOCOL, 'benchmark': a.benchmark,
        'formal_full_test': True, 'n': 1, 'generation_count': count,
        'single_accuracy': totals['single_correct'] / count,
        'generation_seconds': seconds, 'model_load_seconds': load_seconds,
        'actual_max_num_seqs': actual_seqs,
        'scores_by_difficulty': {k: {**dict(v), 'single_accuracy': v['single_correct'] / v['items']}
                                 for k, v in by_difficulty.items()}}
    write_json(output / 'DONE.json', result)
    write_json(output / 'PHASE.json', {'phase': 'done'})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
