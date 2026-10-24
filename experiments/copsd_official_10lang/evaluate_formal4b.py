"""Run one immutable 4B formal case with original prompt/generation/grading calls.

Engineering contracts are CPU-preflight-only. This runner never chooses
languages/checkpoints, starts queues, changes adapters or resumes training.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from evaluate_vllm import official_api, question_seed, write_json
from formal4b_contract import PROTOCOL, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    options = parser.parse_args()
    root = options.root.resolve()
    frozen, cases = prepare(root, options.freeze)
    if options.prepare_only:
        selected = cases if options.case == 'all' else {options.case: cases[options.case]}
        assert all(not case['output'].exists() for case in selected.values()), 'Never overwrite an evaluation'
        print(json.dumps({'prepare_only': True, 'protocol': PROTOCOL,
            'cases': {name: {'benchmark': case['spec']['benchmark'],
                'language': case['spec']['language'], 'items': case['count'],
                'adapter_provided': case['adapter'] is not None} for name, case in selected.items()},
            'GPU_initialized': False, 'engineering_preflight_only': frozen['engineering_preflight_only']}))
        return
    assert not frozen['engineering_preflight_only'], 'CPU fixture cannot launch formal inference'
    case = cases[options.case]
    assert not case['output'].exists(), 'Never overwrite an evaluation'
    a = SimpleNamespace(adapter=case['adapter'], benchmark=case['spec']['benchmark'])
    lang = case['spec']['language']
    source, items, count = case['source'], case['items'], case['count']
    difficulties, output, config = case['difficulties'], case['output'], case['config']
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    from benchmark_eval_concurrency8192 import wait_for_empty_GPU
    wait_for_empty_GPU(visible)
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
