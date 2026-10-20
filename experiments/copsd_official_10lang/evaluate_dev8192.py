"""Fixed development100 at the released PolyMath8192 response budget.

Official loader/prompt/scorer are retained. max_num_seqs16 is a declared
resource setting to fit long responses in the measured~155k-token KV cache.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

from evaluate_vllm import official_api, question_seed, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--language', choices='sw te th bn zh ja ru de fr es en'.split(), required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--expected-step', type=int)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', a.run_id)
    root, lang = a.root, a.language
    source = root/'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    released = source.with_name('run_eval_4b_all_checkpoints.sh')
    assert re.search(r'(?m)^MAX_NEW_TOKENS=8192\s*$', released.read_text())
    data = root/f'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_{"sw" if lang == "en" else lang}.json'
    rr = json.loads(data.read_text())
    items = [{'id': r['problem_id'], 'question': r['problem'] if lang == 'en' else r['problem_'+lang], 'answer': r['Answer']} for r in rr]
    assert len(items) == len({r['id'] for r in items}) == 100
    output = root/'logs/copsd_official_10lang/development_vllm8192'/a.run_id
    assert not output.exists()
    if a.adapter:
        assert a.adapter.resolve().is_relative_to((root/'models/checkpoint').resolve())
        assert a.expected_step is not None and json.loads((a.adapter/'trainer_state.json').read_text())['global_step'] == a.expected_step
        assert (a.adapter/'adapter_model.safetensors').stat().st_size > 0
    else:
        assert a.expected_step is None
    if a.prepare_only:
        print('Fixed development100/source8192/checkpoint/output preflight passed; no inference.')
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
    output.mkdir(parents=True)
    config = {'protocol': 'project_dev_single_vllm8192_v2', 'language': lang,
        'model': str(root/'models/Qwen3-1.7B'), 'adapter': str(a.adapter) if a.adapter else None,
        'checkpoint_step': a.expected_step, 'items': 100, 'samples': 1, 'max_new_tokens': 8192,
        'temperature': 1., 'top_p': .95, 'top_k': -1, 'seed': 42, 'max_num_seqs': 16,
        'source': str(source), 'input': str(data), 'output': str(output),
        'input_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
        'official_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'released_length_script_sha256': hashlib.sha256(released.read_bytes()).hexdigest(),
        'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'packages': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'vllm', 'math-verify']},
        'scope': 'Independent development100, official released length but projectn1/seeds; not formal or officialpass12.'}
    write_json(output/'CONFIG.json', config)
    write_json(output/'PHASE.json', {'phase': 'model_load'})
    tick = time.monotonic()
    llm, tokenizer = api['load_vllm_model'](config['model'], config['adapter'], gpu_memory_utilization=.9,
        tensor_parallel_size=1, max_model_len=40960, enable_thinking=True)
    load_seconds = time.monotonic()-tick
    actual_seqs = llm.llm_engine.vllm_config.scheduler_config.max_num_seqs
    assert actual_seqs == 16
    prompts = [api['build_qwen3_prompt'](tokenizer, r['question'], lang.upper(), True) for r in items]
    params = [SamplingParams(n=1, temperature=1., top_p=.95, top_k=-1, min_p=0., presence_penalty=0.,
        max_tokens=8192, seed=question_seed(42, lang, r['id'])) for r in items]
    request = LoRARequest('checkpoint_lora', 1, str(a.adapter)) if a.adapter else None
    write_json(output/'PHASE.json', {'phase': 'generation', 'actual_max_num_seqs': actual_seqs})
    tick = time.monotonic()
    outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=True)
    seconds = time.monotonic()-tick
    assert len(outputs) == 100
    totals = {'items': 100, 'single_correct': 0, 'generated_tokens': 0, 'truncated': 0,
              'normal_terminations': 0, 'answer_extraction_failures': 0}
    with (output/'ITEMS.jsonl').open('x') as stream:
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
            stream.write(json.dumps({**item, 'prompt': prompt, 'samples': [sample]}, ensure_ascii=False)+'\n')
    totals.update(protocol=config['protocol'], formal_test=False, n=1, generation_count=100,
        single_accuracy=totals['single_correct']/100, generation_seconds=seconds,
        model_load_seconds=load_seconds, actual_max_num_seqs=actual_seqs)
    write_json(output/'DONE.json', totals)
    write_json(output/'PHASE.json', {'phase': 'done'})
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
