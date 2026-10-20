"""Paired2048/8192 inference on first32 frozen development questions.

Both budgets rerun the same32 prompts with the same model/request seeds, so
cached100-item batch differences do not masquerade as the length effect.
Training, existing evaluators and formal results remain frozen.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from evaluate_vllm import official_api, question_seed, write_json


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--language', choices=['sw', 'bn', 'fr'], required=True)
    p.add_argument('--mode', choices=['base', 'ckpt100'], required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    root, lang = a.root, a.language
    version = 'v1' if a.mode == 'base' or lang in ['sw', 'bn'] else 'v2'
    name = f'{lang}_17b_dev100_vllm_{a.mode}_20261020_{version}'
    cached = root/'logs/copsd_official_10lang/development_vllm'/name
    cfg = read(cached/'CONFIG.json')
    assert cfg['protocol'] == 'project_dev_single_vllm_v1' and cfg['max_new_tokens'] == 2048
    assert cfg['language'] == lang and cfg['samples'] == 1
    assert cfg['temperature'] == 1 and cfg['top_p'] == .95 and cfg['top_k'] == -1
    source = Path(cfg['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == cfg['official_source_sha256']
    assert hashlib.sha256(Path(cfg['input']).read_bytes()).hexdigest() == cfg['input_sha256']
    assert hashlib.sha256(Path(__file__).with_name('evaluate_dev_vllm.py').read_bytes()).hexdigest() == cfg['evaluator_sha256']
    original = [json.loads(s) for s in (cached/'ITEMS.jsonl').read_text().splitlines()]
    assert len(original) == 100 and len({r['id'] for r in original}) == 100
    rows = original[:32]
    for r in rows:
        assert len(r['samples']) == 1 and r['samples'][0]['seed'] == question_seed(42, lang, r['id'])
    if a.mode == 'ckpt100':
        cp = Path(cfg['adapter'])
        assert read(cp/'trainer_state.json')['global_step'] == 100 and (cp/'adapter_model.safetensors').stat().st_size > 0
    else:
        assert cfg['adapter'] is None
    output = root/'logs/copsd_official_10lang/length_sensitivity'/f'{lang}_{a.mode}_dev32_2048_8192_20261020_v1'
    assert not output.exists()
    if a.prepare_only:
        print(f'{lang}/{a.mode}: first32 frozen prompts/seeds, cached identity and checkpoint verified; no inference.')
        return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    api = official_api(source, {'LLM': LLM, 'SamplingParams': SamplingParams})
    output.mkdir(parents=True)
    write_json(output/'CONFIG.PRIVATE.json', {**cfg, 'cached_source': str(cached),
        'protocol': 'project_development_length_sensitivity32_v1', 'selection': 'first32 frozen dev order',
        'new_budgets': [2048, 8192], 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write_json(output/'PHASE.json', {'phase': 'model_load'})
    tick = time.monotonic()
    llm, tokenizer = api['load_vllm_model'](cfg['model'], cfg['adapter'], gpu_memory_utilization=.9,
        tensor_parallel_size=1, max_model_len=40960, enable_thinking=True)
    load_seconds = time.monotonic()-tick
    prompts = [r['prompt'] for r in rows]
    assert prompts == [api['build_qwen3_prompt'](tokenizer, r['question'], lang.upper(), True) for r in rows]
    request = LoRARequest('checkpoint_lora', 1, cfg['adapter']) if cfg['adapter'] else None
    all_rows, metrics = {}, {}
    for budget in [2048, 8192]:
        write_json(output/'PHASE.json', {'phase': f'generation_{budget}'})
        params = [SamplingParams(n=1, temperature=1., top_p=.95, top_k=-1, min_p=0.,
            presence_penalty=0., max_tokens=budget, seed=r['samples'][0]['seed']) for r in rows]
        tick = time.monotonic()
        outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=True)
        seconds = time.monotonic()-tick
        assert len(outputs) == 32
        samples = []
        totals = {'items': 32, 'single_correct': 0, 'generated_tokens': 0, 'truncated': 0,
                  'normal_terminations': 0, 'answer_extraction_failures': 0}
        for row, response, prompt in zip(rows, outputs, prompts):
            assert response.prompt == prompt and len(response.outputs) == 1
            result = response.outputs[0]
            answer = api['extract_boxed_answer'](result.text)
            correct = bool(api['grade_answer'](answer, row['answer']))
            ids = list(result.token_ids)
            assert len(ids) <= budget
            totals['single_correct'] += correct
            totals['generated_tokens'] += len(ids)
            totals['truncated'] += result.finish_reason == 'length'
            totals['normal_terminations'] += result.finish_reason == 'stop'
            totals['answer_extraction_failures'] += answer is None
            samples.append({'id': row['id'], 'question': row['question'], 'answer': row['answer'],
                'prompt': prompt, 'seed': row['samples'][0]['seed'], 'token_ids': ids,
                'boxed_answer': answer, 'correct': correct, 'finish_reason': result.finish_reason,
                'raw_output': tokenizer.decode(ids, skip_special_tokens=False)})
        with (output/f'max{budget}_ITEMS.PRIVATE.jsonl').open('x') as stream:
            for r in samples: stream.write(json.dumps(r, ensure_ascii=False)+'\n')
        totals['generation_seconds'] = seconds
        all_rows[budget], metrics[str(budget)] = samples, totals
        write_json(output/f'max{budget}_DONE.json', totals)
    first, second = all_rows[2048], all_rows[8192]
    result = {'protocol': 'project_development_length_sensitivity32_v1', 'language': lang, 'mode': a.mode,
        'metrics': metrics, 'model_load_seconds': load_seconds,
        'gained': sum(not x['correct'] and y['correct'] for x, y in zip(first, second)),
        'lost': sum(x['correct'] and not y['correct'] for x, y in zip(first, second)),
        'identical_generated_prefixes_out_of32': sum(y['token_ids'][:len(x['token_ids'])] == x['token_ids'] for x, y in zip(first, second)),
        'cached100_batch_first32_correct': sum(r['samples'][0]['correct'] for r in rows),
        'scope': 'Exploratory independent development first32; two freshly run length budgets, same32 prompts/model/seeds. Not official pass12, formal test, stable multilingual efficacy or a training change.'}
    write_json(output/'DONE.json', result)
    write_json(output/'PHASE.json', {'phase': 'done'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
