"""Batched math evaluation using pinned COPSD prompt/scorer/model functions.

Official AfriMGSM calls the original evaluate_afrimgsm and output processor.
Project single-sample evaluation adapts only the frozen local dataset schema,
per-question seeds and output accounting. No training code is imported.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from evaluate_transformers import load_items


def official_api(source, engine_api=None):
    from math_verify import parse, verify
    from transformers import AutoTokenizer

    constants = {'LANGUAGE_PROMPTS', 'POLYMATH_LANGUAGE_ALIASES',
                 'LANGUAGE_TO_AFRIMGSM_CONFIG'}
    functions = {'normalize_polymath_language', 'extract_boxed_answer',
                 'grade_answer', 'get_afrimgsm_ground_truth', 'build_qwen3_prompt',
                 'load_vllm_model', 'process_outputs_and_save', 'run_generation',
                 'evaluate_afrimgsm'}
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
    namespace = {'__name__': 'frozen_copsd_eval_functions', 'Path': Path,
                 'Counter': Counter, 'json': json, 'AutoTokenizer': AutoTokenizer,
                 'parse': parse, 'verify': verify}
    namespace.update(engine_api or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    for name in ['LANGUAGE_PROMPTS', 'extract_boxed_answer', 'grade_answer',
                 'build_qwen3_prompt', 'load_vllm_model']:
        if name not in namespace:
            raise ValueError('Missing pinned official function: ' + name)
    # Deliberately skip only the official CLI and its YOUR PATH cache settings.
    return namespace


def question_seed(seed, language, item_id):
    text = f'{seed}:{language}:{item_id}:0'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % 2**32


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-source', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-kind', choices=['mgsm_rev2', 'polymath_project', 'afrimgsm'], required=True)
    parser.add_argument('--benchmark', choices=['mgsm_rev2', 'polymath', 'gsm8k', 'afrimgsm'], required=True)
    parser.add_argument('--protocol', choices=['project_single_vllm_v1', 'afrimgsm_release_vllm_v1'], required=True)
    parser.add_argument('--language', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--max-items', type=int, default=0)
    args = parser.parse_args()
    if os.environ.get('GPU_QUEUE_GPU_COUNT') != '1':
        raise RuntimeError('One explicitly queued GPU required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.startswith('GPU-') or ',' in visible:
        raise RuntimeError('One GPU UUID required')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.adapter and not (args.adapter / 'adapter_model.safetensors').is_file():
        raise FileNotFoundError('Adapter required; never silently fall back to Base')
    release = args.protocol == 'afrimgsm_release_vllm_v1'
    if release and (args.benchmark, args.input_kind, args.language.lower()) != ('afrimgsm', 'afrimgsm', 'swa'):
        raise ValueError('Release protocol is the pinned SW AfriMGSM reference')
    if not release and args.benchmark == 'afrimgsm':
        raise ValueError('Use the separately identified official AfriMGSM protocol')
    expected_kind = {'mgsm_rev2': 'mgsm_rev2', 'polymath': 'polymath_project',
                     'gsm8k': 'polymath_project', 'afrimgsm': 'afrimgsm'}
    if args.input_kind != expected_kind[args.benchmark]:
        raise ValueError('Benchmark and input schema differ')

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    api = official_api(args.official_source, {'LLM': LLM, 'SamplingParams': SamplingParams})
    items = list(load_items(args.input, args.input_kind, args.language.lower()))
    expected = {'mgsm_rev2': 250, 'polymath': 500, 'gsm8k': 1319, 'afrimgsm': 250}[args.benchmark]
    if len(items) != expected:
        raise ValueError(f'Expected {expected} frozen benchmark questions, got {len(items)}')
    if args.max_items:
        if not 0 < args.max_items <= len(items):
            raise ValueError('Invalid engineering subset size')
        items = items[:args.max_items]
    samples, budget = (12, 1024) if release else (1, 2048)
    args.output_dir.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(samples=samples, max_new_tokens=budget, temperature=1.0,
                  top_p=0.95, top_k=-1, min_p=0.0, presence_penalty=0.0,
                  enable_thinking=True, model_context_limit=40960,
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  official_source_sha256=hashlib.sha256(args.official_source.read_bytes()).hexdigest(),
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  backend='vLLM; pinned COPSD functions, local-data compatibility',
                  seed_policy='Official SamplingParams seed=None; engine default0' if release else 'SHA256(seed:language:id:0), per request',
                  scope='engineering_subset' if args.max_items else 'frozen_formal_test',
                  packages={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'vllm', 'math-verify']})
    write_json(args.output_dir / 'CONFIG.json', config)
    write_json(args.output_dir / 'PHASE.json', {'phase': 'model_load'})
    load_start = time.monotonic()
    llm, tokenizer = api['load_vllm_model'](
        str(args.base_model), str(args.adapter) if args.adapter else None,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1, max_model_len=40960, enable_thinking=True)
    load_seconds = time.monotonic() - load_start
    request = LoRARequest('checkpoint_lora', 1, str(args.adapter)) if args.adapter else None
    captured = {}
    evaluation_started = time.monotonic()
    write_json(args.output_dir / 'PHASE.json', {'phase': 'generation'})
    if release:
        from datasets import Dataset
        with args.input.open(newline='') as stream:
            frozen_rows = list(csv.DictReader(stream, delimiter='\t'))[:len(items)]
        def frozen_dataset(name, language, split):
            if (name, language, split) != ('masakhane/afrimgsm', 'swa', 'test'):
                raise ValueError('Unexpected official dataset request')
            return Dataset.from_list(frozen_rows)
        print('Using locked local AfriMGSM snapshot; original evaluation function retained.', flush=True)
        api['load_dataset'] = frozen_dataset
        original_generate = api['run_generation']
        def recorded_generation(**kwargs):
            started = time.monotonic()
            outputs = original_generate(**kwargs)
            captured.update(outputs=outputs, generation_seconds=time.monotonic() - started)
            write_json(args.output_dir / 'PHASE.json', {'phase': 'official_grading_and_output'})
            return outputs
        api['run_generation'] = recorded_generation
        _, graded = api['evaluate_afrimgsm'](
            llm, tokenizer, max_new_tokens=budget, temperature=1.0, top_p=0.95,
            top_k=-1, min_p=0.0, presence_penalty=0.0, val_n=12,
            output_file=str(args.output_dir / 'OFFICIAL_RESULTS.json'),
            lora_request=request, base_model_name=str(args.base_model),
            enable_thinking=True, language='SWA', afrimgsm_split='test')
        outputs = captured['outputs']
        generation_seconds = captured['generation_seconds']
    else:
        prompts = [api['build_qwen3_prompt'](tokenizer, x['question'], args.language.upper(), True) for x in items]
        params = [SamplingParams(n=1, temperature=1.0, top_p=0.95, top_k=-1,
                                 min_p=0.0, presence_penalty=0.0, max_tokens=budget,
                                 seed=question_seed(args.seed, args.language.lower(), x['id'])) for x in items]
        started = time.monotonic()
        outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=True)
        generation_seconds = time.monotonic() - started
        write_json(args.output_dir / 'PHASE.json', {'phase': 'grading_and_output'})
        graded = None

    if len(outputs) != len(items):
        raise RuntimeError('Incomplete generation')
    totals = Counter()
    by_split = {}
    splits = {}
    if args.benchmark == 'polymath':
        splits = {x['id']: x['split'] for x in json.loads(args.input.read_text())}
    with (args.output_dir / 'ITEMS.jsonl').open('x') as stream:
        for i, (item, output) in enumerate(zip(items, outputs)):
            if len(output.outputs) != samples:
                raise RuntimeError('Wrong number of completions')
            if release and (graded[i]['problem'] != item['question'] or
                            str(graded[i]['ground_truth']) != str(item['answer'])):
                raise RuntimeError('Official output question/answer alignment differs')
            variants = []
            for j, completion in enumerate(output.outputs):
                ids = list(completion.token_ids)
                if len(ids) > budget:
                    raise RuntimeError('Generation exceeded fixed budget')
                boxed = api['extract_boxed_answer'](completion.text)
                correct = (bool(graded[i]['generations'][j]['correct']) if release
                           else bool(api['grade_answer'](boxed, item['answer'])))
                variants.append({'sample': j, 'tokens': len(ids), 'token_ids': ids,
                                 'finished': completion.finish_reason == 'stop',
                                 'finish_reason': completion.finish_reason, 'stop_reason': completion.stop_reason,
                                 'boxed_answer': boxed, 'correct': correct,
                                 'raw_output': tokenizer.decode(ids, skip_special_tokens=False),
                                 'official_generated_text': completion.text})
                totals['generated_tokens'] += len(ids)
                totals['normal_terminations'] += completion.finish_reason == 'stop'
                totals['truncated'] += completion.finish_reason == 'length'
                totals['answer_extraction_failures'] += boxed is None
            totals['single_correct'] += variants[0]['correct']
            totals['pass_n_correct'] += any(v['correct'] for v in variants)
            row = {**item, 'prompt': output.prompt, 'samples': variants}
            if splits:
                split = splits[item['id']]
                row['split'] = split
                counter = by_split.setdefault(split, Counter())
                counter['items'] += 1
                counter['single_correct'] += variants[0]['correct']
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = {**dict(totals), 'items': len(items), 'generation_count': len(items)*samples,
              'single_accuracy': totals['single_correct']/len(items),
              'pass_n': totals['pass_n_correct']/len(items), 'n': samples,
              'model_load_seconds': load_seconds, 'generation_seconds': generation_seconds,
              'evaluation_seconds': time.monotonic() - evaluation_started,
              'generated_tokens_per_generation_second': totals['generated_tokens']/generation_seconds,
              'protocol': args.protocol, 'benchmark': args.benchmark,
              'formal_full_test': not bool(args.max_items),
              'scores_by_difficulty': {k: {**dict(v), 'single_accuracy': v['single_correct']/v['items']} for k,v in by_split.items()}}
    if release:
        official = json.loads((args.output_dir / 'OFFICIAL_RESULTS.json').read_text())
        if official['pass_at_n'] != result['pass_n_correct']:
            raise RuntimeError('Official pass@12 count differs')
    write_json(args.output_dir / 'DONE.json', result)
    write_json(args.output_dir / 'PHASE.json', {'phase': 'done'})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
