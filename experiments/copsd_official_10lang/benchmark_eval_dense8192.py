"""One full-budget dense-merge throughput test against saved native LoRA16.

Engineering only: same64 SW development prompts,8192 budget,concurrency16.
Records changed token sequences and device identity; never reports math gains.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_eval_concurrency8192 import prepare as native_prepare, wait_for_empty_GPU
from evaluate_vllm import official_api, question_seed, write_json
from export_eval_dense_cpu import digest


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, code):
    lock = read(code / 'eval_dense8192_frozen_20261021.PUBLIC.json')
    for name, sha in lock['code_sha256'].items():
        assert digest(code / name) == sha, name
    _, reference, rows = native_prepare(root, code)
    export = root / lock['export_relative']
    manifest_path = export / 'EXPORT.MANIFEST.PRIVATE.json'
    assert digest(manifest_path) == lock['export_manifest_sha256']
    manifest = read(manifest_path)
    assert manifest['completed'] and manifest['backend'] == 'CPU_PEFT_safe_merge_BF16_v1'
    assert manifest['source_adapter_sha256'] == digest(Path(reference['adapter']) / 'adapter_model.safetensors')
    for name, info in manifest['files'].items():
        assert (export / name).stat().st_size == info['bytes'] and digest(export / name) == info['sha256']
    native = root / lock['native16_output_relative']
    metric = read(native / 'DONE.json')
    assert digest(native / 'ITEMS.jsonl') == metric['items_sha256'] == lock['native16_items_sha256']
    assert metric['concurrency'] == 16 and metric['items'] == 64
    old = [json.loads(line) for line in (native / 'ITEMS.jsonl').read_text().splitlines()]
    assert [r['id'] for r in old] == [r['problem_id'] for r in rows]
    output = root / lock['output_relative']
    assert not output.exists()
    return lock, reference, rows, export, native, old, metric, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args()
    code = Path(__file__).parent
    lock, reference, rows, export, native, old, native_metric, output = prepare(a.root, code)
    if a.prepare_only:
        print(json.dumps({'CPU_preflight': True, 'GPU_started': False, 'items': 64,
            'concurrency': 16, 'max_new_tokens': 8192, 'math_scoring': False}))
        return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    wait_for_empty_GPU(visible)
    output.mkdir(parents=True)
    write_json(output / 'CONFIG.json', {'frozen': lock, 'native_reference': reference,
        'CUDA_VISIBLE_DEVICES': visible, 'backend': 'independent_BF16_merged_dense',
        'concurrency': 16, 'max_new_tokens': 8192, 'items': 64})
    with (output / 'gpu.csv').open('x') as stream:
        monitor = subprocess.Popen(['nvidia-smi', '-i', visible,
            '--query-gpu=timestamp,memory.used,utilization.gpu', '--format=csv,noheader,nounits', '-l', '1'],
            stdout=stream, stderr=subprocess.STDOUT)
        try:
            from vllm import LLM, SamplingParams
            def configured(**kwargs):
                assert 'max_num_seqs' not in kwargs
                return LLM(**kwargs, max_num_seqs=16)
            api = official_api(Path(reference['source']), {'LLM': configured, 'SamplingParams': SamplingParams})
            tick = time.monotonic()
            llm, tokenizer = api['load_vllm_model'](str(export), None,
                gpu_memory_utilization=.9, tensor_parallel_size=1, max_model_len=40960, enable_thinking=True)
            load_seconds = time.monotonic() - tick
            assert llm.llm_engine.vllm_config.scheduler_config.max_num_seqs == 16
            prompts = [api['build_qwen3_prompt'](tokenizer, row['problem_sw'], 'SW', True) for row in rows]
            assert prompts == [row['prompt'] for row in old]
            params = [SamplingParams(n=1, temperature=1., top_p=.95, top_k=-1, min_p=0.,
                presence_penalty=0., max_tokens=8192, seed=question_seed(42, 'sw', row['problem_id'])) for row in rows]
            tick = time.monotonic()
            generated = llm.generate(prompts, params, use_tqdm=True)
            seconds = time.monotonic() - tick
            assert len(generated) == 64
            records = []
            for row, prompt, out in zip(rows, prompts, generated):
                assert out.prompt == prompt and len(out.outputs) == 1
                answer = out.outputs[0]
                tokens = list(answer.token_ids)
                assert len(tokens) <= 8192
                records.append({'id': row['problem_id'], 'question': row['problem_sw'],
                    'answer': row['Answer'], 'prompt': prompt, 'seed': question_seed(42, 'sw', row['problem_id']),
                    'token_ids': tokens, 'finish_reason': answer.finish_reason,
                    'raw_output': tokenizer.decode(tokens, skip_special_tokens=False)})
            with (output / 'ITEMS.jsonl').open('x') as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        finally:
            monitor.terminate()  # Only our telemetry child, never GPU-query PIDs.
            monitor.wait(timeout=10)
    readings = []
    for line in (output / 'gpu.csv').read_text().splitlines():
        fields = line.split(',')
        if len(fields) == 3:
            try: readings.append((float(fields[1]), float(fields[2])))
            except ValueError: pass
    assert readings
    count = sum(len(row['token_ids']) for row in records)
    native_device = read(native / 'CONFIG.json')['CUDA_VISIBLE_DEVICES']
    result = {'backend': 'independent_BF16_merged_dense', 'items': 64, 'generated_tokens': count,
        'generation_seconds': seconds, 'load_seconds': load_seconds,
        'tokens_per_generation_second': count / seconds, 'samples_per_generation_second': 64 / seconds,
        'observed_token_throughput_ratio_to_native16': (count / seconds) / native_metric['tokens_per_generation_second'],
        'observed_sample_throughput_ratio_to_native16': native_metric['generation_seconds'] / seconds,
        'native16_metrics': native_metric, 'same_physical_GPU_as_native16': native_device == visible,
        'sampled_physical_GPU_peak_MiB': max(v[0] for v in readings),
        'sampled_GPU_util_mean_percent': sum(v[1] for v in readings) / len(readings),
        'token_sequences_identical_to_native16': sum(x['token_ids'] == y['token_ids'] for x, y in zip(old, records)),
        'finish_reasons_identical_to_native16': sum(x['finish_reason'] == y['finish_reason'] for x, y in zip(old, records)),
        'truncated': sum(row['finish_reason'] == 'length' for row in records),
        'items_sha256': digest(output / 'ITEMS.jsonl'),
        'scope': 'Engineering throughput only on64 identical development prompts. BF16 merge changes numerics and can change trajectory lengths, so token/s and samples/s are both reported. Not mathematics efficacy, native-LoRA equivalence or a replacement protocol. Compare device identity and observed operating conditions before interpreting speed ratios. Source training checkpoints and existing formal outputs unchanged.'}
    write_json(output / 'DONE.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
