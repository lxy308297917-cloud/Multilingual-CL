"""Same-GPU native-LoRA concurrency comparison on64 frozen development prompts.

Engineering measurements only: full8192 budget, no training or score selection.
Each mode runs in a fresh process so the previous engine cannot retain memory.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_vllm import official_api, question_seed, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, code):
    lock = json.loads((code / 'eval_concurrency8192_frozen_20261021.PUBLIC.json').read_text())
    for name, sha in lock['code_sha256'].items():
        assert digest(code / name) == sha, name
    config_path = root / lock['reference_config_relative']
    assert digest(config_path) == lock['reference_config_sha256']
    config = json.loads(config_path.read_text())
    assert config['language'] == 'sw' and config['checkpoint_step'] == 20
    assert config['protocol'] == 'project_dev_single_vllm8192_v2'
    assert config['max_new_tokens'] == 8192 and config['samples'] == 1
    assert digest(config['input']) == config['input_sha256'] == lock['input_sha256']
    assert digest(config['source']) == config['official_source_sha256']
    assert digest(Path(config['adapter']) / 'adapter_model.safetensors') == lock['adapter_sha256']
    assert json.loads((Path(config['adapter']) / 'trainer_state.json').read_text())['global_step'] == 20
    assert all(importlib.metadata.version(name) == version for name, version in config['packages'].items())
    rows = json.loads(Path(config['input']).read_text())
    assert len(rows) == len({r['problem_id'] for r in rows}) == 100
    return lock, config, rows[:64]


def wait_for_empty_GPU(visible):
    for attempt in range(16):
        value = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader'], text=True)
        pids = [int(line.split(',')[1]) for line in value.splitlines()
                if line.split(',')[0].strip() == visible]
        if not pids:
            return
        # Never signal GPU PIDs. A previous engine may still be exiting.
        if any(Path(f'/proc/{pid}').exists() and Path(f'/proc/{pid}').stat().st_uid != os.getuid()
               for pid in pids):
            raise RuntimeError('Other-user compute appeared on assigned GPU; refusing launch')
        if attempt == 15:
            raise RuntimeError('Assigned GPU still has compute; refusing overlap')
        time.sleep(2)


def worker(root, code, output, concurrency):
    lock, reference, rows = prepare(root, code)
    destination = output / f'concurrency{concurrency}'
    assert not destination.exists()
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    wait_for_empty_GPU(visible)
    destination.mkdir(parents=True)
    write_json(destination / 'CONFIG.json', {'reference': reference,
        'benchmark': lock, 'actual_concurrency_requested': concurrency,
        'CUDA_VISIBLE_DEVICES': visible, 'items': 64})
    telemetry = destination / 'gpu.csv'
    with telemetry.open('x') as stream:
        monitor = subprocess.Popen(['nvidia-smi', '-i', visible,
            '--query-gpu=timestamp,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits', '-l', '1'], stdout=stream, stderr=subprocess.STDOUT)
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            def configured_llm(**kwargs):
                assert 'max_num_seqs' not in kwargs
                return LLM(**kwargs, max_num_seqs=concurrency)
            api = official_api(Path(reference['source']), {'LLM': configured_llm, 'SamplingParams': SamplingParams})
            tick = time.monotonic()
            llm, tokenizer = api['load_vllm_model'](reference['model'], reference['adapter'],
                gpu_memory_utilization=.9, tensor_parallel_size=1, max_model_len=40960, enable_thinking=True)
            load_seconds = time.monotonic() - tick
            assert llm.llm_engine.vllm_config.scheduler_config.max_num_seqs == concurrency
            prompts = [api['build_qwen3_prompt'](tokenizer, r['problem_sw'], 'SW', True) for r in rows]
            params = [SamplingParams(n=1, temperature=1., top_p=.95, top_k=-1, min_p=0.,
                presence_penalty=0., max_tokens=8192, seed=question_seed(42, 'sw', r['problem_id'])) for r in rows]
            request = LoRARequest('checkpoint_lora', 1, reference['adapter'])
            tick = time.monotonic()
            outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=True)
            generation_seconds = time.monotonic() - tick
            assert len(outputs) == len(rows) == 64
            records = []
            for item, prompt, out in zip(rows, prompts, outputs):
                assert out.prompt == prompt and len(out.outputs) == 1
                response = out.outputs[0]
                tokens = list(response.token_ids)
                assert len(tokens) <= 8192
                records.append({'id': item['problem_id'], 'question': item['problem_sw'],
                    'answer': item['Answer'], 'prompt': prompt,
                    'seed': question_seed(42, 'sw', item['problem_id']),
                    'token_ids': tokens, 'finish_reason': response.finish_reason,
                    'raw_output': tokenizer.decode(tokens, skip_special_tokens=False)})
            with (destination / 'ITEMS.jsonl').open('x') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        finally:
            monitor.terminate()  # Only this function's own telemetry child.
            monitor.wait(timeout=10)
    readings = []
    for line in telemetry.read_text().splitlines():
        fields = line.split(',')
        if len(fields) == 3:
            try: readings.append((float(fields[1]), float(fields[2])))
            except ValueError: pass
    assert readings
    count = sum(len(r['token_ids']) for r in records)
    result = {'concurrency': concurrency, 'items': 64, 'generated_tokens': count,
        'generation_seconds': generation_seconds, 'load_seconds': load_seconds,
        'tokens_per_generation_second': count / generation_seconds,
        'samples_per_generation_second': 64 / generation_seconds,
        'sampled_physical_GPU_peak_MiB': max(x[0] for x in readings),
        'sampled_GPU_util_mean_percent': sum(x[1] for x in readings) / len(readings),
        'truncated': sum(r['finish_reason'] == 'length' for r in records),
        'items_sha256': digest(destination / 'ITEMS.jsonl'),
        'scope': 'Full-budget native-LoRA engineering benchmark, not a mathematics accuracy report. Physical telemetry includes load/display overhead.'}
    write_json(destination / 'DONE.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=[16, 32, 64])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    code = Path(__file__).parent
    lock, _, _ = prepare(args.root, code)
    output = args.root / lock['output_relative']
    if args.prepare_only:
        assert not output.exists()
        print(json.dumps({'CPU_preflight': True, 'GPU_started': False, 'items_each': 64,
                          'concurrency': [16, 32, 64], 'max_new_tokens': 8192}))
        return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    if args.concurrency:
        worker(args.root, code, output, args.concurrency)
        return
    assert not output.exists()
    output.mkdir(parents=True)
    for concurrency in [16, 32, 64]:
        subprocess.run([sys.executable, str(Path(__file__)), '--root', str(args.root),
                        '--concurrency', str(concurrency)], check=True)
    metrics, records = {}, {}
    for concurrency in [16, 32, 64]:
        folder = output / f'concurrency{concurrency}'
        metrics[concurrency] = json.loads((folder / 'DONE.json').read_text())
        records[concurrency] = [json.loads(x) for x in (folder / 'ITEMS.jsonl').read_text().splitlines()]
        for before, after in zip(records[16], records[concurrency]):
            assert all(before[k] == after[k] for k in ['id', 'question', 'answer', 'prompt', 'seed'])
        metrics[concurrency]['generation_time_speedup_vs16'] = metrics[16]['generation_seconds'] / metrics[concurrency]['generation_seconds']
        metrics[concurrency]['exact_token_sequence_matches_vs16'] = sum(a['token_ids'] == b['token_ids'] for a, b in zip(records[16], records[concurrency]))
    write_json(output / 'SUMMARY.PUBLIC.json', {'observed_at': datetime.datetime.now().isoformat(),
        'metrics': metrics, 'new_generation_responses': 192, 'same_GPU': visible,
        'scope': 'Engineering comparison only; not a training or formal efficacy result. Batch changes may alter sampled outputs despite equal per-request seeds; exact-match counts are measured rather than assumed. Existing frozen runs are separate; future scientific comparisons must use one fixed adopted configuration for all models.'})


if __name__ == '__main__':
    main()
