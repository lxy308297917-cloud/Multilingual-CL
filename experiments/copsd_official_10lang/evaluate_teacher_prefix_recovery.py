"""Frozen privileged teacher recovery after preselected student histories.

Diagnostic only: first32 independent BN development questions, first256 response
tokens from Base or failed BN100 student. Not a replay of original training KL.
Both conditions receive2048 NEW continuation tokens; prefix cost is separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from evaluate_vllm import official_api, question_seed, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args()
    assert not a.output.exists()
    root = a.root
    prepared = root/'logs/copsd_official_10lang/teacher_context/bn_dev100_20261020_v1'
    cfg = read(prepared/'CONFIG.PRIVATE.json')
    rows = read(prepared/'PROMPTS.PRIVATE.json')
    assert hashlib.sha256((prepared/'PROMPTS.PRIVATE.json').read_bytes()).hexdigest() == cfg['prompts_sha256']
    source = Path(cfg['source'])
    for name, key in [('multilingual_data_collator.py', 'collator_sha256'), ('language_config.py', 'language_config_sha256')]:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == cfg[key]
    data = root/'logs/copsd_official_10lang/development_vllm'
    names = {'Base_history': 'bn_17b_dev100_vllm_base_20261020_v1',
             'BN100_history': 'bn_17b_dev100_vllm_ckpt100_20261020_v1'}
    histories = {}
    for condition, name in names.items():
        rr = [json.loads(s) for s in (data/name/'ITEMS.jsonl').read_text().splitlines()]
        assert len(rr) == 100 and len({r['id'] for r in rr}) == 100
        assert [r['id'] for r in rr] == [r['id'] for r in rows]
        assert [r['answer'] for r in rr] == [r['answer'] for r in rows]
        assert all(len(r['samples'][0]['token_ids']) >= 256 for r in rr[:32])
        histories[condition] = rr[:32]
    if a.prepare_only:
        print('First32 IDs/answers aligned; both saved student histories have256 tokens. No GPU/model loaded.')
        return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    from vllm import LLM, SamplingParams
    api = official_api(source/'polymath_eval/evaluate_math.py', {'LLM': LLM, 'SamplingParams': SamplingParams})
    a.output.mkdir(parents=True)
    write_json(a.output/'CONFIG.PRIVATE.json', {
        **cfg, 'protocol': 'privileged_teacher_prefix_recovery_dev32_v1',
        'question_selection': 'first32 in frozen development manifest, no outcome selection',
        'prefix_tokens': 256, 'new_continuation_tokens': 2048, 'samples': 1,
        'temperature': 1.0, 'top_p': 0.95, 'top_k': -1, 'seed': 42,
        'conditions': names, 'teacher_adapter': None,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write_json(a.output/'PHASE.json', {'phase': 'model_load'})
    tick = time.monotonic()
    llm, tokenizer = api['load_vllm_model'](cfg['model'], None, gpu_memory_utilization=0.9,
        tensor_parallel_size=1, max_model_len=40960, enable_thinking=True)
    load_seconds = time.monotonic()-tick
    summaries = {}
    for condition, rr in histories.items():
        prefixes = [r['samples'][0]['token_ids'][:256] for r in rr]
        inputs = [{'prompt_token_ids': row['teacher_prompt_token_ids']+prefix}
                  for row, prefix in zip(rows[:32], prefixes)]
        sampling = [SamplingParams(n=1, temperature=1.0, top_p=0.95, top_k=-1,
            max_tokens=2048, seed=question_seed(42, 'bn', row['id'])) for row in rows[:32]]
        write_json(a.output/'PHASE.json', {'phase': condition})
        tick = time.monotonic()
        outputs = llm.generate(inputs, sampling, use_tqdm=True)
        seconds = time.monotonic()-tick
        assert len(outputs) == 32
        totals = {'items': 32, 'correct': 0, 'truncated': 0, 'normal_terminations': 0,
                  'answer_extraction_failures': 0, 'generated_tokens': 0,
                  'supplied_history_tokens': 8192, 'history_already_has_boxed': 0,
                  'teacher_prompt_tokens': sum(len(x['prompt_token_ids']) for x in inputs)}
        with (a.output/(condition+'_ITEMS.PRIVATE.jsonl')).open('x') as stream:
            for row, prefix, inp, output in zip(rows[:32], prefixes, inputs, outputs):
                assert output.prompt_token_ids == inp['prompt_token_ids'] and len(output.outputs) == 1
                response = output.outputs[0]
                prefix_text = tokenizer.decode(prefix, skip_special_tokens=False)
                combined = prefix_text+response.text
                answer = api['extract_boxed_answer'](combined)
                correct = bool(api['grade_answer'](answer, row['answer']))
                totals['correct'] += correct
                totals['truncated'] += response.finish_reason == 'length'
                totals['normal_terminations'] += response.finish_reason == 'stop'
                totals['answer_extraction_failures'] += answer is None
                totals['history_already_has_boxed'] += api['extract_boxed_answer'](prefix_text) is not None
                totals['generated_tokens'] += len(response.token_ids)
                stream.write(json.dumps({'id': row['id'], 'answer': row['answer'],
                    'prefix_token_ids': prefix, 'new_token_ids': list(response.token_ids),
                    'seed': question_seed(42, 'bn', row['id']), 'correct': correct,
                    'boxed_answer': answer, 'finish_reason': response.finish_reason,
                    'combined_text': combined}, ensure_ascii=False)+'\n')
        totals['generation_seconds'] = seconds
        summaries[condition] = totals
        write_json(a.output/(condition+'_DONE.json'), totals)
    result = {'protocol': 'privileged_teacher_prefix_recovery_dev32_v1', 'conditions': summaries,
        'model_load_seconds': load_seconds, 'teacher_adapter': None, 'formal_test': False,
        'scope': 'Frozen privileged Base continuation after equal256-token Base/BN100 development histories, with2048 new tokens. First32 fixed problems; not full training-distribution replay, a causal proof, fair benchmark or SFT synthesis.'}
    write_json(a.output/'DONE.json', result)
    write_json(a.output/'PHASE.json', {'phase': 'done'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
