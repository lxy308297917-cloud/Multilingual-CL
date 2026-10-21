"""Length8192 diagnostic using unchanged official student/privileged prefixes."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from evaluate_vllm import official_api, question_seed, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    assert not args.output.exists()
    cfg = json.loads((args.prepared / 'CONFIG.PRIVATE.json').read_text())
    assert cfg['protocol'] == 'privileged_teacher_conditioning_diagnostic_v1'
    source = Path(cfg['source'])
    for file, key in [('multilingual_data_collator.py', 'collator_sha256'),
                      ('language_config.py', 'language_config_sha256')]:
        assert hashlib.sha256((source / file).read_bytes()).hexdigest() == cfg[key]
    prompts = args.prepared / 'PROMPTS.PRIVATE.json'
    assert hashlib.sha256(prompts.read_bytes()).hexdigest() == cfg['prompts_sha256']
    rows = json.loads(prompts.read_text())
    assert len(rows) == 100 and len({r['id'] for r in rows}) == 100
    from vllm import LLM, SamplingParams
    api = official_api(source / 'polymath_eval/evaluate_math.py', {'LLM': LLM, 'SamplingParams': SamplingParams})
    args.output.mkdir(parents=True)
    config = {**cfg, 'protocol': 'privileged_teacher_conditioning_length8192_v1', 'samples': 1, 'max_new_tokens': 8192, 'temperature': 1.0,
        'top_p': 0.95, 'top_k': -1, 'seed': 42, 'adapter': None,
        'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'packages': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'vllm', 'math-verify']}}
    write_json(args.output / 'CONFIG.PRIVATE.json', config)
    write_json(args.output / 'PHASE.json', {'phase': 'model_load'})
    tick = time.monotonic()
    llm, tokenizer = api['load_vllm_model'](cfg['model'], None, gpu_memory_utilization=0.9,
        tensor_parallel_size=1, max_model_len=40960, enable_thinking=True)
    load_seconds = time.monotonic() - tick
    actual_seqs = llm.llm_engine.vllm_config.scheduler_config.max_num_seqs
    assert isinstance(actual_seqs, int) and actual_seqs > 0
    config['actual_max_num_seqs'] = actual_seqs
    write_json(args.output / 'CONFIG.PRIVATE.json', config)
    summaries = {}
    for condition in ['student', 'teacher']:
        write_json(args.output / 'PHASE.json', {'phase': condition + '_generation'})
        inputs = [{'prompt_token_ids': row[condition + '_prompt_token_ids']} for row in rows]
        sampling = [SamplingParams(n=1, temperature=1.0, top_p=0.95, top_k=-1,
            min_p=0.0, presence_penalty=0.0, max_tokens=8192,
            seed=question_seed(42, cfg['language'], row['id'])) for row in rows]
        tick = time.monotonic()
        outputs = llm.generate(inputs, sampling, use_tqdm=True)
        seconds = time.monotonic() - tick
        assert len(outputs) == 100
        totals = {'items': 100, 'correct': 0, 'normal_terminations': 0, 'truncated': 0,
                  'answer_extraction_failures': 0, 'generated_tokens': 0}
        with (args.output / (condition + '_ITEMS.PRIVATE.jsonl')).open('x') as stream:
            for row, output in zip(rows, outputs):
                assert output.prompt_token_ids == row[condition + '_prompt_token_ids']
                assert len(output.outputs) == 1
                response = output.outputs[0]
                answer = api['extract_boxed_answer'](response.text)
                correct = bool(api['grade_answer'](answer, row['answer']))
                totals['correct'] += correct
                totals['normal_terminations'] += response.finish_reason == 'stop'
                totals['truncated'] += response.finish_reason == 'length'
                totals['answer_extraction_failures'] += answer is None
                totals['generated_tokens'] += len(response.token_ids)
                stream.write(json.dumps({'id': row['id'], 'answer': row['answer'],
                    'seed': question_seed(42, cfg['language'], row['id']), 'correct': correct,
                    'boxed_answer': answer, 'finish_reason': response.finish_reason,
                    'token_ids': list(response.token_ids),
                    'text': tokenizer.decode(response.token_ids, skip_special_tokens=False)}, ensure_ascii=False) + '\n')
        totals['generation_seconds'] = seconds
        summaries[condition] = totals
        write_json(args.output / (condition + '_DONE.json'), totals)
    result = {'protocol': 'privileged_teacher_conditioning_length8192_v1', 'max_new_tokens': 8192, 'actual_max_num_seqs': actual_seqs, 'language': cfg['language'], 'formal_test': False,
        'conditions': summaries, 'model_load_seconds': load_seconds,
        'teacher_adapter': None, 'same_original_Base': True,
        'scope': 'Teacher receives English reference. Conditional-generation diagnostic only; not fair benchmark accuracy, not training-label synthesis, and not an evaluation of all on-policy teacher-token distributions.'}
    write_json(args.output / 'DONE.json', result)
    write_json(args.output / 'PHASE.json', {'phase': 'done'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
