"""Recovered checkpoint20 additional development200, official prompts/scorer, native LoRA, concurrency16, independent4B."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
import time


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--language', required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    code = Path(__file__).parent
    freeze_path = code / 'te4b_resume5_eval_frozen_20261024.PUBLIC.json'
    frozen = json.loads(freeze_path.read_text())
    assert frozen['protocol'] == 'project_TE4B_resume5_followup_dev200_single8192_c16_v1'
    for name,digest in frozen['runtime_code_sha256'].items(): assert sha(code/name)==digest,name
    spec = dict(frozen['cases'][a.case])
    release_path=None
    if spec['adapter_relative']:
        release_path=a.root/frozen['checkpoint_release_relative']
        release=json.loads(release_path.read_text())
        assert release['evaluation_freeze_sha256']==sha(freeze_path)
        assert release['training_freeze_sha256']==frozen['training_freeze_sha256']
        assert release['global_step']==20 and release['training_seed']==release['data_seed']==42
        spec['adapter_sha256']=release['adapter_sha256']
    assert a.language in spec['evaluation_languages']
    lang, root = a.language, a.root
    lock=root/frozen['model_lock_relative']; assert sha(lock)==frozen['model_lock_sha256']
    identity=json.loads(lock.read_text()); assert identity['revision']==frozen['model_revision']
    for name,info in identity['files'].items(): assert sha(root/identity['model_relative']/name)==info['sha256'],name
    source = root / 'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    assert sha(source) == frozen['official_evaluation_source_sha256']
    data = root / frozen['inputs'][lang]['relative']
    assert sha(data) == frozen['inputs'][lang]['sha256']
    rows = json.loads(data.read_text())
    assert [str(r['problem_id']) for r in rows] == frozen['selected_IDs']
    items = [{'id': str(r['problem_id']), 'question': r['problem'] if lang == 'en' else r['problem_' + lang], 'answer': r['Answer']} for r in rows]
    assert len(items) == len({r['id'] for r in items}) == 200
    adapter = root / spec['adapter_relative'] if spec['adapter_relative'] else None
    if adapter:
        assert adapter.resolve().is_relative_to((root / frozen['checkpoint_container_relative']).resolve())
        state = json.loads((adapter / 'trainer_state.json').read_text())
        assert state['global_step'] == 20 and state['max_steps'] == 100
        assert sha(adapter / 'adapter_model.safetensors') == spec['adapter_sha256']
    run_id = f'te4b_resume5_followup200_{a.case}_{lang}_20261024'
    output = root / 'logs/copsd_official_10lang/te4b_resume5_followup_development200' / run_id
    assert not output.exists()
    if a.prepare_only:
        print(f'{a.case}/{lang}: frozen200 IDs, data/source/adapter hashes and output preflight passed; no inference.')
        return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    from evaluate_vllm import official_api, question_seed, write_json
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    def resource_llm(**kwargs):
        assert 'max_num_seqs' not in kwargs
        return LLM(**kwargs, max_num_seqs=16)
    api = official_api(source, {'LLM': resource_llm, 'SamplingParams': SamplingParams})
    processes = subprocess.run(['nvidia-smi', '-i', visible, '--query-compute-apps=pid', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    lines = [s.strip() for s in processes.stdout.splitlines() if s.strip()]
    assert all(s.isdigit() for s in lines), processes.stdout
    assert all(int(s) == os.getpid() for s in lines), 'Assigned GPU acquired another compute process; refusing model load.'
    output.mkdir(parents=True)
    config = {'protocol': frozen['protocol'], 'case': a.case, 'language': lang,
        'model': str(root / 'models/Qwen3-4B'), 'adapter': str(adapter) if adapter else None,
        'model_revision': frozen['model_revision'], 'cohort': 'independent_4B',
        'adapter_sha256': spec.get('adapter_sha256'), 'checkpoint_release_sha256': sha(release_path) if release_path else None, 'checkpoint_step': 20 if adapter else None,
        'items': 200, 'samples': 1, 'max_new_tokens': 8192, 'max_num_seqs': 16,
        'temperature': 1., 'top_p': .95, 'top_k': -1, 'seed': 42,
        'source': str(source), 'input': str(data), 'output': str(output),
        'input_sha256': sha(data), 'official_source_sha256': sha(source),
        'data_freeze_sha256': sha(freeze_path), 'evaluator_sha256': sha(Path(__file__)),
        'packages': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'vllm', 'math-verify']},
        'scope': 'Fixed200 excluded before4Btraining, originalnativeprefix/native16. TE20R isOriginalBaseFirst5plusResume15updates withTrainer/data seed42 andexpandableCUDAallocator; BaseTEandBaseEnglish200reusedfromacceptednative16. Fixed200questionIDs previouslyusedin1.7B analysis, notnewblindtest orparametersearch.'}
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
    request = LoRARequest('checkpoint_lora', 1, str(adapter)) if adapter else None
    write_json(output / 'PHASE.json', {'phase': 'generation', 'actual_max_num_seqs': actual_seqs})
    tick = time.monotonic()
    outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=True)
    seconds = time.monotonic() - tick
    assert len(outputs) == 200
    totals = {'items': 200, 'single_correct': 0, 'generated_tokens': 0,
              'truncated': 0, 'normal_terminations': 0, 'answer_extraction_failures': 0}
    with (output / 'ITEMS.jsonl').open('x') as stream:
        for item, prompt, out in zip(items, prompts, outputs):
            assert out.prompt == prompt and len(out.outputs) == 1
            response = out.outputs[0]
            ids = list(response.token_ids)
            assert 0 < len(ids) <= 8192 and response.finish_reason in ['stop', 'length']
            answer = api['extract_boxed_answer'](response.text)
            correct = bool(api['grade_answer'](answer, item['answer']))
            sample = {'seed': question_seed(42, lang, item['id']), 'correct': correct,
                'boxed_answer': answer, 'token_ids': ids, 'tokens': len(ids),
                'finish_reason': response.finish_reason, 'raw_output': tokenizer.decode(ids, skip_special_tokens=False)}
            totals['single_correct'] += correct
            totals['generated_tokens'] += len(ids)
            totals['truncated'] += response.finish_reason == 'length'
            totals['normal_terminations'] += response.finish_reason == 'stop'
            totals['answer_extraction_failures'] += answer is None
            stream.write(json.dumps({**item, 'prompt': prompt, 'samples': [sample]}, ensure_ascii=False) + '\n')
    totals.update(protocol=config['protocol'], formal_test=False, n=1, generation_count=200,
        single_accuracy=totals['single_correct'] / 200, generation_seconds=seconds,
        model_load_seconds=load_seconds, actual_max_num_seqs=actual_seqs)
    write_json(output / 'DONE.json', totals)
    write_json(output / 'PHASE.json', {'phase': 'done'})
    print(json.dumps(totals), flush=True)


if __name__ == '__main__': main()
