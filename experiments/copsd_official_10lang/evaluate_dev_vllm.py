"""Project development evaluation; shares pinned official prompt/loader/scorer.

Separate protocol from historical HF dev and from formal tests. Supports the
same frozen held-out problem IDs in ten languages plus their English questions.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from evaluate_vllm import official_api, question_seed, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--language', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lang = args.language.lower()
    assert lang in 'sw te th bn zh ja ru de fr es en'.split()
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    assert not args.output.exists()
    if args.adapter:
        assert (args.adapter / 'adapter_model.safetensors').is_file()
    rows = json.loads(args.input.read_text())
    items = [{'id': r['problem_id'], 'question': r['problem'] if lang == 'en' else r['problem_'+lang],
              'answer': r['Answer']} for r in rows]
    assert len(items) == 100 and len({r['id'] for r in items}) == 100
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    api = official_api(args.source, {'LLM': LLM, 'SamplingParams': SamplingParams})
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(protocol='project_dev_single_vllm_v1', split='held_out_development100',
                  samples=1, max_new_tokens=2048, temperature=1.0, top_p=0.95, top_k=-1,
                  seed=42, seed_policy='SHA256(seed:language:id:0), per request',
                  enable_thinking=True, input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  official_source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  packages={n:importlib.metadata.version(n) for n in ['torch','transformers','vllm','math-verify']},
                  caveat='Changing backend can change samples; compare Base and adapter within this protocol only.')
    write_json(args.output/'CONFIG.json', config)
    write_json(args.output/'PHASE.json', {'phase':'model_load'})
    tick = time.monotonic()
    llm, tokenizer = api['load_vllm_model'](str(args.model), str(args.adapter) if args.adapter else None,
        gpu_memory_utilization=0.9, tensor_parallel_size=1, max_model_len=40960, enable_thinking=True)
    load_seconds = time.monotonic()-tick
    prompts = [api['build_qwen3_prompt'](tokenizer,x['question'],lang.upper(),True) for x in items]
    params = [SamplingParams(n=1,temperature=1.0,top_p=0.95,top_k=-1,min_p=0.0,
                            presence_penalty=0.0,max_tokens=2048,seed=question_seed(42,lang,x['id'])) for x in items]
    request = LoRARequest('checkpoint_lora',1,str(args.adapter)) if args.adapter else None
    write_json(args.output/'PHASE.json',{'phase':'generation'})
    tick=time.monotonic()
    outputs=llm.generate(prompts,params,lora_request=request,use_tqdm=True)
    generation_seconds=time.monotonic()-tick
    assert len(outputs)==len(items)
    write_json(args.output/'PHASE.json',{'phase':'grading'})
    totals={'items':100,'single_correct':0,'generated_tokens':0,'normal_terminations':0,
            'truncated':0,'answer_extraction_failures':0}
    with (args.output/'ITEMS.jsonl').open('x') as stream:
        for item,prompt,out in zip(items,prompts,outputs):
            assert out.prompt==prompt and len(out.outputs)==1
            result=out.outputs[0]; ids=list(result.token_ids)
            assert len(ids)<=2048
            boxed=api['extract_boxed_answer'](result.text)
            correct=bool(api['grade_answer'](boxed,item['answer']))
            sample={'sample':0,'seed':question_seed(42,lang,item['id']), 'correct':correct,
                    'boxed_answer':boxed,'tokens':len(ids),'token_ids':ids,
                    'finished':result.finish_reason=='stop','finish_reason':result.finish_reason,
                    'stop_reason':result.stop_reason,'raw_output':tokenizer.decode(ids,skip_special_tokens=False),
                    'official_generated_text':result.text}
            totals['single_correct']+=correct
            totals['generated_tokens']+=len(ids)
            totals['normal_terminations']+=sample['finished']
            totals['truncated']+=result.finish_reason=='length'
            totals['answer_extraction_failures']+=boxed is None
            stream.write(json.dumps({**item,'prompt':prompt,'samples':[sample]},ensure_ascii=False)+'\n')
    totals.update(single_accuracy=totals['single_correct']/100,n=1,generation_count=100,
                  protocol='project_dev_single_vllm_v1',formal_test=False,
                  generation_seconds=generation_seconds,model_load_seconds=load_seconds,
                  generated_tokens_per_generation_second=totals['generated_tokens']/generation_seconds)
    write_json(args.output/'DONE.json',totals)
    write_json(args.output/'PHASE.json',{'phase':'done'})
    print(json.dumps(totals),flush=True)


if __name__=='__main__':
    main()
