"""Fixed SW privileged-teacher continuation after equal saved evaluation histories.

No training or claim to replay actual training-time KL. First32 old development
questions, not the new followup200; 256 history tokens plus8192 new tokens.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from evaluate_vllm import official_api, question_seed, write_json


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(); root = a.root; code = Path(__file__).parent
    freeze = code / 'sw_teacher_recovery32_frozen_20261022.PUBLIC.json'
    f = read(freeze)
    for name, digest in f['code_sha256'].items(): assert sha(code / name) == digest, name
    prepared = root / f['prepared_relative']
    for name, digest in f['prepared_sha256'].items(): assert sha(prepared / name) == digest, name
    cfg = read(prepared / 'CONFIG.PRIVATE.json')
    assert cfg['language'] == 'sw' and cfg['include_problem_en'] and cfg['include_reference_solution_en']
    assert cfg['student_enable_thinking'] and cfg['reason_first'] is False
    source = Path(cfg['source'])
    for name, field in [('multilingual_data_collator.py', 'collator_sha256'), ('language_config.py', 'language_config_sha256')]:
        assert sha(source / name) == cfg[field]
    assert sha(source / 'polymath_eval/evaluate_math.py') == f['official_evaluation_sha256']
    rows = read(prepared / 'PROMPTS.PRIVATE.json')[:32]
    assert [r['id'] for r in rows] == f['selected_IDs'] and len(rows) == 32
    histories = {}
    for condition, spec in f['history_inputs'].items():
        path = root / spec['relative']; assert sha(path) == spec['sha256']
        saved = [json.loads(x) for x in path.read_text().splitlines()][:32]
        assert [r['id'] for r in saved] == f['selected_IDs']
        assert [r['answer'] for r in saved] == [r['answer'] for r in rows]
        assert all(len(r['samples'][0]['token_ids']) >= 256 for r in saved)
        histories[condition] = [r['samples'][0]['token_ids'][:256] for r in saved]
    histories['No_history'] = [[] for _ in rows]
    output = root / f['output_relative']; assert not output.exists()
    if a.prepare_only:
        print('Frozen32 IDs, all3 equal256-token histories, teacher prompts/source/output verified; no GPU.'); return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    proc = subprocess.run(['nvidia-smi', '-i', visible, '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    assert all(x.strip().isdigit() and int(x.strip()) == os.getpid() for x in proc.stdout.splitlines() if x.strip()), 'Foreign process on assigned GPU'
    from vllm import LLM, SamplingParams
    def resource_llm(**kwargs): return LLM(**kwargs, max_num_seqs=64)
    api = official_api(source / 'polymath_eval/evaluate_math.py', {'LLM': resource_llm, 'SamplingParams': SamplingParams})
    output.mkdir(parents=True)
    config = dict(protocol=f['protocol'], frozen_sha256=sha(freeze), teacher_model=cfg['model'], teacher_adapter=None,
        conditions=list(histories), history_tokens=256, max_new_tokens=8192, censored_new_tokens=2048,
        packages={n:importlib.metadata.version(n) for n in ['torch','transformers','vllm','math-verify']})
    write_json(output / 'CONFIG.PRIVATE.json', config); write_json(output / 'PHASE.json', {'phase':'model_load'})
    tick=time.monotonic();llm, tokenizer=api['load_vllm_model'](cfg['model'],None,gpu_memory_utilization=.9,tensor_parallel_size=1,max_model_len=40960,enable_thinking=True)
    model_load_seconds=time.monotonic()-tick
    config['actual_max_num_seqs']=llm.llm_engine.vllm_config.scheduler_config.max_num_seqs
    assert config['actual_max_num_seqs']==64;write_json(output/'CONFIG.PRIVATE.json',config)
    summaries={}
    for condition,prefixes in histories.items():
        assert all(api['extract_boxed_answer'](tokenizer.decode(prefix,skip_special_tokens=False)) is None for prefix in prefixes)
        inputs=[{'prompt_token_ids':row['teacher_prompt_token_ids']+prefix} for row,prefix in zip(rows,prefixes)]
        params=[SamplingParams(n=1,temperature=1.,top_p=.95,top_k=-1,min_p=0.,presence_penalty=0.,max_tokens=8192,seed=question_seed(42,'sw',r['id'])) for r in rows]
        write_json(output/'PHASE.json',{'phase':condition});tick=time.monotonic()
        outputs=llm.generate(inputs,params,use_tqdm=True);seconds=time.monotonic()-tick
        assert len(outputs)==32
        totals=dict(items=32,correct=0,censored2048_correct=0,normal_terminations=0,truncated=0,answer_extraction_failures=0,generated_tokens=0,supplied_history_tokens=sum(map(len,prefixes)),teacher_input_tokens=sum(len(x['prompt_token_ids']) for x in inputs))
        with (output/(condition+'_ITEMS.PRIVATE.jsonl')).open('x') as stream:
            for row,prefix,inp,out in zip(rows,prefixes,inputs,outputs):
                assert out.prompt_token_ids==inp['prompt_token_ids'] and len(out.outputs)==1
                sample=out.outputs[0];ids=list(sample.token_ids);assert 0<len(ids)<=8192 and sample.finish_reason in ['stop','length']
                answers=[api['extract_boxed_answer'](tokenizer.decode(prefix+part,skip_special_tokens=False)) for part in [ids,ids[:2048]]]
                correct=[bool(api['grade_answer'](ans,row['answer'])) for ans in answers]
                for key,value in [('correct',correct[0]),('censored2048_correct',correct[1]),('normal_terminations',sample.finish_reason=='stop'),('truncated',sample.finish_reason=='length'),('answer_extraction_failures',answers[0] is None),('generated_tokens',len(ids))]:totals[key]+=value
                stream.write(json.dumps(dict(id=row['id'],answer=row['answer'],prefix_token_ids=prefix,new_token_ids=ids,seed=question_seed(42,'sw',row['id']),boxed_answer=answers[0],censored2048_box=answers[1],correct=correct[0],censored2048_correct=correct[1],finish_reason=sample.finish_reason),ensure_ascii=False)+'\n')
        totals['generation_seconds']=seconds;summaries[condition]=totals;write_json(output/(condition+'_DONE.json'),totals)
    result=dict(protocol=f['protocol'],conditions=summaries,model_load_seconds=model_load_seconds,teacher_adapter=None,actual_max_num_seqs=64,formal_test=False,training_updates=0,
        scope=f['scope'],clean_budget_caveat='No_history receives8192 total response tokens; all3 history conditions receive256 supplied plus8192 new. Primary matched comparisons are between equal history conditions. Censored2048 is a prefix of these same generated continuations, not a separately sampled2048 run.')
    write_json(output/'DONE.json',result);write_json(output/'PHASE.json',{'phase':'done'});print(json.dumps(result),flush=True)


if __name__=='__main__':main()
