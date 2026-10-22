"""One prompt-only Base control: SW question/instruction, English thinking prefix.

Reuse the frozen evaluator unchanged. Only its prompt constructor and recording
metadata are wrapped; preserve official answer grading and question-level seeds.
This is a posthoc mechanism diagnostic, not the official native-SW baseline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    code=Path(__file__).parent;freeze=code/'sw4b_prefix_only_frozen_20261022.PUBLIC.json';f=read(freeze)
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    assert read(code/'sw4b_server_SW_dev100_partial_20261022.PUBLIC.json')['trained_correct_with_whole_thinking_English_top1']==10
    if not a.prepare_only:
        state=read(a.root/'gpu-queue/queues/copsd-sw4b-server-checkpoint-dev-v1/state.json')
        assert state['phase']=='completed' and state['job']['exit_code']==0,'Primary SW/EN evaluation has priority'
    import evaluate_sw4b_server_dev8192 as original
    from transformers import AutoTokenizer
    source=a.root/'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    assert sha(source)==f['official_source_sha256']
    original_api=original.official_api
    api=original_api(source)
    native=api['LANGUAGE_PROMPTS']['SW']['think_prefix'];english=api['LANGUAGE_PROMPTS']['EN']['think_prefix']
    assert native==f['native_think_prefix'] and english==f['English_think_prefix']
    tokenizer=AutoTokenizer.from_pretrained(a.root/'models/Qwen3-4B',local_files_only=True)
    rows=read(a.root/'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json')
    assert len(rows)==100
    prior_path=a.root/'logs/copsd_official_10lang/development_vllm8192'/f['native_Base_run_id']/'ITEMS.jsonl'
    assert sha(prior_path)==f['native_Base_items_sha256']
    prior={str(r['id']):r for r in map(json.loads,prior_path.read_text().splitlines())}
    manifest=[]
    def changed(tokenizer,question,language,enable_thinking=True):
        assert language.upper()=='SW' and enable_thinking
        before=api['build_qwen3_prompt'](tokenizer,question,language,enable_thinking)
        assert before.endswith(native)
        after=before[:-len(native)]+english
        assert before[:-len(native)]==after[:-len(english)]
        return after
    for row in rows:
        before=api['build_qwen3_prompt'](tokenizer,row['problem_sw'],'SW',True);after=changed(tokenizer,row['problem_sw'],'SW')
        saved=prior[str(row['problem_id'])]
        assert before==saved['prompt'] and saved['question']==row['problem_sw'] and saved['answer']==row['Answer']
        assert saved['samples'][0]['seed']==original.question_seed(42,'sw',row['problem_id'])
        manifest.append({'id':row['problem_id'],'question_seed':saved['samples'][0]['seed'],'native_prompt_sha256':hashlib.sha256(before.encode()).hexdigest(),'English_prefix_prompt_sha256':hashlib.sha256(after.encode()).hexdigest()})
    assert sha(a.root/'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json')==f['input_sha256']
    def wrapped_api(*args,**kwargs):
        namespace=original_api(*args,**kwargs)
        namespace['build_qwen3_prompt']=changed
        return namespace
    original.official_api=wrapped_api
    original_write=original.write_json
    def recording(path,value):
        if path.name=='CONFIG.json':
            value['protocol']=f['protocol']
            value['scope']='Posthoc prefix-only mechanism diagnostic, not officialSW baseline or selectedcheckpoint.'
            value['original_evaluator_sha256']=value['evaluator_sha256']
            value['evaluator_sha256']=sha(Path(__file__))
            value['prefix_diagnostic_frozen_sha256']=sha(freeze)
            value['only_prompt_change']={'native_think_prefix':native,'replacement_English_think_prefix':english,'user_question_and_SW_instruction_unchanged':True}
            original_write(path.parent/'PROMPT_DELTA_MANIFEST.PRIVATE.json',manifest)
        return original_write(path,value)
    original.write_json=recording
    argv=['--root',str(a.root),'--language','sw','--run-id',f['run_id']]
    if a.prepare_only:argv.append('--prepare-only')
    sys.argv=[str(Path(__file__))]+argv
    original.main()
    if a.prepare_only:print('All100 prompt deltas verified: only assistant thinking suffix changed; no inference.')


if __name__=='__main__':main()
