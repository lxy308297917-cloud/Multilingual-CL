"""Prepare a fixed4B resource probe with64 unchanged SW rows including long contexts."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''):d.update(block)
    return d.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();h=a.root;code=Path(__file__).parent
    model=h/'models/Qwen3-4B';revision='1cfa9a7208912126459214e8b04321603b3df60c'
    index=read(model/'model.safetensors.index.json');weights=sorted(set(index['weight_map'].values()));files={}
    for name in ['config.json','model.safetensors.index.json',*weights,'tokenizer.json','tokenizer_config.json','generation_config.json']:
        path=model/name;assert path.stat().st_size>0
        if name=='config.json' or name in weights:
            metadata=(model/'.cache/huggingface/download'/(name+'.metadata')).read_text().splitlines();assert metadata[0]==revision
            digest=sha(path)
            if name in weights and len(metadata[1])==64:assert digest==metadata[1]
        else:digest=sha(path)
        files[name]={'bytes':path.stat().st_size,'sha256':digest}
    for name in ['tokenizer.json','tokenizer_config.json']:assert sha(model/name)==sha(h/'models/Qwen3-1.7B'/name)
    source=h/'data/copsd_official_20261015/sw_lr20_pool_20261022_v1/MANIFEST.PRIVATE.json';m=read(source);pool=h/m['data_relative'];assert sha(pool)==m['data_sha256']
    lookup={str(r['problem_id']):r for r in read(pool)};ids=m['first640_expected_IDs'];assert len(ids)==len(set(ids))==640
    sys.path.insert(0,str(h/'COPSD-official-20261015'))
    from multilingual_data_collator import MultilingualSelfDistillationDataCollator
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    collator=MultilingualSelfDistillationDataCollator(tokenizer=tokenizer,max_length=20000,reason_first=False,student_enable_thinking=True,include_problem_en=True,include_reference_solution_en=True)
    lengths={};prompt_map={}
    for identifier in ids:
        batch=collator([{**lookup[identifier],'target_lang':'SW'}]);student=batch['student_prompts'][0].tolist();teacher=batch['teacher_prompts'][0].tolist()
        lengths[identifier]={'student_prompt_tokens':len(student),'teacher_prompt_tokens':len(teacher)}
        prompt=tokenizer.decode(student,skip_special_tokens=False)
        prompt_map[identifier]=hashlib.sha256(prompt.encode()).hexdigest()
    stress=[max(ids,key=lambda x:lengths[x]['teacher_prompt_tokens']),max(ids,key=lambda x:lengths[x]['student_prompt_tokens'])]
    selected=list(dict.fromkeys(ids[:62]+stress))
    for identifier in ids:
        if len(selected)>=64:break
        if identifier not in selected:selected.append(identifier)
    assert len(selected)==64 and set(stress)<=set(selected)
    assert len({prompt_map[x] for x in selected})==64
    assert max(v['teacher_prompt_tokens'] for v in lengths.values())+2048<=20000
    assert max(v['student_prompt_tokens'] for v in lengths.values())+2048<=20000
    directory=h/'data/copsd_official_20261015/sw4b_resource2_20261022_v1';directory.mkdir(exist_ok=False)
    data=directory/'SW64.PRIVATE.json';data.write_text(json.dumps([lookup[x] for x in selected],ensure_ascii=False,indent=2)+'\n')
    mapping=directory/'PROMPT_IDENTITIES.PRIVATE.json';mapping.write_text(json.dumps({prompt_map[x]:{'id':x,**lengths[x]} for x in selected},indent=2)+'\n')
    lock=h/'logs/copsd_official_10lang/qwen3_4b_resource_model_20261022.PRIVATE.json'
    with lock.open('x') as z:json.dump({'revision':revision,'model_relative':'models/Qwen3-4B','files':files},z,indent=2);z.write('\n')
    result={'observed_at':datetime.datetime.now().isoformat(),'model_revision':revision,'model_lock_relative':str(lock.relative_to(h)),'model_lock_sha256':sha(lock),
        'data_relative':str(data.relative_to(h)),'data_sha256':sha(data),'prompt_identity_relative':str(mapping.relative_to(h)),'prompt_identity_sha256':sha(mapping),
        'parent2700_manifest_sha256':sha(source),'parent2700_manifest_relative':str(source.relative_to(h)),'selected_IDs':selected,'stress_IDs':stress,
        'max_future640_student_prompt_tokens':max(v['student_prompt_tokens'] for v in lengths.values()),'max_future640_teacher_prompt_tokens':max(v['teacher_prompt_tokens'] for v in lengths.values()),
        'max_smoke64_student_prompt_tokens':max(lengths[x]['student_prompt_tokens'] for x in selected),'max_smoke64_teacher_prompt_tokens':max(lengths[x]['teacher_prompt_tokens'] for x in selected),
        'scope':'Engineering resource probe only:62 initial planned exposure IDs plus maximumteacher/student-prefix cases, filled deterministically to64 unique rows. Selection uses length, not correctness. All rows fromheldout-safe2700 pool, no target-answer synthesis. Any future formal4B run starts fresh on2700 pool; never continues these debug weights or mixes with1.7B rankings.'}
    path=code/'sw4b_resource2_prepared_20261022.PUBLIC.json'
    with path.open('x') as z:json.dump(result,z,indent=2);z.write('\n')
    print(json.dumps({k:result[k] for k in ['max_future640_student_prompt_tokens','max_future640_teacher_prompt_tokens','max_smoke64_student_prompt_tokens','max_smoke64_teacher_prompt_tokens']}))


if __name__=='__main__':main()
