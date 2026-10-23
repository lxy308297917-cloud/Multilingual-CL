"""Bind the actual first failing trajectory to its original teacher/student prompts."""
from pathlib import Path
from datetime import datetime
import hashlib,json,sys
from transformers import AutoTokenizer


def main():
    root=Path(__file__).resolve().parents[3];code=Path(__file__).parent;read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();f=read(code/'te4b_server20_frozen_20261024.PUBLIC.json');folder=root/f['checkpoint_relative'];rollout=[json.loads(x) for x in (folder/'SERVER_ROLLOUTS.PRIVATE.jsonl').read_text().splitlines()];exposure=read(folder/'EXPOSURE_PROMPTS.PRIVATE.json');index=228;record=rollout[index];assert record['global_step']==7 and len(record['completion_ids'])==2048
    sys.path.insert(0,str(root/'COPSD-ddp-teacher-unwrapped-20261018'));from multilingual_data_collator import MultilingualSelfDistillationDataCollator;from language_config import LANGUAGE_CONFIG
    tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen3-4B',local_files_only=True);collator=MultilingualSelfDistillationDataCollator(tokenizer,max_length=20000,reason_first=False,student_enable_thinking=True);cfg=LANGUAGE_CONFIG['TE'];manifest=read(root/f['training_manifest_relative']);data=read(root/manifest['data_relative']);matched=[]
    for row in data:
        user=cfg['labels']['problem_target']+': '+row['problem_te']+'\n\n'+cfg['student_instruction'];text=tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=False);text=collator._append_qwen3_think_prefix(text,'TE');text=text.replace(tokenizer.pad_token,'') if tokenizer.pad_token else text
        if hashlib.sha256(text.encode()).hexdigest()==record['student_prompt_sha256']:matched.append(row)
    assert len(matched)==1;batch=collator([{**matched[0],'target_lang':'TE'}])
    for key in ['student_prompts','teacher_prompts']:assert hashlib.sha256(batch[key].numpy().tobytes()).hexdigest()==exposure[index][key+'_sha256']
    output=root/'logs/copsd_official_10lang/te4b_oom_replay_20261024_v1';output.mkdir(exist_ok=False);inputs={'student':batch['student_prompts'][0].tolist(),'teacher':batch['teacher_prompts'][0].tolist(),'completion':record['completion_ids'],'pad_token_id':tokenizer.pad_token_id};(output/'INPUTS.PRIVATE.json').write_text(json.dumps(inputs)+'\n')
    report={'prepared_at':datetime.now().isoformat(),'index_zero_based':index,'original_global_step':7,'response_tokens':2048,'student_prompt_tokens':len(inputs['student']),'teacher_prompt_tokens':len(inputs['teacher']),'inputs_sha256':sha(output/'INPUTS.PRIVATE.json'),'checkpoint5_adapter_sha256':sha(folder/'checkpoint-5/adapter_model.safetensors'),'core_source_sha256':sha(root/'COPSD-ddp-teacher-unwrapped-20261018/multilingual_opsd_trainer.py'),'matched_actual_prompt_tensor_hashes':True,'scope':'Exact failing trajectory and promptshapes. GPU validation uses available checkpoint5, not loststep7 weights; memory/engineering only, no generation or optimizer updates.'};(output/'PREPARED.PUBLIC.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
