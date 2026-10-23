"""Replay saved first-update4B BN trajectories; no generation or parameter updates.

Select14 fixed causal response positions per example, preserving full preceding
history. First update uses originalBase student and disabled-adapter teacher.
Outputs diagnose logits gradients, not full LoRA gradients or downstream efficacy.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return [json.loads(s) for s in f]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['prepare','check','run'],required=True);a=p.parse_args()
    root=a.root;code=Path(__file__).parent;source=root/'COPSD-official-20261015'
    folder=root/'logs/copsd_official_10lang/bn4b_initial_teacher32_replay_20261023_v1'; prepared=folder/'PREPARED.PUBLIC.json';inputs=folder/'INPUTS.PRIVATE.json'
    f=read(code/'bn4b_server20_frozen_20261023.PUBLIC.json');original=root/f['checkpoint_relative'];rp=original/'SERVER_ROLLOUTS.PRIVATE.jsonl';ep=original/'EXPOSURE_PROMPTS.PRIVATE.json'
    manifest=root/f['training_manifest_relative'];assert sha(manifest)==f['training_manifest_sha256'];m=read(manifest);data=root/m['data_relative'];assert sha(data)==m['data_sha256']
    rollout=rows(rp);assert len(rollout)==640 and all(r['global_step']==0 for r in rollout[:32])
    other=read(code/'bn4b_unclip20_frozen_20261023.PUBLIC.json');assert rows(root/other['checkpoint_relative']/'SERVER_ROLLOUTS.PRIVATE.jsonl')[:32]==rollout[:32]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen3-4B',local_files_only=True)
    if a.mode=='prepare':
        assert not folder.exists();sys.path.insert(0,str(source))
        from multilingual_data_collator import MultilingualSelfDistillationDataCollator
        collator=MultilingualSelfDistillationDataCollator(tokenizer,max_length=20000,reason_first=False,student_enable_thinking=True,include_problem_en=True,include_reference_solution_en=True)
        targets={r['student_prompt_sha256'] for r in rollout[:32]};matched={}
        for row in read(data):
            batch=collator([{**row,'target_lang':'BN'}]);sp=batch['student_prompts'][0].tolist()
            text=tokenizer.decode(sp,skip_special_tokens=False);text=text.replace(tokenizer.pad_token,'') if tokenizer.pad_token else text
            digest=hashlib.sha256(text.encode()).hexdigest()
            if digest in targets:
                assert digest not in matched
                matched[digest]={'id':row['problem_id'],'student':sp,'teacher':batch['teacher_prompts'][0].tolist(),'student_tensor_sha256':hashlib.sha256(batch['student_prompts'].numpy().tobytes()).hexdigest(),'teacher_tensor_sha256':hashlib.sha256(batch['teacher_prompts'].numpy().tobytes()).hexdigest()}
        assert set(matched)==targets
        exposures=read(ep);rr=[]
        for i,r in enumerate(rollout[:32]):
            row=matched[r['student_prompt_sha256']]
            assert row['student_tensor_sha256']==exposures[i]['student_prompts_sha256'] and row['teacher_tensor_sha256']==exposures[i]['teacher_prompts_sha256']
            rr.append({**row,'completion':r['completion_ids']})
        folder.mkdir(parents=True);inputs.write_text(json.dumps(rr)+'\n')
        cfg={'protocol':'project_BN4B_actual_initial32_teacher_logits_replay_v1','script_sha256':sha(Path(__file__)),'inputs_sha256':sha(inputs),'original_rollouts_sha256':sha(rp),'exposure_prompts_sha256':sha(ep),'data_sha256':sha(data),'official_source_sha256':{n:sha(source/n) for n in ['multilingual_data_collator.py','multilingual_opsd_trainer.py','language_config.py']},'positions':[0,1,7,31,63,127,255,511,767,1023,1279,1535,1791,2047],'temperature':1.1,'items':32,'no_parameter_updates':True,'no_generation':True,'first_update_clipped_unclipped_trajectories_identical':True,'matched_actual_student_and_teacher_prompt_tensor_hashes':True,'scope':'OriginalBase beforefirstoptimizer update. Selected positions withfullcausal histories; outputlogit gradients, not LoRA gradients, completepositions, laterupdates or efficacy.'}
        prepared.write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps({'prepared':True,'rows':32,'max_teacher_sequence':max(len(r['teacher'])+len(r['completion']) for r in rr)}));return
    cfg=read(prepared);assert cfg['script_sha256']==sha(Path(__file__)) and cfg['inputs_sha256']==sha(inputs)
    assert cfg['original_rollouts_sha256']==sha(rp) and cfg['exposure_prompts_sha256']==sha(ep) and cfg['data_sha256']==sha(data)
    for n,digest in cfg['official_source_sha256'].items():assert sha(source/n)==digest
    lock=read(code/'bn4b_unclip20_eval_frozen_20261023.PUBLIC.json');lp=root/lock['model_lock_relative'];assert sha(lp)==lock['model_lock_sha256'];identity=read(lp)
    for n,info in identity['files'].items():assert sha(root/identity['model_relative']/n)==info['sha256']
    records=read(inputs);assert len(records)==32
    if a.mode=='check':print('Prepared32 exacttraininginputs andpinnedmodel/sources verified; noGPUloaded');return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT')=='1';visible=os.environ.get('CUDA_VISIBLE_DEVICES','');assert visible.startswith('GPU-') and ',' not in visible
    import subprocess
    processes=subprocess.run(['nvidia-smi','-i',visible,'--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=15)
    assert all(s.strip().isdigit() and int(s)==os.getpid() for s in processes.stdout.splitlines() if s.strip())
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    cls=next(n for n in ast.parse((source/'multilingual_opsd_trainer.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='OPSDTrainer')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generalized_jsd_loss');fn.decorator_list=[]
    namespace={'torch':torch,'F':F};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source/'multilingual_opsd_trainer.py'),'exec'),namespace);loss=namespace['generalized_jsd_loss']
    assert not (folder/'DONE.PUBLIC.json').exists();tick=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(root/'models/Qwen3-4B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
    output=[]
    for index,row in enumerate(records):
        valid=[i for i in cfg['positions'] if i<len(row['completion']) and row['completion'][i]!=tokenizer.pad_token_id]
        def forward(prefix):
            ids=torch.tensor([prefix+row['completion']],device='cuda');mask=ids.ne(tokenizer.pad_token_id).long();positions=torch.tensor([len(prefix)-1+i for i in valid],device='cuda')
            with torch.no_grad():return model(input_ids=ids,attention_mask=mask,logits_to_keep=positions,use_cache=False).logits.detach()
        student=forward(row['student']);teacher=forward(row['teacher']);assert student.shape==teacher.shape and student.shape[1]==len(valid)
        label=torch.tensor([[row['completion'][i] for i in valid]],device='cuda')
        details={'index':index,'positions':valid,'valid_positions':len(valid),'dtypes':{},'sampled_teacher_minus_student_logprob':[]}
        for dtype,name in [(torch.bfloat16,'original_BF16'),(torch.float32,'FP32_from_BF16_logits')]:
            s=student.to(dtype).detach().requires_grad_(True);t=teacher.to(dtype)
            clipped=loss(s,t,labels=label,beta=0,temperature=1.1,token_clip=.05);gc=torch.autograd.grad(clipped,s)[0].float()
            plain=loss(s,t,labels=label,beta=0,temperature=1.1,token_clip=None);gu=torch.autograd.grad(plain,s)[0].float()
            assert all(torch.isfinite(x).all() for x in [clipped,plain,gc,gu])
            top=t.argmax(-1,keepdim=True);cg=gc.gather(-1,top).squeeze(-1);ug=gu.gather(-1,top).squeeze(-1)
            cosine=F.cosine_similarity(gc.flatten(-2),gu.flatten(-2),dim=-1).item()
            details['dtypes'][name]={'clipped_loss':float(clipped),'unclipped_loss':float(plain),'gradient_cosine_all_selected_logits':cosine,'teacher_top_attraction_reversed':int(((ug<0)&(cg>0)).sum()),'teacher_top_unclipped_attraction_positions':int((ug<0).sum())}
            if name=='original_BF16':
                sp=F.log_softmax(s.detach()/1.1,-1);tp=F.log_softmax(t/1.1,-1)
                details['sampled_teacher_minus_student_logprob']=(tp.gather(-1,label.unsqueeze(-1))-sp.gather(-1,label.unsqueeze(-1))).flatten().float().cpu().tolist()
        output.append(details);(folder/'POSITIONS.PRIVATE.json').write_text(json.dumps(output,indent=2)+'\n');print('REPLAY_ITEM',index,len(valid),flush=True)
        del student,teacher,s,t,gc,gu;torch.cuda.empty_cache()
    summary={'protocol':cfg['protocol'],'rows':len(output),'selected_positions':sum(x['valid_positions'] for x in output),'dtypes':{},'wall_seconds':time.monotonic()-tick,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'prepared_sha256':sha(prepared),'script_sha256':sha(Path(__file__)),'scope':cfg['scope'],'no_parameter_updates':True,'no_generation':True}
    for name in output[0]['dtypes']:
        summary['dtypes'][name]={'teacher_top_attraction_reversed':sum(x['dtypes'][name]['teacher_top_attraction_reversed'] for x in output),'teacher_top_unclipped_attraction_positions':sum(x['dtypes'][name]['teacher_top_unclipped_attraction_positions'] for x in output),'examples_with_negative_clipped_unclipped_logit_gradient_cosine':sum(x['dtypes'][name]['gradient_cosine_all_selected_logits']<0 for x in output)}
    (folder/'DONE.PUBLIC.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
