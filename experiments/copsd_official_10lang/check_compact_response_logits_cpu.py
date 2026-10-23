"""Check unchanged full causal forward/loss/LoRA gradients on tiny Qwen3."""
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from peft import LoraConfig,get_peft_model
from transformers import Qwen3Config,Qwen3ForCausalLM
from compact_response_logits import transform_compute_loss
from diagnose_official_component_clip import extract
from mask_reduction_memory import make_mask_reduction_loss


def main():
    root=Path(__file__).resolve().parents[3];code=Path(__file__).parent;source=root/'COPSD-ddp-teacher-unwrapped-20261018/multilingual_opsd_trainer.py';text=source.read_text();cls=next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef) and x.name=='OPSDTrainer');method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='compute_loss');loss_ast=extract(source);loss_ast.decorator_list=[]
    ns={'torch':torch,'F':F,'nullcontext':nullcontext,'is_peft_model':lambda model:True,'empty_cache':lambda:None};exec(compile(ast.Module(body=[method,loss_ast],type_ignores=[]),'pinned_core','exec'),ns);original=ns['compute_loss'];compact=transform_compute_loss(ast.parse(ast.unparse(method)),ns)
    torch.manual_seed(20261024);results=[]
    for dtype in [torch.float32,torch.bfloat16]:
        model=get_peft_model(Qwen3ForCausalLM(Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,use_cache=False,attention_dropout=0)),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).to(dtype).train()
        for n,p in model.named_parameters():
            if 'lora_B' in n:torch.nn.init.normal_(p,std=.1)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        student=torch.randint(1,90,(1,12));teacher=torch.cat([torch.randint(1,90,(1,9)),student[:,5:]],dim=1)
        for partial in [False,True]:
            labels=student.clone();labels[:,:5]=-100
            if partial:labels[:,-2:]=-100
            inputs={'student_prompt_length':5,'teacher_prompt_length':9,'student_input_ids':student,'teacher_input_ids':teacher,'student_attention_mask':torch.ones_like(student),'teacher_attention_mask':torch.ones_like(teacher),'labels':labels}
            dummy=SimpleNamespace(use_thinking_machines_loss=False,use_ema_teacher=False,fixed_teacher=True,accelerator=SimpleNamespace(num_processes=1,unwrap_model=lambda m:m),beta=0,temperature=1.1,top_k_loss=0,jsd_token_clip=.05,generalized_jsd_loss=make_mask_reduction_loss(ns['generalized_jsd_loss'],{}))
            losses=[];gradients=[]
            for fn in [original,compact]:
                model.zero_grad(set_to_none=True);loss=fn(dummy,model,inputs);loss.backward();losses.append(loss.detach());gradients.append({n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad});assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert torch.equal(*losses) and all(torch.equal(gradients[0][n],gradients[1][n]) for n in gradients[0])
            results.append({'dtype':str(dtype),'partial_mask':partial,'loss_bitwise_equal':True,'all_LoRA_gradients_bitwise_equal':True,'frozen_Base_has_no_gradients':True})
    view=torch.randn(1,23,11)[:,9:-1,:];assert view.is_contiguous();clone=view.clone();assert torch.equal(view,clone) and clone.untyped_storage().nbytes()<view.untyped_storage().nbytes()
    report={'all_passed':True,'cases':results,'contiguous_view_retains_full_storage_fixture_verified':True,'core_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'patch_sha256':hashlib.sha256((code/'compact_response_logits.py').read_bytes()).hexdigest(),'scope':'CPU tinyQwen3 fullforward/teacheradapterdisable/masks/gradientcheckpointing and LoRAgradient identity. No full4B CUDA memory guarantee or efficacy claim.'}
    p=code/'compact_response_logits_CPU_20261024.PUBLIC.json'
    with p.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
