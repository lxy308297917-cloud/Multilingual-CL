"""Offline real-weight/recorded-trajectory diagnosis; no training or GPU use.

Retokenizes saved text and uses float32 CPU inference. This is a controlled
mechanism replay, not bitwise replay of the original BF16 GPU training.
"""
import argparse
import ast
import json
import time
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
import torch
import torch.nn.functional as F
from diagnose_official_component_clip import extract


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--generations',type=Path,required=True)
    p.add_argument('--official',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=2)
    p.add_argument('--response-tokens',type=int,default=64)
    p.add_argument('--parameter-gradients',action='store_true')
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(8)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    fn=extract(a.official);fn.decorator_list=[]
    ns={'torch':torch,'F':F}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.official),'exec'),ns)
    official_loss=ns['generalized_jsd_loss']
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    rows=json.loads(a.generations.read_text())['generations'][:a.samples]
    assert len(rows)==a.samples
    cases=[]
    for r in rows:
        assert r['student_completion']==r['teacher_evaluated_completion']
        s=tokenizer.encode(r['student_prompt'],add_special_tokens=False)
        t=tokenizer.encode(r['teacher_prompt'],add_special_tokens=False)
        trajectory=tokenizer.encode(r['student_completion'],add_special_tokens=False)[:a.response_tokens]
        assert len(trajectory)==a.response_tokens and tokenizer.pad_token_id not in trajectory
        cases.append((s,t,trajectory))
    started=time.monotonic()
    base=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
            torch_dtype=torch.float32,attn_implementation='sdpa').eval()
    def forward(model,prompt,trajectory):
        ids=torch.tensor([prompt+trajectory])
        with torch.inference_mode():
            # Last prefix token predicts response[0]; response[-1]'s output is unused.
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),
                         logits_to_keep=len(trajectory)+1,use_cache=False).logits[:,:-1].clone()
        assert logits.shape[1]==len(trajectory)
        return logits
    teacher=[forward(base,t,z) for s,t,z in cases]
    model=PeftModel.from_pretrained(base,a.checkpoint,is_trainable=a.parameter_gradients).eval()
    module=ast.parse(a.official.read_text())
    klass=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='OPSDTrainer')
    compute=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='compute_loss')
    compute.decorator_list=[]
    ns.update(empty_cache=lambda:None,is_peft_model=lambda m:isinstance(m,PeftModel),
              nullcontext=nullcontext)
    exec(compile(ast.Module(body=[compute],type_ignores=[]),str(a.official),'exec'),ns)
    results=[]
    for i,(s,t,z) in enumerate(cases):
        student=forward(model,s,z)
        with model.disable_adapter():
            unchanged=forward(model,t,z)
        assert torch.equal(unchanged,teacher[i]),'Disabled-adapter teacher differs from original Base'
        actual_loss_inputs={}
        parameter_check=None
        if i==0:
            def capture(**kwargs):
                actual_loss_inputs.update(kwargs)
                return official_loss(**kwargs)
            trainer=SimpleNamespace(use_thinking_machines_loss=False,use_ema_teacher=False,
                fixed_teacher=True,accelerator=SimpleNamespace(unwrap_model=lambda m:m),
                generalized_jsd_loss=capture,beta=0,temperature=1.1,top_k_loss=None,jsd_token_clip=.05)
            student_ids=torch.tensor([s+z]);teacher_ids=torch.tensor([t+z])
            labels=student_ids.clone();labels[:,:len(s)]=-100
            inputs={'student_prompt_length':len(s),'teacher_prompt_length':len(t),
                'labels':labels,'student_input_ids':student_ids,'teacher_input_ids':teacher_ids,
                'student_attention_mask':torch.ones_like(student_ids),
                'teacher_attention_mask':torch.ones_like(teacher_ids)}
            actual=ns['compute_loss'](trainer,model,inputs)
            assert torch.equal(actual_loss_inputs['labels'],torch.tensor([z]))
            assert not actual_loss_inputs['teacher_logits'].requires_grad
            assert torch.allclose(actual_loss_inputs['student_logits'],student,atol=1e-5,rtol=1e-5)
            assert torch.allclose(actual_loss_inputs['teacher_logits'],teacher[i],atol=1e-5,rtol=1e-5)
            print('Actual pinned compute_loss: response masks, causal shift and disabled-adapter teacher match independent reconstruction.',flush=True)
            if a.parameter_gradients:
                params=[p for p in model.parameters() if p.requires_grad]
                gc=torch.autograd.grad(actual,params,retain_graph=True)
                values={**actual_loss_inputs,'token_clip':None}
                lu=official_loss(**values)
                gu=torch.autograd.grad(lu,params)
                dot=sum(float((x*y).sum(dtype=torch.float64)) for x,y in zip(gc,gu))
                nc=sum(float(x.square().sum(dtype=torch.float64)) for x in gc)
                nu=sum(float(x.square().sum(dtype=torch.float64)) for x in gu)
                parameter_check={'trainable_parameter_elements':sum(p.numel() for p in params),
                    'all_trainable_parameters_are_LoRA':all('lora_' in n for n,p in model.named_parameters() if p.requires_grad),
                    'clipped_vs_unclipped_parameter_gradient_dot':dot,
                    'clipped_vs_unclipped_parameter_gradient_cosine':dot/(nc*nu)**.5,
                    'opposite_true_KL_descent_direction_in_parameter_space':dot<0,
                    'scope':'One fixed real recorded prefix, float32 CPU eval-mode LoRA gradient; not AdamW update, online training trajectory or downstream causal outcome.'}
                print('LORA_PARAMETER_GRADIENTS',json.dumps(parameter_check),flush=True)
                del gc,gu
        gradients=[];losses=[]
        for clip in [None,.05]:
            logits=student.detach().clone().requires_grad_(True)
            value=official_loss(student_logits=logits,teacher_logits=teacher[i],
                    labels=torch.tensor([z]),beta=0,temperature=1.1,top_k=None,token_clip=clip)
            gradient,=torch.autograd.grad(value,logits)
            gradients.append(gradient);losses.append(float(value.detach()))
        full,clipped=gradients
        top=teacher[i].argmax(-1,keepdim=True)
        gfull=full.gather(-1,top).squeeze(-1)
        gclip=clipped.gather(-1,top).squeeze(-1)
        dot=(full*clipped).sum(-1)
        cosine=dot/(full.norm(dim=-1)*clipped.norm(dim=-1)).clamp_min(1e-30)
        components=F.kl_div(F.log_softmax(student/1.1,-1),
                           F.log_softmax(teacher[i]/1.1,-1),log_target=True,reduction='none')
        results.append({'sample_index':i,'response_positions':len(z),
            'teacher_disabled_adapter_logits_equal_original_Base':True,
            'unclipped_KL':losses[0],'official_clipped_loss':losses[1],
            'teacher_argmax_component_clipped_positions':int((components.gather(-1,top)>.05).sum()),
            'teacher_argmax_attraction_reversed_positions':int(((gfull<0)&(gclip>0)).sum()),
            'positions_gradient_opposes_unclipped_KL':int((dot<0).sum()),
            'mean_per_position_gradient_cosine':float(cosine.mean()),
            'overall_gradient_cosine':float((full*clipped).sum()/(full.norm()*clipped.norm()).clamp_min(1e-30)),
            'student_teacher_response_positions_equal':True})
        if i==0:
            results[-1]['actual_pinned_compute_loss_response_mask_and_shift_verified']=True
            results[-1]['actual_pinned_compute_loss_teacher_no_grad_and_Base_logits_verified']=True
            if parameter_check is not None:results[-1]['LoRA_parameter_gradient_check']=parameter_check
        print(json.dumps(results[-1]),flush=True)
    out={'experiment':'real_Qwen3_1_7B_saved_training_trajectory_CPU_gradient_replay',
         'checkpoint_global_step':json.loads((a.checkpoint/'trainer_state.json').read_text())['global_step'],
         'recorded_generation_global_step':json.loads(a.generations.read_text())['step'],
         'dtype':'float32 CPU inference/autograd','cases':results,
         'elapsed_seconds':time.monotonic()-started,
         'scope':'Real model weights and saved training prompts/continuations, deterministic first records and response prefixes. '
                 'Text retokenized; eval-mode CPU float32 replay, not original BF16 GPU logits or full trajectories. '
                 'Direct mechanism evidence only, not proof of the cause of downstream regression or successful fix. '
                 'No model update, new rollout, benchmark selection or GPU allocation.'}
    a.output.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
