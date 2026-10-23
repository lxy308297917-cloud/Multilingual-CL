"""Check response-view compaction on saved failing TE trajectory, checkpoint5.

No rollout generation or optimizer updates. Same response2048 and fullvocabulary.
Short prefix subcase compares original/repaired loss and LoRA gradients; full
cached trajectory checks actual4B memory with optimizer states resident.
"""
import ast
from contextlib import nullcontext
from datetime import datetime
import hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
import subprocess,time
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM
from compact_response_logits import transform_compute_loss
from diagnose_official_component_clip import extract
from mask_reduction_memory import make_mask_reduction_loss


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    root=Path(__file__).resolve().parents[3];code=Path(__file__).parent;directory=root/'logs/copsd_official_10lang/te4b_oom_replay_20261024_v1';cfg=read(directory/'PREPARED.PUBLIC.json');assert sha(directory/'INPUTS.PRIVATE.json')==cfg['inputs_sha256'];row=read(directory/'INPUTS.PRIVATE.json');source=root/'COPSD-ddp-teacher-unwrapped-20261018/multilingual_opsd_trainer.py';assert sha(source)==cfg['core_source_sha256'];train=read(code/'te4b_server20_frozen_20261024.PUBLIC.json');checkpoint=root/train['checkpoint_relative']/'checkpoint-5';assert sha(checkpoint/'adapter_model.safetensors')==cfg['checkpoint5_adapter_sha256']
    assert os.environ['GPU_QUEUE_GPU_COUNT']=='1';visible=os.environ['CUDA_VISIBLE_DEVICES'];assert visible.startswith('GPU-') and ',' not in visible
    pids=subprocess.check_output(['nvidia-smi','-i',visible,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True);assert all(int(x.strip())==os.getpid() for x in pids.splitlines() if x.strip())
    classes=ast.parse(source.read_text());cls=next(x for x in classes.body if isinstance(x,ast.ClassDef) and x.name=='OPSDTrainer');method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='compute_loss');loss=extract(source);loss.decorator_list=[];ns={'torch':torch,'F':F,'nullcontext':nullcontext,'is_peft_model':lambda m:True,'empty_cache':torch.cuda.empty_cache};exec(compile(ast.Module(body=[method,loss],type_ignores=[]),'pinned_core','exec'),ns);original=ns['compute_loss'];compact=transform_compute_loss(ast.parse(ast.unparse(method)),ns)
    torch.use_deterministic_algorithms(True)
    tick=time.monotonic();model=AutoModelForCausalLM.from_pretrained(root/'models/Qwen3-4B',torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda');model=PeftModel.from_pretrained(model,checkpoint,is_trainable=True).train();model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();optimizer= torch.load(checkpoint/'optimizer.pt',map_location='cuda',weights_only=True);assert {int(x['step']) for x in optimizer['state'].values()}=={5}
    dummy=SimpleNamespace(use_thinking_machines_loss=False,use_ema_teacher=False,fixed_teacher=True,accelerator=SimpleNamespace(num_processes=1,unwrap_model=lambda m:m),beta=0,temperature=1.1,top_k_loss=0,jsd_token_clip=.05,generalized_jsd_loss=make_mask_reduction_loss(ns['generalized_jsd_loss'],{}))
    def inputs(response):
        student=torch.tensor([row['student']+response],device='cuda');teacher=torch.tensor([row['teacher']+response],device='cuda');labels=student.clone();labels[:,:len(row['student'])]=-100;labels[student==row['pad_token_id']]=-100
        return {'student_prompt_length':len(row['student']),'teacher_prompt_length':len(row['teacher']),'student_input_ids':student,'teacher_input_ids':teacher,'student_attention_mask':student.ne(row['pad_token_id']).long(),'teacher_attention_mask':teacher.ne(row['pad_token_id']).long(),'labels':labels}
    small=inputs(row['completion'][:16]);baseline={};losses=[];maxdiff=0.;allclose=True
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    for index,fn in enumerate([original,compact]):
        model.zero_grad(set_to_none=True);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=fn(dummy,model,small)
        value.backward();losses.append(float(value.detach()))
        for n,p in model.named_parameters():
            if not p.requires_grad:assert p.grad is None;continue
            grad=p.grad.detach().cpu()
            if index==0:baseline[n]=grad.clone()
            else:maxdiff=max(maxdiff,float((grad-baseline[n]).abs().max()));allclose=allclose and torch.allclose(grad,baseline[n],atol=1e-6,rtol=2e-5)
        del value
    assert abs(losses[0]-losses[1])<=1e-6 and allclose,(losses,maxdiff)
    del baseline,small;torch.use_deterministic_algorithms(False);model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();full=inputs(row['completion'])
    with torch.autocast('cuda',dtype=torch.bfloat16):value=compact(dummy,model,full)
    assert torch.isfinite(value);value.backward();torch.cuda.synchronize();assert all(p.grad is None for p in model.parameters() if not p.requires_grad);assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    peak=torch.cuda.max_memory_allocated();free,total=torch.cuda.mem_get_info();reserved_peak=torch.cuda.max_memory_reserved();assert total-reserved_peak>=512*2**20,'Insufficient measured process headroom'
    out={'completed_at':datetime.now().isoformat(),'short_comparison_deterministic_kernels':True,'full_memory_check_uses_normal_nondeterministic_kernel_policy':True,'short_comparison_CPU_CUDA_RNG_restored':True,'actual_LoRA_dropout':model.peft_config['default'].lora_dropout,'original_short_loss':losses[0],'compacted_short_loss':losses[1],'short_LoRA_gradients_allclose':bool(allclose),'short_max_abs_LoRA_gradient_difference':maxdiff,'full_cached_response_tokens':len(row['completion']),'student_prompt_tokens':len(row['student']),'teacher_prompt_tokens':len(row['teacher']),'full_compacted_loss':float(value.detach()),'all_gradients_finite':True,'frozen_Base_has_no_gradients':True,'checkpoint5_optimizer_states_resident':True,'peak_allocated_GPU_bytes':peak,'peak_reserved_GPU_bytes':reserved_peak,'total_GPU_bytes':total,'process_headroom_bytes':total-reserved_peak,'wall_seconds':time.monotonic()-tick,'patch_sha256':sha(code/'compact_response_logits.py'),'script_sha256':sha(Path(__file__)),'prepared_sha256':sha(directory/'PREPARED.PUBLIC.json'),'no_optimizer_updates':True,'no_generation':True,'scope':cfg['scope']+' Full2048 actualcase checked; short16 only equivalence fixture, not replacement trainingbudget. Does not guarantee all future prompt lengths or bitwise resumed sampling.'}
    with (code/'te4b_oom_compaction_CUDA_20261024.PUBLIC.json').open('x') as stream:stream.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':main()
