"""Independent4B full-budget two-update resource probe; official core unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as z:
        for block in iter(lambda:z.read(4*2**20),b''):d.update(block)
    return d.hexdigest()


def prepare(root):
    code=Path(__file__).parent;f=read(code/'sw4b_resource2_frozen_20261022.PUBLIC.json');assert shutil.disk_usage(root).free>=20*2**30
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    p=read(code/'sw4b_resource2_prepared_20261022.PUBLIC.json');assert sha(code/'sw4b_resource2_prepared_20261022.PUBLIC.json')==f['prepared_sha256']
    lock=root/p['model_lock_relative'];assert sha(lock)==p['model_lock_sha256'];m=read(lock);assert m['revision']==f['model_revision']
    model=root/m['model_relative']
    for name,info in m['files'].items():assert (model/name).stat().st_size==info['bytes'] and sha(model/name)==info['sha256'],name
    for name,digest in read(code/'seeded_cost_runtime_20261020.CONFIG.json')['code_sha256'].items():assert sha(code/name)==digest,name
    core=read(root/'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json')
    for name,digest in core['core_source_sha256'].items():assert sha(root/'COPSD-ddp-teacher-unwrapped-20261018'/name)==digest,name
    compatibility=read(code/'ja_nonefixed_runtime_20261020.CONFIG.json');assert sha(root/'envs/copsd-vllm011-20261018/lib/python3.10/site-packages/vllm/cumem_allocator.abi3.so')==compatibility['binary_sha256']
    data=root/p['data_relative'];mapping=root/p['prompt_identity_relative'];assert sha(data)==p['data_sha256'] and sha(mapping)==p['prompt_identity_sha256']
    rows=read(data);assert len(rows)==64 and [str(r['problem_id']) for r in rows]==p['selected_IDs']
    pool=root/p['parent2700_manifest_relative'];assert sha(pool)==p['parent2700_manifest_sha256'];pool=read(pool)
    assert set(p['selected_IDs'])<=set(pool['training_IDs'])
    assert not set(p['selected_IDs'])&set(read(code/'followup200_frozen_20261021.PUBLIC.json')['selected_IDs'])
    output=root/f['checkpoint_relative'];assert not output.exists()
    return f,p,rows,data,read(mapping),output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    f,p,rows,data,mapping,output=prepare(a.root)
    if a.prepare_only:
        print(json.dumps({'CPU_preflight':True,'scale':'4B','revision':f['model_revision'],'rows':64,'response_budget':2048,'updates':2,'GPU_started':False}));return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT')=='1'
    from benchmark_eval_concurrency8192 import wait_for_empty_GPU
    wait_for_empty_GPU(os.environ['CUDA_VISIBLE_DEVICES'])
    from cumem_none_ref_compat import install
    install(read(Path(__file__).with_name('ja_nonefixed_runtime_20261020.CONFIG.json'))['binary_sha256'])
    import prompt_budget_guard
    prompt_budget_guard.install()
    import prefix20_entry as prefix
    original_init=prefix.seeded.original_init;record=[]
    def init(trainer,*args,**kwargs):
        original_init(trainer,*args,**kwargs)
        assert trainer.model.config.hidden_size==2560 and trainer.model.config.num_hidden_layers==36
        assert trainer.fixed_teacher and trainer.beta==0 and trainer.jsd_token_clip==.05
        assert trainer.args.seed==trainer.args.data_seed==42 and trainer.args.learning_rate==5e-6
        assert trainer.model.peft_config['default'].r==64 and trainer.model.peft_config['default'].lora_alpha==128
        assert all(param.device.type=='cuda' for param in trainer.model.parameters())
        assert all('lora_' in name for name,param in trainer.model.named_parameters() if param.requires_grad)
        assert list(trainer.train_dataset['problem_id'])==[r['problem_id'] for r in rows]
        controller=trainer._update_batch;old_generate=controller.raw_generate
        def generated(prompts,*pos,**named):
            outputs=old_generate(prompts,*pos,**named)
            for prompt,out in zip(prompts,outputs):
                key=hashlib.sha256(prompt.encode()).hexdigest();identity=mapping[key]
                record.append({**identity,'prompt_sha256':key,'generated_tokens':len(out.outputs[0].token_ids),'finish_reason':out.outputs[0].finish_reason})
            return outputs
        controller.raw_generate=generated
        output.mkdir(parents=True,exist_ok=True)
        (output/'MODEL_SCALE_ACTUAL.json').write_text(json.dumps({'scale':'4B','model_revision':f['model_revision'],'all_HF_parameters_on_CUDA':True,'teacher_same_disabled_adapter_Base':True,'trainable_only_LoRA':True,'vllm_gpu_memory_utilization':trainer.vllm_gpu_memory_utilization,'core_loss_changed':False,'cohort_independent_from1_7B':True},indent=2)+'\n')
        from transformers import TrainerCallback
        class SaveResource(TrainerCallback):
            def on_train_end(self,args,state,control,**unused):
                assert state.global_step==2 and len(record)==64 and len({r['id'] for r in record})==64
                assert set(r['id'] for r in record)==set(p['selected_IDs'])
                (output/'RESOURCE_ROLLOUTS.PRIVATE.json').write_text(json.dumps(record,indent=2)+'\n')
        trainer.add_callback(SaveResource())
    prefix.seeded.original_init=init
    argv=['--train_language','SW','--translated_data_path',str(data),'--model_name_or_path',str(a.root/'models/Qwen3-4B'),
        '--learning_rate','5e-6','--max_grad_norm','0.1','--per_device_train_batch_size','1','--gradient_accumulation_steps','32',
        '--gradient_checkpointing','--gradient_checkpointing_kwargs','{"use_reentrant": false}','--max_steps','100','--save_steps','1','--logging_steps','1',
        '--output_dir',str(output),'--run_config',f['run_id'],'--max_completion_length','2048','--max_length','20000','--attn_implementation','sdpa','--torch_dtype','bfloat16',
        '--beta','0','--lmbda','1','--use_peft','--lora_r','64','--lora_alpha','128','--lora_target_modules','q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj',
        '--temperature','1.1','--top_p','0.95','--top_k','20','--fixed_teacher','--student_enable_thinking','--jsd_token_clip','0.05',
        '--use_vllm','--vllm_mode','colocate','--vllm_tensor_parallel_size','1','--vllm_gpu_memory_utilization','0.55','--vllm_enable_sleep_mode','true','--vllm_sync_frequency','1',
        '--report_to','tensorboard','--logging_dir',str(a.root/'logs/copsd_official_10lang/tensorboard'/f['run_id']),'--seed','42','--data_seed','42','--wandb_project','COPSD-4B-resource']
    sys.argv=[str(Path(__file__))]+argv;prefix.main()


if __name__=='__main__':main()
