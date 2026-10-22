"""Independent fresh4B SW20 pilot; requires completed full-budget resource gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import MethodType


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''):h.update(block)
    return h.hexdigest()


def prepare(root, require_gate):
    code=Path(__file__).parent;f=read(code/'sw4b_pilot20_frozen_20261022.PUBLIC.json')
    assert shutil.disk_usage(root).free>=20*2**30
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    prepared=read(code/'sw4b_resource2_prepared_20261022.PUBLIC.json')
    lock=root/prepared['model_lock_relative'];assert sha(lock)==prepared['model_lock_sha256']
    model=read(lock);assert model['revision']==f['model_revision']
    for name,v in model['files'].items():assert sha(root/model['model_relative']/name)==v['sha256'],name
    old=read(code/'sw_lr20_frozen_20261022.PUBLIC.json')
    manifest=root/old['training_manifest_relative'];assert sha(manifest)==old['training_manifest_sha256']
    m=read(manifest);data=root/m['data_relative'];assert sha(data)==m['data_sha256']
    rows=read(data);assert len(rows)==2700 and [str(r['problem_id']) for r in rows]==m['training_IDs']
    original=root/m['original_source_relative'];assert sha(original)==m['original_source_sha256']
    keep=set(m['training_IDs']);assert rows==[r for r in read(original) if str(r['problem_id']) in keep]
    assert not keep & set(read(code/'followup200_frozen_20261021.PUBLIC.json')['selected_IDs'])
    core=read(root/'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json')
    for name,digest in core['core_source_sha256'].items():assert sha(root/'COPSD-ddp-teacher-unwrapped-20261018'/name)==digest,name
    for name,digest in read(code/'seeded_cost_runtime_20261020.CONFIG.json')['code_sha256'].items():assert sha(code/name)==digest,name
    output=root/f['checkpoint_relative'];assert not output.exists()
    gate=code/'sw4b_resource2_v3_completed_20261022.PUBLIC.json'
    if require_gate:
        g=read(gate);assert g['engineering_completed'] and g['completed_updates']==2
        assert g['sampled_physical_headroom_MiB']>=512,'Insufficient sampled physical memory margin'
        assert g['generated_tokens']>0 and g['responses_at2048']>0
        assert g['model_revision']==f['model_revision'] and g['frozen_Base_bitwise_unchanged']
    return f,rows,data,output,gate.exists()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    f,rows,data,output,gate_present=prepare(a.root,not a.prepare_only);code=Path(__file__).parent
    if a.prepare_only:
        print(json.dumps({'CPU_preflight':True,'scale':'4B','fresh_Base':True,'rows':2700,'updates':20,'resource_gate_present':gate_present,'GPU_started':False}));return
    assert os.environ['GPU_QUEUE_GPU_COUNT']=='1'
    from benchmark_eval_concurrency8192 import wait_for_empty_GPU
    wait_for_empty_GPU(os.environ['CUDA_VISIBLE_DEVICES'])
    from cumem_none_ref_compat import install
    install(read(code/'ja_nonefixed_runtime_20261020.CONFIG.json')['binary_sha256'])
    import prompt_budget_guard
    prompt_budget_guard.install()
    import prefix20_entry as prefix
    from mask_reduction_memory import make_mask_reduction_loss
    original_loss=prefix.probe.core.OPSDTrainer.generalized_jsd_loss
    counts={};prefix.probe.core.OPSDTrainer.generalized_jsd_loss=staticmethod(make_mask_reduction_loss(original_loss,counts))
    original_init=prefix.seeded.original_init;records=[]
    def init(trainer,*args,**kwargs):
        original_init(trainer,*args,**kwargs)
        assert trainer.model.config.hidden_size==2560 and trainer.model.config.num_hidden_layers==36
        assert trainer.fixed_teacher and trainer.beta==0 and trainer.jsd_token_clip==.05
        assert trainer.args.seed==trainer.args.data_seed==42 and trainer.args.learning_rate==5e-6
        assert list(trainer.train_dataset['problem_id'])==[r['problem_id'] for r in rows]
        assert all(param.device.type=='cuda' for param in trainer.model.parameters())
        assert all('lora_' in name for name,param in trainer.model.named_parameters() if param.requires_grad)
        old_prefetch=trainer.get_batch_samples
        def prefetch(unused,*pos,**named):
            batches,count=old_prefetch(*pos,**named)
            for batch in batches:
                record={}
                for key in ['student_prompts','teacher_prompts']:
                    value=batch[key].detach().cpu().contiguous()
                    record[key+'_sha256']=hashlib.sha256(value.numpy().tobytes()).hexdigest()
                records.append(record)
            return batches,count
        trainer.get_batch_samples=MethodType(prefetch,trainer)
        output.mkdir(parents=True,exist_ok=True)
        (output/'MODEL_SCALE_ACTUAL.json').write_text(json.dumps({'scale':'4B','revision':f['model_revision'],'fresh_original_Base':True,'no_debug_checkpoint_resume':True,'all_HF_parameters_on_CUDA':True,'teacher_same_disabled_adapter_Base':True,'mathematical_loss_retained_with_reduction_order_change':True,'data_sha256':sha(data)},indent=2)+'\n')
        from transformers import TrainerCallback
        class SaveActual(TrainerCallback):
            def on_train_end(self,args,state,control,**unused):
                assert state.global_step==20 and len(records)==640 and counts['position_reduction_calls']==640
                (output/'EXPOSURE_PROMPTS.PRIVATE.json').write_text(json.dumps(records,indent=2)+'\n')
                (output/'MASK_REDUCTION_ACTUAL.json').write_text(json.dumps(counts,indent=2)+'\n')
        trainer.add_callback(SaveActual())
    prefix.seeded.original_init=init
    argv=['--train_language','SW','--translated_data_path',str(data),'--model_name_or_path',str(a.root/'models/Qwen3-4B'),
        '--learning_rate','5e-6','--max_grad_norm','0.1','--per_device_train_batch_size','1','--gradient_accumulation_steps','32',
        '--gradient_checkpointing','--gradient_checkpointing_kwargs','{"use_reentrant": false}','--max_steps','100','--save_steps','5','--logging_steps','2',
        '--output_dir',str(output),'--run_config',f['run_id'],'--max_completion_length','2048','--max_length','20000','--attn_implementation','sdpa','--torch_dtype','bfloat16',
        '--beta','0','--lmbda','1','--use_peft','--lora_r','64','--lora_alpha','128','--lora_target_modules','q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj',
        '--temperature','1.1','--top_p','0.95','--top_k','20','--fixed_teacher','--student_enable_thinking','--jsd_token_clip','0.05',
        '--use_vllm','--vllm_mode','colocate','--vllm_tensor_parallel_size','1','--vllm_gpu_memory_utilization','0.55','--vllm_enable_sleep_mode','true','--vllm_sync_frequency','1',
        '--report_to','tensorboard','--logging_dir',str(a.root/'logs/copsd_official_10lang/tensorboard'/f['run_id']),'--seed','42','--data_seed','42','--wandb_project','COPSD-SW4B-pilot']
    sys.argv=[str(Path(__file__))]+argv;prefix.main()


if __name__=='__main__':main()
