"""One-factor SW learning-rate diagnostic on the same heldout-safe pool.

Official trainer/loss/optimizer unchanged. Fresh paired5e-6 reference versus1e-6,
20 updates under100-update scheduler. Not a new official reproduction or selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from types import MethodType


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(root,condition):
    assert shutil.disk_usage(root).free >= 20 * 2**30
    code=Path(__file__).parent;freeze=code/'sw_lr20_frozen_20261022.PUBLIC.json';f=read(freeze)
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    from matched_pool_single_entry import validate_assets
    validate_assets(root,code)
    manifest=root/f['training_manifest_relative'];assert sha(manifest)==f['training_manifest_sha256'];m=read(manifest)
    data=root/m['data_relative'];assert sha(data)==m['data_sha256'];rows=read(data)
    assert len(rows)==2700 and [str(r['problem_id']) for r in rows]==m['training_IDs']
    source=root/m['original_source_relative'];assert sha(source)==m['original_source_sha256']
    excluded=root/m['exclusion_relative'];assert sha(excluded)==m['exclusion_sha256']
    assert sha(code/'followup200_frozen_20261021.PUBLIC.json')==f['followup_data_freeze_sha256']
    old=read(source);keep=set(m['training_IDs']);assert rows==[r for r in old if str(r['problem_id']) in keep]
    assert not keep & set(read(code/'followup200_frozen_20261021.PUBLIC.json')['selected_IDs'])
    spec=f['conditions'][condition];output=root/spec['checkpoint_relative'];assert not output.exists()
    for name in spec['evaluation_run_ids'].values():assert not (root/'logs/copsd_official_10lang/development_vllm8192'/name).exists()
    assert sha(root/f['initial_reference_relative'])==f['initial_reference_sha256']
    return f,spec,rows,data,output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--condition',choices=['reference','lower_lr'],required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    f,spec,rows,data,output=prepare(a.root,a.condition);code=Path(__file__).parent
    if a.prepare_only:
        print(json.dumps({'CPU_preflight':True,'condition':a.condition,'learning_rate':spec['learning_rate'],'rows':len(rows),'updates':20,'new_GPU_work':False}));return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT')=='1'
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','');assert visible.startswith('GPU-') and ',' not in visible
    processes=subprocess.run(['nvidia-smi','-i',visible,'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    assert all(s.strip().isdigit() and int(s.strip())==os.getpid() for s in processes.stdout.splitlines() if s.strip()),'Foreign process on allocated GPU'
    from cumem_none_ref_compat import install
    compat=read(code/'ja_nonefixed_runtime_20261020.CONFIG.json');install(compat['binary_sha256'])
    import prompt_budget_guard
    prompt_budget_guard.install()
    import prefix20_entry as prefix
    original_init=prefix.seeded.original_init
    records=[]
    def init(trainer,*args,**kwargs):
        original_init(trainer,*args,**kwargs)
        assert trainer.args.seed==trainer.args.data_seed==42 and trainer.args.learning_rate==spec['learning_rate']
        assert trainer.fixed_teacher and trainer.beta==0 and trainer.jsd_token_clip==.05
        assert list(trainer.train_dataset['problem_id'])==[r['problem_id'] for r in rows]
        old_prefetch=trainer.get_batch_samples
        def prefetch(unused,*positional,**named):
            batches,count=old_prefetch(*positional,**named)
            for batch in batches:
                record={}
                for key in ['student_prompts','teacher_prompts']:
                    tensors=batch[key].detach().cpu().contiguous()
                    record[key+'_sha256']=hashlib.sha256(tensors.numpy().tobytes()).hexdigest()
                records.append(record)
            return batches,count
        trainer.get_batch_samples=MethodType(prefetch,trainer)
        output.mkdir(parents=True,exist_ok=True)
        (output/'LR_DIAGNOSTIC_ACTUAL.json').write_text(json.dumps({'protocol':f['protocol'],'condition':a.condition,'learning_rate':trainer.args.learning_rate,'frozen_sha256':sha(code/'sw_lr20_frozen_20261022.PUBLIC.json'),'training_data_sha256':sha(data),'rows':len(rows),'seed':42,'core_loss_replaced':False,'scheduler_horizon':100,'stop_after_updates':20},indent=2)+'\n')
        from transformers import TrainerCallback
        class SavePrompts(TrainerCallback):
            def on_train_end(self,args,state,control,**unused):
                assert state.global_step==20 and len(records)==640
                (output/'LR_EXPOSURE_PROMPTS.PRIVATE.json').write_text(json.dumps(records,indent=2)+'\n')
        trainer.add_callback(SavePrompts())
    prefix.seeded.original_init=init
    args=['--train_language','SW','--translated_data_path',str(data),'--model_name_or_path',str(a.root/'models/Qwen3-1.7B'),
        '--learning_rate',str(spec['learning_rate']),'--max_grad_norm','0.1','--per_device_train_batch_size','1','--gradient_accumulation_steps','32',
        '--gradient_checkpointing','--gradient_checkpointing_kwargs','{"use_reentrant": false}','--max_steps','100','--save_steps','5','--logging_steps','2',
        '--output_dir',str(output),'--run_config',spec['run_id'],'--max_completion_length','2048','--max_length','20000','--attn_implementation','sdpa','--torch_dtype','bfloat16',
        '--beta','0','--lmbda','1','--use_peft','--lora_r','64','--lora_alpha','128','--lora_target_modules','q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj',
        '--temperature','1.1','--top_p','0.95','--top_k','20','--fixed_teacher','--student_enable_thinking','--jsd_token_clip','0.05',
        '--use_vllm','--vllm_mode','colocate','--vllm_tensor_parallel_size','1','--vllm_gpu_memory_utilization','0.4','--vllm_enable_sleep_mode','true','--vllm_sync_frequency','1',
        '--report_to','tensorboard','--logging_dir',str(a.root/'logs/copsd_official_10lang/tensorboard'/spec['run_id']),'--seed','42','--data_seed','42','--wandb_project','COPSD-SW-lr-diagnostic']
    sys.argv=[str(Path(__file__))]+args;prefix.main()


if __name__=='__main__':main()
