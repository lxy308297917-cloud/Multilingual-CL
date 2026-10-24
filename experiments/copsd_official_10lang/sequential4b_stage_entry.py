"""Selected 4B research stages and exactly matched fresh-Base single controls.

Official core, full2048 online generation and fresh stage optimization retained.
"""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
from types import MethodType


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''):h.update(block)
    return h.hexdigest()


from sequential4b_context import prepare

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--freeze',type=Path,required=True);a=parser.parse_args()
    f,rows,data,output=prepare(a.root,a.freeze)
    if a.prepare_only:
        print(json.dumps({'CPU_preflight':True,'scale':'4B','training_GPUs':1,'inference_GPUs':1,'updates':20,'rows':len(rows),'new_GPU_work':False}));return
    gate=read(Path(__file__).with_name('sw4b_server_resource_v2_completed_20261022.PUBLIC.json'))
    assert gate['engineering_completed'] and gate['completed_updates']==2
    assert gate['frozen_sha256']==sha(Path(__file__).with_name('sw4b_server_resource_v2_frozen_20261022.PUBLIC.json'))
    assert gate['model_revision']==f['model_revision']
    assert all(r['sampled_headroom_MiB']>=512 for r in gate['memory_per_role'])
    assert os.environ['GPU_QUEUE_GPU_COUNT']=='2'
    visible=os.environ['CUDA_VISIBLE_DEVICES'];assert visible.startswith('GPU-') and ',' not in visible
    assert visible!=os.environ['COPSD_SERVER_GPU_UUID']
    from benchmark_eval_concurrency8192 import wait_for_empty_GPU
    wait_for_empty_GPU(visible)
    output.mkdir(parents=True)
    (output/'SEQUENTIAL4B_STAGE_CONTEXT.PRIVATE.json').write_text(json.dumps({'context':f['stage_context'],'selection_record_sha256':f['selection_record_sha256'],'freeze_sha256':sha(a.freeze)},indent=2)+'\n')
    import torch
    from transformers import TrainerCallback,set_seed
    source=a.root/'COPSD-ddp-teacher-unwrapped-20261018';sys.path.insert(0,str(source))
    import multilingual_opsd_trainer as core
    from server_rollout_compat import compatible_client,ServerUpdateBatch
    from lora_sync_weights import iter_merged_weights
    from cost_instrumentation import CostTracker
    from mask_reduction_memory import make_mask_reduction_loss
    from resume_torch28_entry import enable_numpy_rng_compat
    enable_numpy_rng_compat()
    core.VLLMClient=compatible_client(core.VLLMClient)
    counts={};tracker=CostTracker()
    core.OPSDTrainer.generalized_jsd_loss=staticmethod(tracker.observe_loss(make_mask_reduction_loss(core.OPSDTrainer.generalized_jsd_loss,counts)))
    from compact_response_logits import make_compact_compute_loss
    core.OPSDTrainer.compute_loss=make_compact_compute_loss(core.OPSDTrainer.compute_loss)
    original_init=core.OPSDTrainer.__init__;original_step=core.OPSDTrainer.training_step
    original_generation=core.OPSDTrainer._generate_on_policy_outputs_vllm
    exposure_prompts=[]
    receipts=Path(os.environ['COPSD_SERVER_RECEIPTS']);consumed=0
    def fingerprint(parameters):
        h=hashlib.sha256()
        for name,value in parameters:
            h.update(name.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    def sync(trainer):
        nonlocal consumed
        trainer._server_batch.assert_drained();tick=time.monotonic();expected=[]
        for name,tensor in iter_merged_weights(trainer.model):
            digest=hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            expected.append({'name':name,'shape':list(tensor.shape),'dtype':str(tensor.dtype),'sha256':digest,'original_loader_returned':True})
            trainer.vllm_client.update_named_param(name,tensor)
        trainer.vllm_client.reset_prefix_cache()
        actual=[json.loads(line) for line in receipts.read_text().splitlines()]
        assert actual[consumed:]==expected,'Remote received/loaded tensors differ from exported student'
        consumed=len(actual)
        assert fingerprint((n,v) for n,v in trainer.model.named_parameters() if not v.requires_grad)==trainer._server_base_hash
        row={'global_step':trainer.state.global_step,'parameter_count':len(expected),
            'export_and_remote_loaded_sha256':hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest(),
            'adapter_hash':fingerprint((n,v) for n,v in trainer.model.named_parameters() if v.requires_grad),
            'seconds_including_diagnostic_hashing':time.monotonic()-tick}
        trainer._server_syncs.append(row)
        (output/'SERVER_SYNC_PROGRESS.json').write_text(json.dumps(trainer._server_syncs,indent=2)+'\n')
        print('SERVER_SYNC_VERIFIED',json.dumps(row),flush=True)
    class Resource(TrainerCallback):
        def __init__(self,trainer):self.trainer=trainer
        def on_train_begin(self,args,state,control,**unused):
            t=self.trainer;assert state.global_step==0
            from sequential_stage_entry import verify_optimization_start
            t._stage_optimizer_receipt=verify_optimization_start(t,state,0)
            assert args.max_steps==100 and args.gradient_accumulation_steps==32 and args.per_device_train_batch_size==1
            t._server_base_hash=fingerprint((n,v) for n,v in t.model.named_parameters() if not v.requires_grad)
            t._server_started=time.monotonic();t._move_model_to_vllm();t._last_vllm_sync_step=0
        def on_step_end(self,args,state,control,**unused):
            if state.global_step>=20:control.should_training_stop=True;control.should_save=True
            return control
        def on_save(self,args,state,control,**unused):
            cost=tracker.snapshot();cost['global_step']=state.global_step
            (output/'SERVER_COST_PROGRESS.json').write_text(json.dumps(cost,indent=2)+'\n')
        def on_train_end(self,args,state,control,**unused):
            t=self.trainer;t._server_batch.assert_drained()
            assert state.global_step==20 and t._server_microsteps==640 and len(t._server_syncs)==21
            assert counts['position_reduction_calls']==640 and len(exposure_prompts)==640
            assert fingerprint((n,v) for n,v in t.model.named_parameters() if not v.requires_grad)==t._server_base_hash
            assert len({r['adapter_hash'] for r in t._server_syncs})==21
            cost=tracker.snapshot();assert cost['loss_calls']==cost['teacher_forward_calls']==cost['student_forward_calls']==640
            proof={'global_step':20,'completed_updates':20,'micro_steps':640,'effective_batch':32,'scheduler_horizon':100,
                'all_HF_parameters_on_CUDA':True,'base_unchanged_bitwise':True,'teacher_via_disabled_original_Base':True,
                'syncs':t._server_syncs,'cost':cost,'memory_reduction':counts,'batch_generation':t._server_batch.records,
                'train_seconds_including_diagnostic_hashing':time.monotonic()-t._server_started,
                'engineering_only':False,'stage_context':f['stage_context'],'parent_step':20 if f.get('parent_checkpoint_relative') else None,'stage_handoff':t._stage_handoff_receipt,'stage_optimizer_start':t._stage_optimizer_receipt,'scope':'Selected4B research stage on its frozen language-bound900-question pool;20updates with resetoptimizer/scheduler under100horizon. Method efficacy and forgetting require complete paired stage evaluations.'}
            (output/'SEQUENTIAL4B_RESEARCH_UPDATES_DONE.json').write_text(json.dumps(proof,indent=2)+'\n')
            t._server_batch.detach();tracker.detach()
    def init(trainer,*args,**kwargs):
        set_seed(kwargs['args'].seed);original_init(trainer,*args,**kwargs)
        assert trainer.use_vllm and trainer.vllm_mode=='server' and trainer.accelerator.num_processes==1
        assert trainer.model.config.hidden_size==2560 and trainer.model.config.num_hidden_layers==36
        assert trainer.fixed_teacher and trainer.beta==0 and trainer.jsd_token_clip==.05
        assert trainer.args.seed==f['training_seed'] and trainer.args.data_seed==f['data_seed']
        assert f['train_arguments'][f['train_arguments'].index('--train_language')+1]==f['target_language']
        assert trainer.args.resume_from_checkpoint is None
        assert trainer.model.peft_config['default'].r==64 and trainer.model.peft_config['default'].lora_alpha==128
        assert all(v.device.type=='cuda' for v in trainer.model.parameters())
        assert all('lora_' in n for n,v in trainer.model.named_parameters() if v.requires_grad)
        assert list(trainer.train_dataset['problem_id'])==[r['problem_id'] for r in rows]
        from sequential_stage_entry import initialize_stage_model
        parent=a.root/f['parent_checkpoint_relative'] if f.get('parent_checkpoint_relative') else None
        trainer._stage_handoff_receipt=initialize_stage_model(trainer,parent,20 if parent else None)
        original_prefetch=trainer.get_batch_samples
        def prefetch(unused,*pos,**named):
            batches,count=original_prefetch(*pos,**named)
            for batch in batches:
                row={}
                for key in ['student_prompts','teacher_prompts']:
                    tensor=batch[key].detach().cpu().contiguous()
                    row[key+'_sha256']=hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                exposure_prompts.append(row)
            (output/'EXPOSURE_PROMPTS.PRIVATE.json').write_text(json.dumps(exposure_prompts,indent=2)+'\n')
            return batches,count
        trainer.get_batch_samples=MethodType(prefetch,trainer)
        trainer._server_microsteps=0;trainer._server_syncs=[]
        trainer._server_batch=ServerUpdateBatch(trainer,output);trainer._server_batch.install()
        tracker.attach(trainer);trainer.add_callback(Resource(trainer))
    def step(trainer,*args,**kwargs):
        result=original_step(trainer,*args,**kwargs);torch.cuda.synchronize()
        assert bool(torch.isfinite(result).all());trainer._server_microsteps+=1
        print('SERVER_MICROSTEP',json.dumps({'global_step':trainer.state.global_step,'micro_step':trainer._server_microsteps,'loss_finite':True}),flush=True)
        return result
    def generation(trainer,*args,**kwargs):
        assert trainer._last_vllm_sync_step==trainer.state.global_step
        assert fingerprint((n,v) for n,v in trainer.model.named_parameters() if v.requires_grad)==trainer._server_syncs[-1]['adapter_hash']
        return original_generation(trainer,*args,**kwargs)
    core.OPSDTrainer.__init__=init;core.OPSDTrainer._move_model_to_vllm=sync
    core.OPSDTrainer.training_step=step;core.OPSDTrainer._generate_on_policy_outputs_vllm=generation
    argv=f['train_arguments']+['--translated_data_path',str(data),'--model_name_or_path',str(a.root/'models/Qwen3-4B'),
        '--output_dir',str(output),'--run_config',f['run_id'],'--vllm_server_host','127.0.0.1','--vllm_server_port',os.environ['COPSD_SERVER_HTTP_PORT'],
        '--logging_dir',str(a.root/'logs/copsd_official_10lang/tensorboard'/f['run_id'])]
    sys.argv=[str(source/'multilingual_opsd_train.py')]+argv
    namespace=runpy.run_path(sys.argv[0],run_name='__main__');trainer=namespace['trainer']
    trainer.vllm_client.close_communicator();atexit.unregister(trainer.vllm_client.close_communicator)
    for callback in trainer.callback_handler.callbacks:
        writer=getattr(callback,'tb_writer',None)
        if writer:writer.flush();writer.close()
    import wandb
    wandb.finish()
    (output/'SERVER_CLIENT_SHUTDOWN_DONE.json').write_text(json.dumps({'communicator_closed':True,'scope':'Parent must separately confirm bothroles exited normally.'})+'\n')


if __name__=='__main__':main()
