"""Fresh two-GPU BN4B20 single-language pilot, conditional on completed dedicated-server resource gate.

Training uses one CUDA GPU; inference another. Ordinary LoRA/frozenBase teacher,
AdamW and full2048 response loss retained. Debug weights never seed an efficacy run.
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


def prepare(root):
    code=Path(__file__).parent;f=read(code/'bn4b_server20_frozen_20261023.PUBLIC.json')
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    for name,digest in f['installed_source_sha256'].items():assert sha(root/name)==digest,name
    prepared=read(code/'sw4b_resource2_prepared_20261022.PUBLIC.json')
    lock=root/prepared['model_lock_relative'];assert sha(lock)==prepared['model_lock_sha256'];model=read(lock)
    assert model['revision']==f['model_revision']
    for name,v in model['files'].items():assert sha(root/model['model_relative']/name)==v['sha256'],name
    pool=read(code/'sw_lr20_frozen_20261022.PUBLIC.json')
    sw_manifest=root/pool['training_manifest_relative'];assert sha(sw_manifest)==pool['training_manifest_sha256']
    sw=read(sw_manifest);sw_rows=read(root/sw['data_relative']);assert sha(root/sw['data_relative'])==sw['data_sha256']
    manifest=root/f['training_manifest_relative'];assert sha(manifest)==f['training_manifest_sha256']
    m=read(manifest);data=root/m['data_relative'];assert sha(data)==m['data_sha256'];rows=read(data)
    ids=[str(r['problem_id']) for r in rows];assert len(rows)==2700 and ids==sw['training_IDs']
    original=root/m['source_relative'];assert sha(original)==m['source_sha256']
    by_id={str(r['problem_id']):r for r in read(original)};assert rows==[by_id[k] for k in ids]
    assert all(all(x[k]==y[k] for k in ['problem','solution','Answer']) for x,y in zip(rows,sw_rows))
    assert all(isinstance(r['problem_bn'],str) and r['problem_bn'].strip() for r in rows)
    assert not set(ids)&set(read(code/'followup200_frozen_20261021.PUBLIC.json')['selected_IDs'])
    core=read(root/'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json')
    for name,digest in core['core_source_sha256'].items():assert sha(root/'COPSD-ddp-teacher-unwrapped-20261018'/name)==digest,name
    assert read(code/'server_rollout_compat_CPU_20261022.PUBLIC.json')['all_passed']
    assert shutil.disk_usage(root).free>=f['disk_required_bytes']
    output=root/f['checkpoint_relative'];assert not output.exists()
    return f,rows,data,output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    f,rows,data,output=prepare(a.root)
    if a.prepare_only:
        print(json.dumps({'CPU_preflight':True,'scale':'4B','training_GPUs':1,'inference_GPUs':1,'updates':20,'rows':2700,'new_GPU_work':False}));return
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
                'scope':'Fresh BN4B20 seed42 independent training completion only; mathematical efficacy requires matching4BBase and independent paired evaluation. Not100updates or selection of3languages.'}
            (output/'SERVER_PILOT20_UPDATES_DONE.json').write_text(json.dumps(proof,indent=2)+'\n')
            t._server_batch.detach();tracker.detach()
    def init(trainer,*args,**kwargs):
        set_seed(kwargs['args'].seed);original_init(trainer,*args,**kwargs)
        assert trainer.use_vllm and trainer.vllm_mode=='server' and trainer.accelerator.num_processes==1
        assert trainer.model.config.hidden_size==2560 and trainer.model.config.num_hidden_layers==36
        assert trainer.fixed_teacher and trainer.beta==0 and trainer.jsd_token_clip==.05
        assert trainer.args.seed==trainer.args.data_seed==42
        assert trainer.args.resume_from_checkpoint is None
        assert trainer.model.peft_config['default'].r==64 and trainer.model.peft_config['default'].lora_alpha==128
        assert all(v.device.type=='cuda' for v in trainer.model.parameters())
        assert all('lora_' in n for n,v in trainer.model.named_parameters() if v.requires_grad)
        assert list(trainer.train_dataset['problem_id'])==[r['problem_id'] for r in rows]
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
