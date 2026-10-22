"""Exactly4293 main-only sparse updates; shared data, clip scope, LR and seed."""
from engine import *
import argparse,traceback
from torch.utils.tensorboard import SummaryWriter
from safetensors import safe_open

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['S-All','S-Attn','S-MLP'],required=True);ap.add_argument('--resume');ap.add_argument('--output');a=ap.parse_args();group=a.group;assert not a.resume or a.output,'Recovery requires independent output directory';out=pathlib.Path(a.output) if a.output else OUT/group;out.mkdir(parents=True,exist_ok=False);start=time.time();physical=[];stop=threading.Event();writer=None
 def sample():
  while not stop.is_set():
   try:physical.append(int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip()))
   except Exception:pass
   stop.wait(5)
 sampler=threading.Thread(target=sample,daemon=True);sampler.start()
 try:
  assert json.loads((RUN/'TECHNICAL_DONE.json').read_text())['passed'];protocol=json.loads((REPORT/'PROTOCOL_FROZEN.json').read_text());assert sha(REPORT/'BUDGET_FROZEN.json')==protocol['budget_sha256'];ds,m=data();model,tok,named,opt=setup(group);torch.cuda.reset_peak_memory_stats();counts={'main_input':0,'main_supervised':0,'main_padded_processed':0,'aux_input':0,'aux_supervised':0,'generated_tokens':0};durations=[];cumulative_path=0.;cumulative_writes=0;bad=0;module_path={};prior_seconds=0.;start_step=0
  if a.resume:
   cp=pathlib.Path(a.resume);counts=load_checkpoint(cp,model,opt,group);charge=json.loads((cp/'CHARGE_RECEIPT.json').read_text());prior_seconds=charge['charged_seconds'];cumulative_path=charge['cumulative_path_length'];cumulative_writes=charge['cumulative_nonzero_writes'];module_path=charge['module_path_length'];start_step=opt.steps;assert start_step in (2146,4293)
  writer=SummaryWriter(str(H/'logs/module_budget_sparse_20261022/tensorboard'/group))
  write(out/'CONFIG.json',{'group':group,'resume':a.resume,'start_step':start_step,'prior_gpu_seconds':prior_seconds,'protocol_sha256':sha(REPORT/'PROTOCOL_FROZEN.json'),'budget_sha256':sha(REPORT/'BUDGET_FROZEN.json'),'model_sha256':sha(BASE/'model.safetensors'),'tokenizer_sha256':sha(BASE/'tokenizer.json'),'main':m,'code_sha256':sha(__file__),'engine_sha256':sha(pathlib.Path(__file__).parent/'engine.py'),'optimizer_sha256':sha(pathlib.Path(__file__).parent/'optimizer.py'),'git_commit':subprocess.check_output(['git','-C',str(REPORT),'rev-parse','HEAD'],text=True).strip(),'python':sys.executable,'torch':torch.__version__,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'started_unix':time.time(),'gradient_enabled_candidate_elements':1310195712,'eligible_moment_elements':sum(p.numel() for p in opt.named.values()),'selected_K':opt.K,'no_auxiliary_data':True})
  with (out/'STEPS.jsonl').open('w') as log:
   for step in range(start_step,4293):
    if step%20==0:check_compute()
    assert prior_seconds+time.time()-start<4*3600,'Frozen4GPUh training ceiling';tick=time.time();batch=[ds[2*step],ds[2*step+1]];stats=update(model,tok,named,opt,batch);assert stats['selected']==77070336
    for r in batch:counts['main_input']+=len(r['input_ids']);counts['main_supervised']+=sum(x!=-100 for x in r['labels'][1:]);counts['main_padded_processed']+=len(r['input_ids'])
    bad=bad+1 if stats['main_CE']>20 else 0;assert bad<100,'Persistent mainCE>20';durations.append(time.time()-tick);cumulative_path+=stats['actual_norm'];cumulative_writes+=stats['nonzero_write']
    for module,v in stats['per_module'].items():module_path[module]=module_path.get(module,0.)+math.sqrt(v['actual_delta_sq'])
    stats.update(seconds=durations[-1],counts=dict(counts),cumulative_path_length=cumulative_path,cumulative_nonzero_writes=cumulative_writes);log.write(json.dumps(stats)+'\n');log.flush()
    if step==31:
     projected=sum(durations[1:])/len(durations[1:])*4293+180;gate={'passed':projected<4*3600*.9,'projected_seconds_including180s_checkpoint_allowance':projected,'ceiling_seconds':14400,'measured_updates':32};write(out/'THROUGHPUT_GATE.json',gate);assert gate['passed'],'Frozen throughput gate failed'
    if (step+1)%20==0 or step==4292:
     for k in ('main_CE','full_candidate_preclip_norm','actual_norm','candidate_norm','nonzero_write','mask_union','lr'):writer.add_scalar(k,stats[k],step+1)
     write(out/'PROGRESS.json',{'step':step+1,'counts':counts,'elapsed_seconds':prior_seconds+time.time()-start,'mean_update_seconds':sum(durations)/len(durations),'selected_K':opt.K,'physical_peak_mib_observed':max(physical) if physical else None});print(group,step+1,json.dumps({k:stats[k] for k in ('main_CE','actual_norm','nonzero_write','mask_union')}),flush=True)
    if step+1 in (2146,4293):
     model.zero_grad(set_to_none=True);cp=checkpoint(out/('checkpoint_%06d'%(step+1)),model,opt,counts,group);loaded_counts=load_checkpoint(cp,model,opt,group);assert loaded_counts==counts;write(cp/'TRANSPORT_VERIFIED.json',{'step':step+1,'actual_model_optimizer_scheduler_RNG_loader_executed':True,'no_extra_update':True});write(cp/'CHARGE_RECEIPT.json',{'charged_seconds':prior_seconds+time.time()-start,'cumulative_path_length':cumulative_path,'cumulative_nonzero_writes':cumulative_writes,'module_path_length':module_path})
  assert counts['main_input']==1536351 and counts['main_supervised']==599469;final_stats={};frozen_changed=0
  with safe_open(str(BASE/'model.safetensors'),framework='pt',device='cpu') as original:
   for n,p in model.named_parameters():
    old=original.get_tensor(n);new=p.detach().cpu();assert bool(torch.isfinite(new).all())
    if n not in opt.named:assert torch.equal(new,old),'Frozen position changed '+n;continue
    difference=new.float()-old.float();changed=new!=old;assert not bool((changed&~opt.unions[n]).any());module='Attention' if '.self_attn.' in n else 'MLP';v=final_stats.setdefault(module,{'final_base_delta_sq':0.,'final_changed_coordinates':0,'mask_union':0,'eligible_elements':0});v['final_base_delta_sq']+=float(difference.square().sum());v['final_changed_coordinates']+=int(torch.count_nonzero(changed));v['mask_union']+=opt.union_counts[n];v['eligible_elements']+=p.numel()
  for v in final_stats.values():v['final_base_delta_L2']=math.sqrt(v['final_base_delta_sq'])
  writer.close();writer=None;write(out/'TRAINING_DONE.json',{'group':group,'steps':4293,'counts':counts,'K':opt.K,'model':str(out/'checkpoint_004293/model'),'final_model_sha256':sha(out/'checkpoint_004293/model/model.safetensors'),'protocol_sha256':sha(REPORT/'PROTOCOL_FROZEN.json'),'elapsed_seconds':prior_seconds+time.time()-start,'gpu_hours':(prior_seconds+time.time()-start)/3600,'this_execution_gpu_hours':(time.time()-start)/3600,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'physical_peak_mib_observed':max(physical) if physical else None,'cumulative_path_length':cumulative_path,'cumulative_nonzero_write_events':cumulative_writes,'module_path_length':module_path,'final_base_delta_L2':math.sqrt(sum(v['final_base_delta_sq'] for v in final_stats.values())),'final_modules':final_stats,'frozen_positions_unchanged':True,'mask_union_note':'Union of selected masks including LR0; not a count of unique coordinates actually written. Final changed-from-Base count is reported separately.','moments':'FP32 exact full eligible matrix states; excluded matrices have no state','zero_auxiliary':True})
 except BaseException as e:write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start});raise
 finally:
  if writer:writer.close()
  stop.set();sampler.join(timeout=10)
if __name__=='__main__':main()
