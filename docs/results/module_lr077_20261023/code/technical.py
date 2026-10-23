"""Real Qwen longest-sample support/memory checks and serialized S-All4→8 parity."""
from engine import *
import gc,traceback
from safetensors import safe_open

def main():
 start=time.time();OUT.mkdir(parents=True,exist_ok=True);ds,_=data();longest=sorted(range(len(ds)),key=lambda i:len(ds[i]['input_ids']),reverse=True)[:2];batch=[ds[i] for i in longest];results={}
 assert json.loads((REPORT/'OPTIMIZER_TESTS.json').read_text())['passed']
 for group in ('S-All',):
  check_compute();model,tok,named,opt=setup(group);torch.cuda.reset_peak_memory_stats();records=[];counts={'main_input':0,'main_supervised':0,'main_padded_processed':0,'aux_input':0,'aux_supervised':0};cp=None
  for step in range(8):
   record=update(model,tok,named,opt,batch);records.append(record)
   for r in batch:counts['main_input']+=len(r['input_ids']);counts['main_supervised']+=sum(x!=-100 for x in r['labels'][1:]);counts['main_padded_processed']+=len(r['input_ids'])
   if group=='S-All' and step==3:cp=checkpoint(OUT/'technical/S-All/checkpoint_000004',model,opt,counts,group)
  assert all(r['selected']==77070336 for r in records)
  changed=frozen_changed=outside_union=0
  with safe_open(str(BASE/'model.safetensors'),framework='pt',device='cpu') as original:
   for n,p in model.named_parameters():
    old=original.get_tensor(n);new=p.detach().cpu()
    if n in opt.named:
     mask=new!=old;changed+=int(torch.count_nonzero(mask));outside_union+=int(torch.count_nonzero(mask&~opt.unions[n]))
    elif not torch.equal(new,old):frozen_changed+=int(torch.count_nonzero(new!=old))
  assert frozen_changed==outside_union==0;assert changed>0
  result={'group':group,'updates':8,'longest_lengths':[len(r['input_ids']) for r in batch],'selected_per_step':77070336,'frozen_coordinate_changes':frozen_changed,'changed_outside_mask_union':outside_union,'actual_BF16_changed_coordinates':changed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'records':records,'FP32_moments_verified':all(s['m'].dtype==s['v'].dtype==torch.float32 for s in opt.state.values())}
  if group=='S-All':
   snap={'model':{n:p.detach().cpu().clone() for n,p in model.named_parameters()},'optimizer':opt.state_dict(),'rng':rng(),'counts':dict(counts)};restored_counts=load_checkpoint(cp,model,opt,group)
   for step in range(4,8):
    update(model,tok,named,opt,batch)
    for r in batch:restored_counts['main_input']+=len(r['input_ids']);restored_counts['main_supervised']+=sum(x!=-100 for x in r['labels'][1:]);restored_counts['main_padded_processed']+=len(r['input_ids'])
   assert restored_counts==snap['counts']
   for n,p in model.named_parameters():assert torch.equal(p.detach().cpu(),snap['model'][n]),n
   actual=opt.state_dict()
   for n in opt.named:
    for k in ('m','v'):assert torch.equal(actual['state'][n][k],snap['optimizer']['state'][n][k]),(n,k)
    for key in ('masks_packed','unions_packed'):assert torch.equal(actual[key][n],snap['optimizer'][key][n])
   now=rng();assert now['python']==snap['rng']['python'];assert torch.equal(now['torch'],snap['rng']['torch']);assert all(torch.equal(x,y) for x,y in zip(now['cuda'],snap['rng']['cuda']));assert np.array_equal(now['numpy'][1],snap['rng']['numpy'][1]) and now['numpy'][0]==snap['rng']['numpy'][0] and now['numpy'][2:]==snap['rng']['numpy'][2:]
   result['serialized_4to8_all_weights_FP32moments_masks_unions_counts_rng_bitwise_equal']=True;result['technical_checkpoint_receipt_sha256']=sha(cp/'RECOVERABLE.json');del snap,actual,now
   # Only this new redundant technical payload, after full actual replay validation.
   shutil.rmtree(cp/'model');(cp/'state.pt').unlink();result['technical_payload_cleaned_after_verified_parity']=True
  results[group]=result;write(RUN/'TECHNICAL_PARTIAL.json',results);print('TECHNICAL_GROUP_PASSED',group,json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True);del model,tok,named,opt;gc.collect();torch.cuda.empty_cache()
 done={'passed':True,'groups':results,'wall_seconds':time.time()-start,'gpu_hours':(time.time()-start)/3600,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'scope':'LR077 real Qwen longest samples; exact support/frozen checks and scaled-LR4to8 serialized parity; diagnostic updates only, no scientific optimizer carryover'};assert done['gpu_hours']<=.10,'Frozen technical0.10GPUh budget exceeded';write(RUN/'TECHNICAL_DONE.json',done);write(REPORT/'TECHNICAL_DONE.json',done)
if __name__=='__main__':main()
