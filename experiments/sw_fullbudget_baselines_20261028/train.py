"""Independent full-budget training or explicitly identified technical preflight."""
from settings import *
import argparse,traceback
from resources import GPUContext
ap=argparse.ArgumentParser();ap.add_argument('--uuid',required=True);ap.add_argument('--method',choices=METHODS,required=True)
ap.add_argument('--preflight',action='store_true');ap.add_argument('--resume',type=Path);ap.add_argument('--attempt',default='first');a=ap.parse_args()
name=a.method+('_preflight' if a.preflight else '')+'_'+a.attempt
out=H/'models/checkpoint/sw_fullbudget_baselines_20261028'/name
logs=RUN/'training'/name
assert not out.exists() and not logs.exists()
assert read(REPORT/'OVERLAP_AUDIT.json')['passed']
if not a.preflight:
 assert read(REPORT/f'PREFLIGHT_{a.method}.json')['passed']
 assert read(REPORT/'OPTIMIZER_ACCEPTANCE.json')['passed']
try:
 with GPUContext(a.uuid,name) as context:
  import engine,torch,hashlib
  from torch.utils.tensorboard import SummaryWriter
  out.mkdir(parents=True);logs.mkdir(parents=True)
  write(out/'RUN_IDENTITY.json',dict(method=a.method,preflight=a.preflight,attempt=a.attempt,uuid=a.uuid,pid=os.getpid(),
   config_sha256=sha(REPORT/'TRAINING_CONFIG_FROZEN.json'),output_real_path=str(out.resolve()),resume=str(a.resume) if a.resume else None))
  model,tok,named,opt=engine.setup(a.method,out);rows,pool,events=engine.data()
  counts=dict(main_input=0,main_supervised=0,calibration_input=0,calibration_supervised=0,cumulative_actual_path=0.,nonzero_write_events=0)
  if a.resume:counts=engine.load(a.resume,model,opt)
  total=read(REPORT/'DATA_FROZEN.json')['steps']
  if a.preflight:
   wanted=set(read(REPORT/'DATA_FROZEN.json')['longest_source_indices']);longest={}
   with Path(read(REPORT/'DATA_FROZEN.json')['source']).open() as f:
    for i,s in enumerate(f):
     if i in wanted:longest[i]=json.loads(s)
   batches=[[rows[2*i],rows[2*i+1]] for i in range(8)];batches[2]=list(longest.values())
   stats=[]
   def step(i):
    cal=[pool[k] for k in events[opt.steps]] if a.method=='Protected' and opt.steps in events else None
    return engine.update(model,named,opt,batches[i],cal)
   for i in range(4):stats.append(step(i))
   engine.checkpoint(out/'checkpoint_000004',model,opt,dict(preflight_updates=4))
   for i in range(4,8):stats.append(step(i))
   def digest():
    h=hashlib.sha256()
    for n,p in named.items():
     h.update(p.detach().cpu().view(torch.uint16).numpy().tobytes())
     for k in ('m','v'):h.update(opt.state[n][k].cpu().numpy().tobytes())
    return h.hexdigest()
   expected=digest();engine.load(out/'checkpoint_000004',model,opt)
   for i in range(4,8):
    repeated=step(i);assert repeated['actual_norm']==stats[i]['actual_norm']
   assert digest()==expected
   write(REPORT/f'PREFLIGHT_{a.method}.json',dict(passed=True,preflight_only=True,logical_updates=8,replayed_updates=4,
    longest_lengths=[len(r['input_ids']) for r in batches[2]],weights_and_FP32_moments_bitwise_resume=True,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),stats=stats,
    average_regular_step_seconds=sum(sum(s['seconds_by_phase'].values()) for s in stats[3:])/5,finished_unix=time.time()))
  else:
   writer=SummaryWriter(str(logs/'tensorboard'))
   with (logs/'steps.jsonl').open('x') as f:
    while opt.steps<total:
     i=opt.steps;batch=[rows[2*i],rows[2*i+1]]
     cal=[pool[k] for k in events[i]] if a.method=='Protected' and i in events else None
     stats=engine.update(model,named,opt,batch,cal)
     counts['main_input']+=sum(len(r['input_ids']) for r in batch);counts['main_supervised']+=sum(sum(y!=-100 for y in r['labels'][1:]) for r in batch)
     counts['cumulative_actual_path']+=stats['actual_norm'];counts['nonzero_write_events']+=stats['nonzero_write']
     if stats['calibration']:
      counts['calibration_input']+=stats['calibration']['input_tokens'];counts['calibration_supervised']+=stats['calibration']['supervised_tokens']
     f.write(json.dumps(dict(stats,counts=dict(counts)))+'\n')
     if opt.steps%10==0:
      f.flush();write(out/'PROGRESS.json',dict(step=opt.steps,total=total,counts=counts,updated_unix=time.time()))
      for k in ('main_CE','actual_norm','gradient_norm','lr','selected','nonzero_write','mask_union'):writer.add_scalar(k,stats[k],opt.steps)
     if opt.steps%2000==0 or opt.steps==total:engine.checkpoint(out/f'checkpoint_{opt.steps:06d}',model,opt,dict(counts))
   writer.close();d=read(REPORT/'DATA_FROZEN.json');assert counts['main_input']==d['main_input_tokens'] and counts['main_supervised']==d['main_supervised_tokens']
   final=out/f'checkpoint_{total:06d}'/'model'
   write(out/'TRAINING_DONE.json',dict(method=a.method,steps=total,counts=counts,final_model=str(final),
    final_weights_sha256={p.name:sha(p) for p in final.glob('*.safetensors')},
    model_config_sha256=sha(final/'config.json'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    calibrations=opt.calibrations,complete_checkpoint=str(final.parent),finished_unix=time.time()))
except BaseException as e:
 if isinstance(e,SystemExit) and e.code==75:raise
 write(RUN/f'FAILED_{name}.json',dict(error=str(e),traceback=traceback.format_exc(),failed_unix=time.time(),output=str(out)))
 raise
