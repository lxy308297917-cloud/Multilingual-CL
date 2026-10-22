"""Four/eight-step original longest-sample numerical and serialized recovery parity."""
from settings import *
import runpy,gc,torch,shutil

def main():
 start=time.time();assert json.loads((RUN/'NUMERIC_CHECK_DONE.json').read_text())['passed'];root=NEWTRAIN/'technical';assert not root.exists()
 def train(name,n,resume=None):
  out=root/name;sys.argv=['train_d1.py','--method','D','--output',str(out),'--smoke','--smoke-steps',str(n)]+(['--resume',str(resume)] if resume else []);runpy.run_path(str(pathlib.Path(__file__).parent/'train_d1.py'),run_name='__main__');gc.collect();torch.cuda.empty_cache();return out/('checkpoint_%06d'%n)
 cp4=train('first4',4);cp8=train('continuous8',8);res8=train('resume4to8',8,cp4)
 from safetensors import safe_open
 coords=0
 with safe_open(str(cp8/'model/model.safetensors'),framework='pt',device='cpu') as x,safe_open(str(res8/'model/model.safetensors'),framework='pt',device='cpu') as y:
  assert x.keys()==y.keys()
  for k in x.keys():a=x.get_tensor(k);b=y.get_tensor(k);assert torch.equal(a,b),k;coords+=a.numel()
 a=torch.load(cp8/'state.pt',map_location='cpu',weights_only=False,mmap=True);b=torch.load(res8/'state.pt',map_location='cpu',weights_only=False,mmap=True);assert a['step']==b['step']==8 and a['counts']==b['counts'] and a['scheduler']==b['scheduler'] and a['aux_cursor']==b['aux_cursor']
 for i in a['optimizer']['state']:
  for k in ('step','exp_avg','exp_avg_sq'):assert torch.equal(a['optimizer']['state'][i][k],b['optimizer']['state'][i][k]),(i,k)
 assert a['rng']['python']==b['rng']['python'];assert torch.equal(a['rng']['torch'],b['rng']['torch']);assert all(torch.equal(x,y) for x,y in zip(a['rng']['cuda'],b['rng']['cuda']));assert a['rng']['numpy'][0]==b['rng']['numpy'][0] and (a['rng']['numpy'][1]==b['rng']['numpy'][1]).all() and a['rng']['numpy'][2:]==b['rng']['numpy'][2:]
 done={'passed':True,'all_model_coordinates_bitwise_equal':coords,'optimizer_tensors':338,'optimizer_scheduler_counts_auxcursor_rng_bitwise_equal':True,'continuous8':str(cp8),'resumed8':str(res8),'checkpoints_sha256':{str(p):sha(p/'RECOVERABLE.json') for p in (cp4,cp8,res8)},'wall_seconds':time.time()-start,'gpu_hours':(time.time()-start)/3600,'alpha':a['alpha']};write(RUN/'TECHNICAL_PARITY_DONE.json',done);write(REPORT/'TECHNICAL_PARITY_DONE.json',done);del a,b
 # Retain final technical checkpoint and all logs/receipts; delete only our newly-created redundant technical weight/state files.
 for cp in (cp4,res8):
  for p in (cp/'model',cp/'state.pt'):
   if p.is_dir():shutil.rmtree(p)
   else:p.unlink()
 print('SERIALIZED_PARITY_PASSED',flush=True)
if __name__=='__main__':main()
