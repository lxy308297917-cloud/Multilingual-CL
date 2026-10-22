"""One authorized group: full fixed prefix then full frozen endpoint, no intermediate selection."""
from settings import *
import torch,gc,runpy,traceback
start=time.time();group=os.environ['MODULE_BUDGET_GROUP'];assert group in ('S-All','S-Attn','S-MLP')
try:
 for script,args in [('train.py',['--group',group]),('evaluate.py',[])]:
  sys.argv=[script]+args;print('START',group,script,time.time(),flush=True);runpy.run_path(str(REPORT/'code'/script),run_name='__main__');gc.collect();torch.cuda.empty_cache()
 write(RUN/('GROUP_DONE_'+group+'.json'),{'group':group,'wall_seconds':time.time()-start,'finished_unix':time.time(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']})
except BaseException as e:write(RUN/('GROUP_FAILED_'+group+'.json'),{'error':str(e),'traceback':traceback.format_exc(),'wall_seconds':time.time()-start});raise
