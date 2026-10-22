"""One manual, bounded pipeline; no automatic retry or coefficient search."""
from settings import *
import runpy,gc,torch,traceback
start=time.time()
try:
 for script in ('numeric_check.py','technical_parity.py'):
  sys.argv=[script];print('START',script,time.time(),flush=True);runpy.run_path(str(pathlib.Path(__file__).parent/script),run_name='__main__');gc.collect();torch.cuda.empty_cache()
 script='train_d1.py';out=NEWTRAIN/'D1_seed42';sys.argv=[script,'--method','D','--output',str(out)];print('START_D1_FULL_FIXED_BUDGET',time.time(),flush=True);runpy.run_path(str(pathlib.Path(__file__).parent/script),run_name='__main__')
 write(RUN/'PIPELINE_DONE.json',{'wall_seconds':time.time()-start,'D1_output':str(out),'finished_unix':time.time(),'no_automatic_expansion':True})
except BaseException as e:
 write(RUN/'PIPELINE_FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'wall_seconds':time.time()-start});raise
