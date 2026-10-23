"""One manually authorized preflight + one scientific train + frozen endpoint; no scans."""
from settings import *
import runpy,gc,torch,traceback
worker=json.loads(pathlib.Path(os.environ['LR077_WORKER_RECEIPT']).read_text());start=worker['started_unix'];os.environ['MODULE_BUDGET_GROUP']='S-All';phases=[]
write(RUN/'PIPELINE_START.json',{'worker_start_unix':start,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'max_GPUh':3,'protocol_sha256':sha(REPORT/'PROTOCOL_FROZEN.json')})
try:
 assert json.loads((REPORT/'REFERENCE_REUSE_VERIFIED.json').read_text())['passed'];assert json.loads((REPORT/'PUBLIC_REFERENCE_VERIFIED.json').read_text())['commit']=='d7ebd30dd21d208d373ca9eb54e418fc2ed21e97'
 for script,args,ceiling in [('technical.py',[],.10),('train.py',['--group','S-All'],2.05),('evaluate.py',[],.85)]:
  assert time.time()-start<10800;tick=time.time();sys.argv=[script]+args;print('START',STUDY_NAME,script,tick,flush=True)
  runpy.run_path(str(REPORT/'code'/script),run_name='__main__');gc.collect();torch.cuda.empty_cache();elapsed=time.time()-tick;assert elapsed<=ceiling*3600,('Stage budget exceeded',script,elapsed);assert time.time()-start<=10800,'Global3GPUh ceiling exceeded';phases.append({'script':script,'wall_seconds':elapsed,'ceiling_GPUh':ceiling});write(RUN/'PIPELINE_PROGRESS.json',{'phases':phases,'context_elapsed_seconds':time.time()-start})
 write(RUN/'PIPELINE_DONE.json',{'phases':phases,'total_worker_GPUh':(time.time()-start)/3600,'finished_unix':time.time(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'one_scientific_train':True,'no_followup_automatic':True});print('LR077 FULL GPU PIPELINE DONE',flush=True)
except BaseException as e:write(RUN/'PIPELINE_FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'phases':phases,'charged_seconds':time.time()-start});raise
