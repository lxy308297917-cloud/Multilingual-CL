from settings import *
import runpy,gc,torch,traceback

def main():
 stages=[('technical.py',[]),('train.py',['--group',JOB['group']]),('evaluate.py',['--model',STUDY_NAME,'--split','confirmation']),('evaluate.py',['--model',STUDY_NAME,'--split','old'])];phases=[]
 try:
  for script,args in stages:
   budget_guard();tick=time.time();sys.argv=[script]+args;print('START',STUDY_NAME,script,args,tick,flush=True);runpy.run_path(str(REPORT/'code'/script),run_name='__main__');gc.collect();torch.cuda.empty_cache();phases.append({'script':script,'args':args,'wall_seconds':time.time()-tick});write(RUN/('GROUP_PROGRESS_'+STUDY_NAME+'.json'),{'phases':phases,'budget_seconds':budget_seconds()[0]})
  write(RUN/('GROUP_DONE_'+STUDY_NAME+'.json'),{'phases':phases,'finished_unix':time.time(),'one_full_scientific_train':True})
 except BaseException as e:write(RUN/('GROUP_FAILED_'+STUDY_NAME+'.json'),{'phases':phases,'traceback':traceback.format_exc(),'error':str(e)});raise
if __name__=='__main__':main()
