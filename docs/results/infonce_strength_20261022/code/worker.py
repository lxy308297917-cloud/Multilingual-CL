import argparse,runpy,sys,traceback,time
from settings import *
a=argparse.ArgumentParser();a.add_argument('--uuid',required=True);a.add_argument('--script',required=True);a.add_argument('args',nargs=argparse.REMAINDER);a=a.parse_args()
label=a.script.replace('.py','')+'_'+str(os.getpid())
try:
 acquire(a.uuid,label)
 sys.argv=[a.script]+a.args
 runpy.run_path(str(pathlib.Path(__file__).parent/a.script),run_name='__main__')
 write(RUN/('WORKER_DONE_'+label+'.json'),{'finished_unix':time.time(),'pid':os.getpid()})
except BaseException as e:
 write(RUN/('WORKER_FAILED_'+label+'.json'),{'error':str(e),'traceback':traceback.format_exc(),'finished_unix':time.time()});raise
