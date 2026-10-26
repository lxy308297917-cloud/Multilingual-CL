"""CPU-only one-shot completion observer; never launches GPU or training work."""
from settings import *
import subprocess,traceback
started=time.time()
try:
 while not (RUN/'SERIES_DONE.json').exists():
  if (RUN/'SERIES_FAILED.json').exists():raise RuntimeError('Evaluation series failed; preserve partial analysis, no silent retry')
  assert time.time()-started<86460,'Evaluation waiting deadline reached'
  time.sleep(30)
 for script,python in [('summarize.py',PYTHON),('plot.py',ROOT/'work/representation_alignment/plot-env/bin/python'),('report.py',PYTHON)]:
  subprocess.run([str(python),str(REPORT/'code'/script)],cwd=ROOT,check=True,env=dict(os.environ,MPLCONFIGDIR=str(RUN/'mpl-cache')))
 r=read(REPORT/'RESULTS.json');assert not r['missing'] and r['new_generation_count']==512 and usage()[0]<=7200
 write(RUN/'FINALIZATION_DONE.json',{'finished_unix':time.time(),'new_responses':512,'GPU_context_hours':usage()[0]/3600,'no_training':True,'needs_final_review_and_publication':True})
except BaseException as e:write(RUN/'FINALIZATION_FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'finished_unix':time.time()});raise
