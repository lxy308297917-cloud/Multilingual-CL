"""Manually launched, one-shot CPU acceptance after the already authorized queue."""
import sys,subprocess,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from settings import *

try:
 while not (RUN/'ALL_JOBS_DONE.json').exists():
  if (RUN/'QUEUE_FAILED.json').exists():raise RuntimeError('Training/evaluation queue failed; preserve artifacts and inspect before closeout')
  time.sleep(60)
 for method in METHODS:
  subprocess.run([str(PYTHON),str(Path(__file__).with_name('audit_training.py')),'--method',method],check=True)
 subprocess.run([str(EVALPY),str(Path(__file__).with_name('aggregate.py'))],check=True)
 write(REPORT/'SCIENTIFIC_ACCEPTANCE_DONE.json',dict(completed_unix=time.time(),training_methods=list(METHODS),
  unified_endpoint_table=str(REPORT/'results/MAIN_TABLE.csv'),human_interpretation_pending=True))
except BaseException as error:
 write(REPORT/f'CLOSEOUT_FAILED_{time.time_ns()}.json',dict(error=str(error),traceback=traceback.format_exc()))
 raise
