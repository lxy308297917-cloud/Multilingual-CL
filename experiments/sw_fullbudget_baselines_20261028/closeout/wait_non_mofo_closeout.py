"""Deliver an explicitly incomplete suite while honoring the user's MoFO pause."""
from pathlib import Path
import sys,time,subprocess,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from settings import *
try:
 pause=read(RUN/'USER_PAUSE_MOFO_DONE.json');assert pause['user_requested_pause'] and pause['training_process_stopped']
 while not (RUN/'NON_MOFO_JOBS_DONE.json').exists():
  if (RUN/'QUEUE_AFTER_USER_PAUSE_FAILED.json').exists():raise RuntimeError('Non-MoFO queue failed; inspect rather than silently accept')
  time.sleep(30)
 methods=[m for m in METHODS if m!='MoFO']
 for method in methods:
  subprocess.run([str(PYTHON),str(Path(__file__).with_name('audit_training.py')),'--method',method],check=True)
 subprocess.run([str(EVALPY),str(Path(__file__).with_name('aggregate.py')),'--exclude-user-paused-mofo'],check=True)
 write(REPORT/'NON_MOFO_ACCEPTANCE_DONE.json',dict(methods=methods,paused_methods=['MoFO'],
  complete_original_seven_model_suite=False,unified_endpoint_table=str(REPORT/'results_non_mofo_user_paused/MAIN_TABLE.csv'),
  finished_unix=time.time(),manual_interpretation_pending=True))
except BaseException as error:
 write(REPORT/f'NON_MOFO_CLOSEOUT_FAILED_{time.time_ns()}.json',dict(error=str(error),traceback=traceback.format_exc()));raise
