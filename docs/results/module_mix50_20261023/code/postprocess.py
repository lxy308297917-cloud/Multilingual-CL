from settings import *
import subprocess
start=time.time()
while not (RUN/'SERIES_DONE.json').exists():
 state=RUN/'SERIES_PROGRESS.json'
 if state.exists() and json.loads(state.read_text())['failures']:raise RuntimeError('Retained task failure; diagnose before resuming observer')
 assert time.time()-start<24*3600;time.sleep(30)
for script in ('verify.py','analyze.py','report.py'):subprocess.run([sys.executable,str(REPORT/'code'/script)],cwd=ROOT,check=True)
write(RUN/'ANALYSIS_DONE.json',{'finished_unix':time.time()});print('MIX50 ENTIRE ANALYSIS COMPLETE',flush=True)
