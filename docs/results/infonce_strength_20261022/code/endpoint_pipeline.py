from settings import *
import runpy,torch,gc
for script in ('evaluate_d1.py','representation_d1.py'):
 sys.argv=[script];runpy.run_path(str(pathlib.Path(__file__).parent/script),run_name='__main__');gc.collect();torch.cuda.empty_cache()
write(RUN/'ENDPOINT_PIPELINE_DONE.json',{'finished_unix':time.time()})
