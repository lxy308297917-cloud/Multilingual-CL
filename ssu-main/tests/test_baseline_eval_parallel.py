"""Concurrent scheduling must isolate GPUs, preserve task identities and resume."""
import sys,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/src'))
import run_baseline_eval as m
class Parallel(unittest.TestCase):
 def test_disjoint_gpu_lanes_and_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);tasks=[{'name':'t'+str(i),'seed':42,'kind':'general'} for i in range(4)]
   c={'output_root':tmp,'gpu':0,'evaluation_gpus':[0,1],'hf_cache':tmp,'reserve_bytes':0,'training_python':sys.executable,'evaluation_python':sys.executable}
   a=SimpleNamespace(smoke=True,replica='primary',config=root/'config.json',lock_fd=1)
   active=set();calls=[];lock=threading.Lock();barrier=threading.Barrier(2)
   def worker(cmd,env,**kw):
    gpu=env['CUDA_VISIBLE_DEVICES'];dest=Path(cmd[-1]);job=__import__('json').loads((dest/'job.json').read_text())['job']
    with lock:
     self.assertNotIn(gpu,active);active.add(gpu);calls.append((job['name'],gpu))
    barrier.wait(timeout=10)
    m.write(dest/'scores.json',{job['name']:1.0})
    with lock:active.remove(gpu)
    return SimpleNamespace(returncode=0)
   with patch.object(m,'model_identity',return_value={'path':'base'}),patch.object(m,'jobs',return_value=tasks),patch.object(m.subprocess,'run',side_effect=worker):
    dest,summary=m.run_model(root,c,a,{'frozen':True})
    self.assertEqual(set(calls),{('t0','0'),('t1','1'),('t2','0'),('t3','1')})
    self.assertEqual(len(summary),4)
    m.run_model(root,c,a,{'frozen':True})
    self.assertEqual(len(calls),4)
if __name__=='__main__':unittest.main()
