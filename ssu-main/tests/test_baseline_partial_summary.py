import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'analysis'));sys.path.insert(0,str(R/'scripts'))
import summarize_baseline_suite as m
from run_baseline_eval import sha,write,digest,jobs
class Partial(unittest.TestCase):
 def test_incomplete_seeds_never_become_composite(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/'repo';root.mkdir();(root/'configs').mkdir();exp=root.parent/'experiments';suite=exp/'ig_baseline_suite_v1';suite.mkdir(parents=True);out=exp/'eval';out.mkdir()
   c={'output_root':str(out),'base_model':str(root/'base'),'generation_seeds':[42,43,44],'classification_seed':42,'tasks':[{'name':n,'generative':True} for n in ['a','b','c']],'aggregate':{'en':['a','b','c'],'target':['a','b','c']}}
   write(root/'configs/ig_baseline_eval_v2.json',c);write(root/'configs/ig_baseline_repro_v1.json',{'models':{'full_fft':str(root/'fft'),'ssu_en_freeze50':str(root/'ssu')}});write(root/'configs/ig_baseline_train_v1.json',{'output_root':str(suite/'retrained'),'methods':{}});write(out/'protocol_lock.json',{'fixed':True})
   def add(seed):
    for name,value in [('base',100),('fft',110)]:
     mid={'path':str(root/name),'files':{}};dest=out/'evaluation'/name/'primary';write(dest/'model.json',mid)
     for job in jobs(c,False):
      if job['name'] not in ['a','b','c'] or job['seed']!=seed:continue
      attempt=dest/(job['name']+'-seed'+str(seed))/'attempt-1';ident={'protocol':digest({'fixed':True}),'model':digest(mid),'job':job,'smoke':False}
      write(attempt/'job.json',ident);write(attempt/'scores.json',{job['name']:value});write(attempt/'DONE.json',{'identity':ident,'artifacts':{n:sha(attempt/n) for n in ['job.json','scores.json']}})
   with patch.object(m,'R',root):
    add(42);add(43);m.refresh();result=json.loads((suite/'formal_progress.json').read_text());self.assertIsNone(result['aggregates']['legacy_full_fft']['target'])
    add(44);m.refresh();result=json.loads((suite/'formal_progress.json').read_text());self.assertAlmostEqual(result['aggregates']['legacy_full_fft']['target'],10)
if __name__=='__main__':unittest.main()
