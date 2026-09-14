"""Contract tests for the model-path-only evaluation controller."""
import importlib.util,json,tempfile,unittest,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
spec=importlib.util.spec_from_file_location('baseline_controller',R/'scripts/run_baseline_eval.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Contracts(unittest.TestCase):
 def setUp(self):self.c=json.loads((R/'configs/ig_baseline_eval_v1.json').read_text())
 def test_task_budget_and_seeds(self):
  jobs=m.jobs(self.c,False)
  self.assertEqual(len(jobs),23)
  self.assertEqual(len({(j['name'],j['seed']) for j in jobs}),23)
  self.assertEqual(sum(j['generative'] for j in jobs),18)
  for name in ['sum_en','sum_ig','mt_en2ig','mt_ig2en','ifeval','gsm8k']:
   self.assertEqual([j['seed'] for j in jobs if j['name']==name],[42,43,44])
 def test_aggregate_membership(self):
  self.assertEqual(self.c['aggregate'],{'en':['belebele_en','sum_en','mt_ig2en'],'target':['sum_ig','belebele_ig','mt_en2ig']})
 def test_done_requires_identity_and_intact_output(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);ident={'model':'one'}
   self.assertFalse(m.valid_done(d,ident))
   (d/'scores.json').write_text('{"sum_ig":1}');m.write(d/'job.json',ident)
   m.write(d/'DONE.json',{'identity':ident,'artifacts':{'scores.json':m.sha(d/'scores.json'),'job.json':m.sha(d/'job.json')}})
   self.assertTrue(m.valid_done(d,ident))
   with self.assertRaises(RuntimeError):m.valid_done(d,{'model':'two'})
   (d/'scores.json').write_text('{"sum_ig":2}')
   with self.assertRaises(RuntimeError):m.valid_done(d,ident)
if __name__=='__main__':unittest.main()
