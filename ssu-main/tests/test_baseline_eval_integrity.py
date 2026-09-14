import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from baseline_eval_integrity import aggregate,require_scores
class Integrity(unittest.TestCase):
 def test_zero_base_is_undefined_not_success(self):
  rel,total=aggregate({'a':{'mean':1}},{'a':{'mean':0}},{'en':['a']})
  self.assertIsNone(rel['a']);self.assertIsNone(total['en'])
 def test_relative_not_raw_average(self):
  rel,total=aggregate({'a':{'mean':20},'b':{'mean':90}},{'a':{'mean':10},'b':{'mean':100}},{'en':['a','b']})
  self.assertAlmostEqual(total['en'],45)
 def test_invalid_scores_fail(self):
  for data in [{},{'a':float('nan')},{'a':float('inf')}]:
   with self.assertRaises(RuntimeError):require_scores(data)
if __name__=='__main__':unittest.main()
