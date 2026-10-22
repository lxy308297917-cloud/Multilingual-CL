import unittest
from mastery import numeric_value,extract_answer,judge,MasteryCache

def outputs(ok=True,capped=False):
 return [{'kind':'greedy' if i==0 else 'sample','seed':None if i==0 else 16+i,'text':'\\boxed{0.5}' if ok else '\\boxed{2}','capped':capped} for i in range(4)]

class Tests(unittest.TestCase):
 def test_equivalence_and_false_negative_distractors(self):
  for s in ['.5','0.500','1/2','\\frac{1}{2}','5e-1']:
   self.assertEqual(numeric_value(s),numeric_value('0.5'))
  self.assertEqual(numeric_value('1,000'),numeric_value('1000'))
  for s in ['1,2','nan','inf','2 apples','1/0','0.5 or 2']:
   self.assertIsNone(numeric_value(s))
 def test_nested_box_and_no_incidental_number(self):
  self.assertEqual(extract_answer('Compute 9 things. \\boxed{\\frac{1}{2}}')[0],numeric_value('.5'))
  self.assertIsNone(extract_answer('There are 17 apples and 2 people')[0])
 def test_conservative_gate(self):
  rows=outputs();self.assertTrue(judge(rows,'.5')[0]);rows[0]['text']='2';self.assertFalse(judge(rows,'.5')[0])
  self.assertFalse(judge(outputs(capped=True),'.5')[0])
 def test_base_never_revoked_and_new_latches(self):
  cache=MasteryCache('data','prompt')
  def result(a,b):return {'base':{'gold':'.5','outputs':outputs(a)},'new':{'gold':'.5','outputs':outputs(b)}}
  self.assertEqual(cache.refresh(result(True,False),round_id=0,model_identity='base'),{'base':True,'new':False})
  self.assertEqual(cache.refresh(result(False,True),round_id=1,model_identity='s512'),{'base':True,'new':False})
  self.assertEqual(cache.refresh(result(False,True),round_id=2,model_identity='s1024'),{'base':True,'new':True})
  self.assertEqual(cache.refresh(result(False,False),round_id=3,model_identity='s1536'),{'base':True,'new':True})
  self.assertFalse(cache.records['base']['acquired'])
  restored=MasteryCache('data','prompt');restored.load_state_dict(cache.state_dict());self.assertEqual(restored.state_dict(),cache.state_dict())
  with self.assertRaises(ValueError):restored.refresh(result(True,True),round_id=3,model_identity='s')
 def test_malformed_generation_rejected(self):
  with self.assertRaises(ValueError):judge(outputs()[:3],'.5')
  cache=MasteryCache('d','p')
  with self.assertRaises(ValueError):cache.load_state_dict(MasteryCache('wrong','p').state_dict())
if __name__=='__main__':unittest.main(verbosity=2)
