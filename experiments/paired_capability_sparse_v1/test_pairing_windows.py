import unittest
from audit_pairing_windows import audit_windows
class WindowTests(unittest.TestCase):
 def test_global_mixed_strata_can_still_give_identical_actual_window_constraints(self):
  # Eight joint-known, eight EN-only, separated by unknown items. The global
  # adequacy counts pass, yet normalized F=E in EVERY actual two-item window.
  en=[1]*8+[0]*6+[1]*8+[0]*7;sw=[1]*8+[0]*21
  result=audit_windows(en,sw,list(reversed(range(29))))
  self.assertEqual(sum(a and b for a,b in zip(en,sw)),8)
  self.assertEqual(sum(a and not b for a,b in zip(en,sw)),8)
  self.assertEqual(result['E_F_different_windows'],0)
  self.assertFalse(result['necessary_weight_identifiability'])
 def test_mixed_adjacent_mastery_changes_window_not_just_global_metadata(self):
  en=[1,1,0,1];sw=[1,0,0,1]
  result=audit_windows(en,sw,[1,2,3,0])
  self.assertGreater(result['E_F_different_windows'],0)
  self.assertGreater(result['F_S_different_windows'],0)
if __name__=='__main__':unittest.main()
