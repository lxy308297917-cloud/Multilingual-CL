import unittest
from common import *
class Test(unittest.TestCase):
 def test_native_throughput_receipt(self):
  value=throughput_receipt([99.,.7,.9],4293,10800);decoded=json.loads(json.dumps(value));self.assertIs(type(decoded['passed']),bool);self.assertTrue(decoded['passed']);self.assertAlmostEqual(decoded['projected_seconds'],3434.4);self.assertFalse(throughput_receipt([99.,9.,9.],4293,10800)['passed'])
 def test_fixed_bank_microbatch_equivalence(self):
  torch.manual_seed(9);v=torch.randn(8,12,requires_grad=True);bank=torch.randn(20,12);idx=torch.arange(8)
  for kind in ('C','D'):
   whole=torch.autograd.grad(auxiliary(v,bank,idx,kind),v,retain_graph=True)[0];chunked=sum(auxiliary(v[i:i+4],bank,idx[i:i+4],kind)*.5 for i in (0,4));parts=torch.autograd.grad(chunked,v,retain_graph=True)[0];torch.testing.assert_close(whole,parts,atol=1e-6,rtol=1e-5)
 def test_padding(self):
  h=torch.arange(24.).reshape(2,3,4);m=torch.tensor([[1,1,0],[1,1,1]])
  for method in ('mean','last','weighted'):
   a=pool(h,m,method);b=pool(torch.cat([h,torch.ones(2,5,4)*999],1),torch.cat([m,torch.zeros(2,5)],1),method);torch.testing.assert_close(a,b)
 def test_strict_ties(self):
  self.assertEqual(retrieval(torch.ones(3,4),torch.ones(3,4))['mexa_strict'],0.)
  self.assertEqual(retrieval(torch.eye(4),torch.eye(4))['mexa_strict'],1.)
 def test_pairing_changes_gradient(self):
  torch.manual_seed(42);bank=torch.randn(8,9);v=torch.randn(3,9,requires_grad=True)
  for kind in ('C','D'):
   a=torch.autograd.grad(auxiliary(v,bank,torch.tensor([0,1,2]),kind),v,retain_graph=True)[0];b=torch.autograd.grad(auxiliary(v,bank,torch.tensor([1,2,0]),kind),v,retain_graph=True)[0];self.assertGreater(float((a-b).norm()),.01)
 def test_per_example_ce(self):
  x=torch.randn(2,4,7);l=torch.tensor([[-100,1,2,-100],[-100,3,2,4]]);v=ce(x,l);expected=(F.cross_entropy(x[0,:2],l[0,1:3])+F.cross_entropy(x[1,:3],l[1,1:]))/2;torch.testing.assert_close(v,expected)
if __name__=='__main__':unittest.main()
