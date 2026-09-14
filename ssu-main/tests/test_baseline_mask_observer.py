import sys,tempfile,unittest
from pathlib import Path
import torch,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'analysis'));sys.path.insert(0,str(R/'training/src'))
from baseline_mask_audit import collect
from utils.model_utils import _make_mask_hook,_make_scale_hook
class Observer(unittest.TestCase):
 def test_mask_matches_actual_gradient_without_modifying_state(self):
  model=torch.nn.Sequential(torch.nn.Linear(8,4),torch.nn.Linear(4,2));model[1].weight.requires_grad=False
  mask=torch.zeros_like(model[0].weight,dtype=torch.bool);mask[:,2:]=True;model[0].weight.register_hook(_make_mask_hook(mask))
  scale=torch.ones_like(model[0].bias);scale[0]=0;model[0].bias.register_hook(_make_scale_hook(scale))
  before={n:p.detach().clone() for n,p in model.named_parameters()};rng=torch.random.get_rng_state()
  with tempfile.TemporaryDirectory() as temp:
   result=collect(model,temp);self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
   self.assertTrue(all(p.grad is None and torch.equal(p,before[n]) for n,p in model.named_parameters()))
   sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
   packed=np.load(Path(temp)/'masks.npz');params=dict(model.named_parameters())
   for row in result['parameters']:
    if row['mode']=='none':self.assertIsNone(params[row['name']].grad);continue
    expected=np.ones(row['shape'],dtype=bool) if row['mode']=='all' else np.unpackbits(packed[row['key']])[:row['elements']].reshape(row['shape']).astype(bool)
    self.assertTrue(np.array_equal(expected,(params[row['name']].grad!=0).numpy()))
if __name__=='__main__':unittest.main()
