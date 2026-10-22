import unittest
import torch
from train import learning_rate,per_example_ce
from prepare_calibration import arithmetic

class ContractTests(unittest.TestCase):
    def test_real_formal_schedule_prefix_matches_transformers(self):
        from transformers import get_cosine_schedule_with_warmup
        p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.SGD([p],lr=5e-5)
        scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=2124,num_training_steps=42465)
        for step in range(2048):
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],learning_rate(step),places=16)
            optimizer.step();scheduler.step()
        self.assertGreater(learning_rate(2048),0)
        self.assertAlmostEqual(learning_rate(42465),0.)

    def test_variable_length_assistant_ce_is_sample_balanced(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.emb=torch.nn.Embedding(7,7)
            def forward(self,input_ids,attention_mask,use_cache=False):
                class Output:pass
                value=Output();value.logits=self.emb(input_ids);return value
        torch.manual_seed(42);model=Model()
        rows=[{'input_ids':[1,2,3,4],'labels':[-100,-100,3,4]},{'input_ids':[2,5,6],'labels':[-100,-100,6]}]
        got=per_example_ce(model,rows,pad=0,device='cpu')
        logits=model.emb.weight.log_softmax(-1)
        expected=torch.stack([-(logits[2,3]+logits[3,4])/2,-logits[5,6]])
        torch.testing.assert_close(got,expected)
        target=torch.autograd.grad(got.mean(),model.parameters(),retain_graph=True)
        reference=torch.autograd.grad(expected.mean(),model.parameters())
        torch.testing.assert_close(target[0],reference[0])

    def test_reference_arithmetic_does_not_execute_code(self):
        from fractions import Fraction
        self.assertEqual(arithmetic('20/0.4*(1-0.4-0.3)'),Fraction(15))
        for expression in ["__import__('os')",'2**100','x+1','[1,2]']:
            with self.assertRaises(ValueError):arithmetic(expression)

    def test_full_checkpoint_transport_and_identity_failure(self):
        import tempfile,json
        from pathlib import Path
        from train import save_checkpoint,validate_checkpoint_transport
        from core import SparseProjectedAdamW
        from mastery import MasteryCache
        model=torch.nn.Linear(4,2,bias=False);opt=SparseProjectedAdamW(dict(model.named_parameters()),fraction=.5)
        model.weight.grad=torch.ones_like(model.weight);opt.step(1e-4,[])
        cache=MasteryCache('data','prompt');counts={'input':17,'supervised':4}
        with tempfile.TemporaryDirectory() as temp:
            cp=save_checkpoint(Path(temp),model,opt,1,cache,counts,'protocol','D',4.)
            validate_checkpoint_transport(cp,model,opt,cache,counts)
            receipt=json.loads(Path(cp).with_suffix('.json').read_text())
            self.assertGreaterEqual(receipt['transport_validation_wall_seconds'],0)
            self.assertEqual(opt.steps,1)
            with self.assertRaises(RuntimeError):validate_checkpoint_transport(cp,model,opt,cache,{'input':18,'supervised':4})

if __name__=='__main__':unittest.main(verbosity=2)
