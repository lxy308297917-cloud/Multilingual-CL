import copy
import random
import unittest
import numpy as np
import torch
from scipy.optimize import minimize
from core import calibration_context, pair_weights, project_vector, solve_dual, SparseProjectedAdamW

class ProjectionTests(unittest.TestCase):
    def test_single_formula(self):
        z=torch.tensor([2.,3.]); a=torch.tensor([[1.,0.]])
        d,l=project_vector(z,a)
        torch.testing.assert_close(d,torch.tensor([0.,3.],dtype=torch.float64))
        self.assertEqual(l.item(),2.)

    def test_feasible_is_identity(self):
        z=torch.tensor([-2.,-3.]);d,l=project_vector(z,torch.eye(2))
        self.assertTrue(torch.equal(d,z.double())); self.assertEqual(l.sum(),0.)

    def test_joint_not_one_pass(self):
        a=torch.tensor([[1.,0.],[-1.,1.]])
        d,_=project_vector(torch.tensor([2.,3.]),a)
        torch.testing.assert_close(d,torch.zeros(2,dtype=torch.float64),atol=1e-10,rtol=0)

    def test_dependent_and_zero(self):
        a=torch.tensor([[1.,0.],[2.,0.],[0.,0.],[-1.,0.]])
        d,_=project_vector(torch.tensor([2.,3.]),a)
        torch.testing.assert_close(d,torch.tensor([0.,3.],dtype=torch.float64),atol=1e-10,rtol=0)

    def test_random_qp_reference(self):
        rng=np.random.default_rng(42)
        for _ in range(40):
            a=rng.normal(size=(3,9));z=rng.normal(size=9);eps=rng.random(3)*.1
            d,l=project_vector(torch.tensor(z),torch.tensor(a),torch.tensor(eps))
            ref=minimize(lambda v:.5*np.sum((v-z)**2),np.zeros(9),jac=lambda v:v-z,
                         constraints={'type':'ineq','fun':lambda v:eps-a@v,'jac':lambda v:-a},
                         method='SLSQP',options={'ftol':1e-12,'maxiter':200})
            self.assertTrue(ref.success)
            np.testing.assert_allclose(d.numpy(),ref.x,atol=1e-6)
            self.assertLessEqual(float((torch.tensor(a)@d-torch.tensor(eps)).max()),1e-8)

    def test_rescaling(self):
        a=torch.tensor([[1.,2.],[-2.,1.]],dtype=torch.float64);z=torch.tensor([3.,2.])
        d,_=project_vector(z,a);e,_=project_vector(z,a*torch.tensor([1e-10,1e10])[:,None])
        torch.testing.assert_close(d,e,atol=1e-8,rtol=0)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):solve_dual(torch.eye(2),torch.ones(2),-torch.ones(2))
        with self.assertRaises(ValueError):solve_dual(-torch.eye(2),torch.ones(2),torch.zeros(2))

class OptimizerTests(unittest.TestCase):
    def test_adamw_equivalence_full_support(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float32));q=torch.nn.Parameter(p.detach().clone())
        o=SparseProjectedAdamW({'p':p},fraction=1,bf16_policy='record');r=torch.optim.AdamW([q],lr=.002,weight_decay=.01)
        for i in range(5):
            p.grad=torch.tensor([i+1.,-i-2.],dtype=p.dtype);q.grad=p.grad.clone()
            o.step(.002);r.step()
            torch.testing.assert_close(p,q,atol=1e-6,rtol=0)

    def test_support_and_moment_once(self):
        p=torch.nn.Parameter(torch.ones(8));p.grad=torch.arange(1.,9.)
        o=SparseProjectedAdamW({'p':p},fraction=.5)
        original=p.detach().clone();o.step(.01,[{'p':-torch.ones(8)}])
        self.assertTrue(torch.equal(p[~o.masks['p']],original[~o.masks['p']]))
        torch.testing.assert_close(o.state['p']['m'],p.grad*.1)
        torch.testing.assert_close(o.state['p']['v'],p.grad.square()*.001)

    def test_invalid_guard_no_state_advance(self):
        p=torch.nn.Parameter(torch.ones(4));p.grad=torch.ones(4);o=SparseProjectedAdamW({'p':p})
        with self.assertRaises(ValueError):o.step(.01,[{}])
        self.assertEqual(o.steps,0);self.assertFalse(o.state)

    def test_resume_exact(self):
        p=torch.nn.Parameter(torch.randn(16,dtype=torch.bfloat16));o=SparseProjectedAdamW({'p':p},fraction=.5,mask_interval=2)
        p.grad=torch.arange(16,dtype=p.dtype);o.step(.001)
        state=copy.deepcopy(o.state_dict());q=torch.nn.Parameter(p.detach().clone());r=SparseProjectedAdamW({'p':q},fraction=.5,mask_interval=2)
        r.load_state_dict(state)
        for _ in range(3):
            p.grad=torch.arange(16,dtype=p.dtype);q.grad=p.grad.clone()
            self.assertEqual(o.step(.001),r.step(.001));self.assertTrue(torch.equal(p,q))
            self.assertTrue(torch.equal(o.state['p']['m'],r.state['p']['m']))
        bad=copy.deepcopy(state);bad['state']['p']['m']=bad['state']['p']['m'].bfloat16()
        with self.assertRaises(ValueError):r.load_state_dict(bad)

    def test_finite_reduction_detects_nan_and_both_infinities(self):
        from core import _all_finite
        for dtype in (torch.float32,torch.bfloat16,torch.float64):
            self.assertTrue(_all_finite(torch.tensor([-1.,0.,2.],dtype=dtype)))
            for value in (float('nan'),float('inf'),-float('inf')):
                self.assertFalse(_all_finite(torch.tensor([1.,value,2.],dtype=dtype)))
        self.assertTrue(_all_finite(torch.empty(0)))

    def test_resume_metadata_does_not_materialize_full_state(self):
        from unittest.mock import patch
        p=torch.nn.Parameter(torch.ones(16));p.grad=torch.arange(16.)
        opt=SparseProjectedAdamW({'p':p},fraction=.5);opt.step(.01)
        state=copy.deepcopy(opt.state_dict())
        with patch.object(opt,'state_dict',side_effect=AssertionError('Expensive full snapshot')):
            opt.load_state_dict(state)

    def test_cached_index_matches_verified_boolean_reference(self):
        import importlib.util
        from pathlib import Path
        path=Path(__file__).resolve().parents[2]/'docs/research/paired_capability_sparse/verified_v3/core.py'
        spec=importlib.util.spec_from_file_location('verified_boolean_core',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        p=torch.nn.Parameter(torch.linspace(-1,1,32,dtype=torch.bfloat16));q=torch.nn.Parameter(p.detach().clone())
        new=SparseProjectedAdamW({'p':p},fraction=.5,mask_interval=3,bf16_policy='record')
        old=module.SparseProjectedAdamW({'p':q},fraction=.5,mask_interval=3,bf16_policy='record')
        gen=torch.Generator().manual_seed(42)
        for step in range(12):
            p.grad=torch.randn(32,generator=gen).bfloat16();q.grad=p.grad.clone()
            guards=[{'p':torch.randn(32,generator=gen)} for _ in range(2)]
            self.assertEqual(new.step(.005,guards),old.step(.005,guards))
            self.assertTrue(torch.equal(p,q));self.assertTrue(torch.equal(new.masks['p'],old.masks['p']))
            for key in ('m','v'):self.assertTrue(torch.equal(new.state['p'][key],old.state['p'][key]))

    def test_reset_to_step_zero_clears_cached_state(self):
        p=torch.nn.Parameter(torch.ones(16));o=SparseProjectedAdamW({'p':p},fraction=.5)
        fresh=copy.deepcopy(o.state_dict());p.grad=torch.arange(16.)
        o.step(.01);o.load_state_dict(fresh)
        self.assertEqual(o.steps,0);self.assertFalse(o.state);self.assertFalse(o.masks);self.assertFalse(o.indices)
        o.step(.01);torch.testing.assert_close(o.state['p']['m'],p.grad*.1)

    def test_resume_reuses_existing_moment_and_mask_storage(self):
        p=torch.nn.Parameter(torch.ones(16));p.grad=torch.arange(16.)
        o=SparseProjectedAdamW({'p':p},fraction=.5);o.step(.01)
        state=copy.deepcopy(o.state_dict())
        pointers=[o.state['p'][k].data_ptr() for k in ('m','v')]+[o.masks['p'].data_ptr()]
        o.load_state_dict(state)
        self.assertEqual(pointers,[o.state['p'][k].data_ptr() for k in ('m','v')]+[o.masks['p'].data_ptr()])

    def test_bf16_actual_constraint_or_rejection(self):
        p=torch.nn.Parameter(torch.linspace(-1,1,100,dtype=torch.bfloat16));p.grad=torch.linspace(-2,1,100,dtype=p.dtype)
        o=SparseProjectedAdamW({'p':p},fraction=.5);before=p.detach().float().clone();g=torch.linspace(-1,2,100)
        s=o.step(.005,[{'p':g}]);actual=float(g.double()@(p.detach().float()-before).double())
        self.assertLessEqual(actual,1e-12);self.assertAlmostEqual(s['actual_residual'][0],actual,places=10)

class PairAndCalibrationTests(unittest.TestCase):
    def test_pair_shuffle_changes_f_not_e(self):
        en=[1,1,0,0];sw=[1,0,1,0];perm=[1,0,3,2]
        e=pair_weights(en,sw);es=pair_weights(en,sw,permutation=perm)
        f=pair_weights(en,sw,'paired');fs=pair_weights(en,sw,'paired',perm)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(e,es)))
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(f,fs)))
        self.assertEqual(f[0][2:].sum(),0.)

    def test_degenerate_pairing_is_disclosed(self):
        f=pair_weights([1,1],[1,1],'paired');e=pair_weights([1,1],[1,1])
        self.assertTrue(all(torch.equal(x/x.sum(),y/y.sum()) for x,y in zip(f,e)))

    def test_rng_modes_and_gradients_preserved_even_exception(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.5));model.train();model[0].eval()
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        for p in model.parameters():p.grad=torch.ones_like(p)
        ts=torch.get_rng_state().clone();ns=copy.deepcopy(np.random.get_state());ps=random.getstate()
        with self.assertRaises(RuntimeError):
            with calibration_context(model):
                random.random();np.random.rand();torch.rand(4)
                torch.autograd.grad(model(torch.ones(1,2)).sum(),list(model.parameters()))
                raise RuntimeError('intentional')
        self.assertTrue(torch.equal(ts,torch.get_rng_state()));self.assertEqual(ps,random.getstate())
        self.assertTrue(np.array_equal(ns[1],np.random.get_state()[1]));self.assertTrue(model.training);self.assertFalse(model[0].training)
        self.assertTrue(all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters()))

if __name__=='__main__':unittest.main(verbosity=2)
