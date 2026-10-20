"""Meaningful CPU invariants for the research prototype."""
import copy
import unittest

import torch

from core import (Rollout, RiskAwareSparseAdamW, answer_logits, collect_rollout,
                  exact_mask, state_kl)
from demo_cpu import TinyCausalLM, run


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.rollout = Rollout(torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.bool), 0)

    def test_kl_value_grad_and_teacher_detach(self):
        s = torch.tensor([[[.2, -.3], [.5, .1]]], requires_grad=True)
        t = torch.tensor([[[.1, .7], [-.2, .8]]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        loss = state_kl(s, t, mask)
        loss.backward()
        expected = s.detach()[0, 0].softmax(-1) - t.detach()[0, 0].softmax(-1)
        torch.testing.assert_close(s.grad[0, 0], expected)
        self.assertTrue(torch.equal(s.grad[0, 1], torch.zeros(2)))
        self.assertIsNone(t.grad)

    def test_identical_distributions(self):
        x = torch.randn(2, 3, 8, requires_grad=True)
        valid = torch.ones(2, 3, dtype=torch.bool)
        loss = state_kl(x, x.detach(), valid)
        self.assertLess(abs(float(loss)), 1e-7)
        loss.backward()
        self.assertLess(float(x.grad.abs().max()), 1e-7)

    def test_reverse_kl_gradient_finite_difference(self):
        x = torch.tensor([[[.2, -.1, .8]]], requires_grad=True)
        teacher = torch.tensor([[[.7, .1, -.5]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        loss = state_kl(x, teacher, valid, direction='reverse')
        analytical = torch.autograd.grad(loss, x)[0]
        for i in range(3):
            plus, minus = x.detach().clone(), x.detach().clone()
            plus[0, 0, i] += .001
            minus[0, 0, i] -= .001
            numerical = (state_kl(plus, teacher, valid, direction='reverse') -
                         state_kl(minus, teacher, valid, direction='reverse')) / .002
            torch.testing.assert_close(numerical, analytical[0, 0, i], rtol=1e-3, atol=1e-4)

    def test_exact_budget_and_stable_ties(self):
        mask = exact_mask(torch.ones(4, 5), .15)
        self.assertEqual(int(mask.sum()), 3)
        self.assertEqual(mask.flatten().nonzero().flatten().tolist(), [0, 1, 2])
        self.assertEqual(int(exact_mask(-torch.ones(4, 5), .15).sum()), 3)

    def test_sampling_reproducible_and_restores_modes(self):
        model = TinyCausalLM()
        model.train()
        model.output.eval()
        prompt = torch.tensor([[1, 2], [2, 3]])
        a = collect_rollout(model, prompt, version=0, max_new_tokens=4,
                            generator=torch.Generator().manual_seed(4))
        b = collect_rollout(model, prompt, version=0, max_new_tokens=4,
                            generator=torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(a.tokens, b.tokens))
        self.assertTrue(model.training)
        self.assertFalse(model.output.training)
        self.assertFalse(a.tokens.requires_grad)

    def test_teacher_prefix_alignment(self):
        model = TinyCausalLM()
        prompt = torch.tensor([[1, 2]])
        got = answer_logits(model, prompt, self.rollout)
        expected = model(torch.tensor([[1, 2, 1, 2]]))[:, 1:3]
        torch.testing.assert_close(got, expected)
        self.assertEqual(answer_logits(model, torch.tensor([[3, 1, 2]]), self.rollout).shape, got.shape)

    def test_unselected_and_weight_decay_do_not_write(self):
        p = torch.nn.Parameter(torch.ones(4, 5))
        opt = RiskAwareSparseAdamW({'w': p}, lr=.1, wd=.2, fraction=.15, risk_weight=0)
        before = p.detach().clone()
        result = opt.step({'w': torch.ones_like(p)}, {'w': torch.zeros_like(p)},
                          {'w': torch.zeros_like(p)}, new_rollout=self.rollout, old_rollout=self.rollout)
        self.assertEqual(result['matrices']['w']['selected'], 3)
        self.assertEqual(int((p != before).sum()), 3)
        torch.testing.assert_close(p.flatten()[3:], before.flatten()[3:], rtol=0, atol=0)

    def test_risk_changes_selection(self):
        p = torch.nn.Parameter(torch.zeros(2, 2))
        opt = RiskAwareSparseAdamW({'w': p}, fraction=.25, lr=.1, wd=0, anchor_weight=0)
        old = torch.tensor([[-10., 0.], [0., 0.]])
        opt.step({'w': torch.ones_like(p)}, {'w': old}, {'w': torch.zeros_like(p)},
                 new_rollout=self.rollout, old_rollout=self.rollout)
        self.assertEqual(float(p[0, 0]), 0.)
        self.assertLess(float(p[0, 1]), 0.)

    def test_curvature_proxy_protects_when_anchor_gradient_zero(self):
        p = torch.nn.Parameter(torch.zeros(2, 2))
        opt = RiskAwareSparseAdamW({'w': p}, fraction=.25, lr=.1, wd=0, anchor_weight=0)
        sens = torch.tensor([[100., 0.], [0., 0.]])
        opt.step({'w': torch.ones_like(p)}, {'w': torch.zeros_like(p)}, {'w': sens},
                 new_rollout=self.rollout, old_rollout=self.rollout)
        self.assertEqual(float(p[0, 0]), 0.)

    def test_stale_rejected_without_mutation(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        opt = RiskAwareSparseAdamW({'w': p})
        x = {'w': torch.ones_like(p)}
        opt.step(x, x, x, new_rollout=self.rollout, old_rollout=self.rollout)
        before, state = p.detach().clone(), opt.state_dict()
        with self.assertRaises(ValueError):
            opt.step(x, x, x, new_rollout=self.rollout, old_rollout=self.rollout)
        torch.testing.assert_close(p, before, rtol=0, atol=0)
        torch.testing.assert_close(opt.m['w'], state['m']['w'], rtol=0, atol=0)

    def test_invalid_signal_rejected_atomically(self):
        a, b = torch.nn.Parameter(torch.ones(2, 2)), torch.nn.Parameter(torch.ones(2, 2))
        opt = RiskAwareSparseAdamW({'a': a, 'b': b})
        good = {'a': torch.ones_like(a), 'b': torch.ones_like(b)}
        bad = {**good, 'b': torch.full_like(b, float('nan'))}
        with self.assertRaises(FloatingPointError):
            opt.step(good, bad, good, new_rollout=self.rollout, old_rollout=self.rollout)
        self.assertEqual(opt.version, 0)
        self.assertTrue(torch.equal(a, torch.ones_like(a)))
        self.assertTrue(torch.equal(opt.m['a'], torch.zeros_like(a)))

    def test_resume_exact_next_update(self):
        p = torch.nn.Parameter(torch.zeros(4, 5))
        opt = RiskAwareSparseAdamW({'w': p}, lr=.1)
        signals = {'w': torch.ones_like(p)}
        opt.step(signals, signals, signals, new_rollout=self.rollout, old_rollout=self.rollout)
        q = torch.nn.Parameter(p.detach().clone())
        restored = RiskAwareSparseAdamW({'w': q}, lr=.1)
        restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        fresh = Rollout(self.rollout.tokens, self.rollout.valid, 1)
        for optimizer in (opt, restored):
            optimizer.step(signals, signals, signals, new_rollout=fresh, old_rollout=fresh)
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(opt.m['w'], restored.m['w'], rtol=0, atol=0)

    def test_two_stage_demo(self):
        result = run(steps=2)
        self.assertEqual(result['steps'], 4)
        self.assertFalse(result['gpu_used'])
        self.assertEqual({r['stage'] for r in result['trace']}, {1, 2})

    def test_eos_included_and_later_tokens_masked(self):
        class Fixed(torch.nn.Module):
            def forward(self, ids):
                out = torch.full((*ids.shape, 3), -1000.)
                out[..., 1] = 1000.
                return out
        rollout = collect_rollout(Fixed(), torch.tensor([[0]]), version=0,
                                  max_new_tokens=4, eos_id=1)
        self.assertEqual(rollout.tokens.tolist(), [[1]])
        self.assertEqual(rollout.valid.tolist(), [[True]])


if __name__ == '__main__':
    unittest.main()
