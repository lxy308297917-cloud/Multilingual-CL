# On-policy risk-aware sparse continual learning — CPU prototype

**Status: proposed method, CPU mechanism tests only. No Qwen training, no GPU experiment, no downstream gains, no accepted novelty claim.** Date: 2026-10-09.

Start with [Chinese method proposal](../../docs/research/on_policy_sparse_cl/METHOD_ZH.md), [experiment plan](../../docs/research/on_policy_sparse_cl/EXPERIMENT_PLAN_ZH.md), and [ChatGPT review brief](../../docs/research/on_policy_sparse_cl/CHATGPT_REVIEW_BRIEF_ZH.md).

## Files and reading order

1. `core.py`: fresh student sampling; answer alignment; forward/reverse local KL; exact per-matrix quotas; utility/risk selection using proposed AdamW deltas; optimizer state restore.
2. `demo_cpu.py`: two synthetic stages with a demonstration-context teacher, a frozen Base reference and a previous-stage reference. Teacher context is a synthetic token, not evidence of teaching quality.
3. `test_core.py`: 14 CPU invariants. Requires only existing PyTorch, no transformers/model/data downloads.
4. `example_config.json`: illustrative proposal, intentionally `launchable=false`.

```bash
cd /data0/users/lixiaoyang/Multilingual-CL
PYTHONDONTWRITEBYTECODE=1 /data0/users/lixiaoyang/ssu-migration/envs/cl/bin/python experiments/on_policy_sparse_cl_v1/test_core.py
PYTHONDONTWRITEBYTECODE=1 /data0/users/lixiaoyang/ssu-migration/envs/cl/bin/python experiments/on_policy_sparse_cl_v1/demo_cpu.py --steps-per-stage 8
```

These commands use CPU tensors only. The demo's local KL values are not reading, translation, IFEval or math scores. The prototype is independent of all frozen historical training releases.

## Integration boundary

`collect_rollout` currently accepts unpadded equal-length prompts and a model accepting positional token tensors; the toy model is CPU only. A Hugging Face adapter, variable-length attention masks, model identity hashes, task manifests, teacher-quality gates, Qwen integration, scheduler/gradient-clipping parity, AMP/distributed accumulation and complete training resume are **not implemented**. Optimizer resume tests do not establish end-to-end training resume.

Freshness versions must track every student weight change. They detect stale recorded versions, not forged provenance or externally changed weights. No top-p/top-k truncation or non-unit sampling temperature is allowed in the prototype. A production rollout ledger must record exact weights, RNG and generation settings.

Local KL gradients treat sampled prefixes as fixed. This is an on-policy-state **semi-gradient surrogate**, not a proven unbiased gradient of full sequence KL or a policy-gradient/PPO trainer. Teacher gradients are detached; generated student tokens are not trained as automatically correct hard labels.
