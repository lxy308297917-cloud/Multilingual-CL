# Paired capability constraints: minimal research prototype

Specification and evidence: [METHOD_ZH.md](../../docs/research/paired_capability_sparse/METHOD_ZH.md), [source audit](../../docs/research/paired_capability_sparse/SOURCE_AUDIT.json), [related work](../../docs/research/paired_capability_sparse/RELATED_WORK.md).

- `core.py`: FP32-moment AdamW, exact element quotas, global support-constrained joint projection, staged BF16 write checks, CPU constraint staging, optimizer state serialization, pairing weights and calibration RNG/mode isolation.
- `mastery.py`: bounded numeric-equivalence parser and conservative Base/acquired mastery cache with immutable protection floor and acquisition hysteresis.
- `test_core.py`, `test_mastery.py`: 23 numerical/state/pairing tests; independent SciPy constrained-optimization comparisons included.
- `smoke_qwen.py`: four real-model technical steps, actual Aya/translation examples (two longest first), two cross-input-language gold-only CE gradients, full checkpoint and exact resumed-step comparison. This is not a whole-solution CE or paired-method efficacy test.
- `run_smoke.sh`: one manually started, single-GPU queue entry. Do not rerun into an existing output directory. No auto-start/restart.
- `preflight.py`: read-only scientific/resource release report; never launches training.
- `stress_qwen.py`: one longest-sample update with full optimizer state already resident; `run_fast_validation.sh` compares cached-index implementation against the validated numerical path.
- `plan.json`: technical release and proposed comparison budget/gates. The comparison is not released; no formal tests are launched.

Run CPU checks with the existing project environment:

```bash
OMP_NUM_THREADS=4 /data0/users/lixiaoyang/ssu-migration/envs/cl/bin/python -m unittest discover -s experiments/paired_capability_sparse_v1 -p 'test_*.py'
```

The current prototype does **not** claim an end-to-end released comparative trainer: pool QA, grouped decontamination, full-reference/proxy audit, exact dev protocol, storage capacity, and a matched exposure/generation schedule remain prerequisites. Do not turn its CPU tests or technical short run into method results. The gold-only calibration in the technical check deliberately has no mastery claim; it validates gradient separation and write semantics.
