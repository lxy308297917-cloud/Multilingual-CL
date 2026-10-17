# Related-work boundary (checked 2026-10-09)

All descriptions below are scope summaries, not reproduction or empirical guarantees for Qwen1.5B. Published abstracts and applicable method sections were read. The proposed innovation remains provisional.

| Primary source | Existing contribution relevant here | Boundary for this proposal |
|---|---|---|
| [Shenfeld et al., Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) | Demonstration-conditioned teacher supervision on student trajectories, including sequential skill acquisition | On-policy continual distillation itself is established; distinguish forward practical loss, reverse theory, and teacher update choices |
| [Yu et al., Dense Supervision, Sparse Updates](https://arxiv.org/abs/2606.13657) | OPD updates exhibit coordinate sparsity at checkpoint precision; masked training on identified subnetworks is studied | Sparse OPD is established. Our online, signed candidate-update risk selection must beat a faithful sparse OPD comparator |
| [iSDFT](https://arxiv.org/abs/2609.24646) | Information-proximal teacher influence and a frozen initial-policy anchor | Teacher moderation and Base KL cannot be claimed novel; parameter-space selection is the narrower hypothesis |
| [Chen et al., On-Policy Replay](https://arxiv.org/abs/2605.29495) | Latest-checkpoint historical-prompt generation with task filtering and ordinary SFT replay | Include on-policy replay and vanilla replay where task rewards/data permit |
| [Uemura & Suzuki, Learning without Overwriting](https://arxiv.org/abs/2610.05200) | Theory for OPSD and continual reasoning in a graph model, with hint-induced sparse updates | No direct no-forgetting guarantee for real AdamW, multilingual tasks or our risk proxy |

Next literature audit: GAST/gradient-alignment sparse tuning; Fisher/EWC and update-space saliency; continual sparse subnet allocation; OPD rollout freshness and KL gradient estimators. No novelty claim until the closest parameter-selection algorithms are compared explicitly.

The full-vocabulary local KL in `core.py` stops gradients through teacher outputs and sampled prefix occupancy. Do not call it a complete unbiased sequence-KL estimator. Check teacher/reference provenance, response alignment, token weighting and sampling configuration before comparing objectives.
