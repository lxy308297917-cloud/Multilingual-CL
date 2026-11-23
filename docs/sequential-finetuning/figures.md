# 顺序微调与 Single 图表

共 29 张原图，包含阶段热力图、单语对照和四张学习曲线。新版矩阵左侧给绝对分数，右侧给相邻阶段分差。框线标注本语阶段，图表保留各自的划分和主指标。

[总览](README.md) · [逐阶段分数 CSV](stage-matrices.csv)

## 综合与 Full-SFT

- [本语学完后的逐阶段保留差（汇总）](figures/retention-after-own.png)
- [Base、Single、Joint 和顺序终点](figures/baseline-endpoints.png)
- [首轮匹配 Single 对照](figures/matched-single-heatmaps.png)
- [XNLI 和 TyDiQA 单语预筛](figures/xnli-tydiqa-single.png)
- [MASSIVE 两学习率全参官测](figures/massive-full-lr-test.png)
- [MASSIVE 首轮全参开发矩阵](figures/massive-full-sft-first-dev.png)

## MASSIVE

- [12 语 Base／Single 开发预筛](figures/massive-12-language-single.png)
- [原顺序 SeqLoRA 首轮开发](figures/massive-first-seq-dev.png)
- [原顺序 Replay 首轮开发](figures/massive-first-replay-dev.png)
- [SW 阶段变化](figures/massive-sw-stages.png)
- [原顺序学习曲线](figures/massive-original-learning-curves.png)
- [按 Single 排序的换序首轮开发](figures/massive-first-swap-dev.png)
- [换序学习曲线](figures/massive-swap-learning-curves.png)
- [官方测试，两顺序与多方法的完整阶段矩阵](figures/massive-test-absolute-delta.png)
- [原顺序官方开发矩阵](figures/massive-dev-absolute-delta.png)
- [交换末两语的官方开发矩阵](figures/massive-swap-dev-absolute-delta.png)
- [种子 43 官测](figures/massive-seed43-test-absolute-delta.png)
- [种子 43 开发](figures/massive-seed43-dev-absolute-delta.png)

## XNLI

- [含英语五语官测](figures/xnli-original-test.png)
- [首轮 SW 阶段曲线](figures/xnli-original-sw-stages.png)
- [无英语两组训练指令开发矩阵](figures/xnli-no-en-dev.png)
- [无英语 SW 官测确认](figures/xnli-no-en-sw-test.png)

## MasakhaNER、TyDiQA、MTOP

- [MasakhaNER 原顺序开发矩阵](figures/masakhaner-first-dev.png)
- [MasakhaNER 两顺序开发子集矩阵](figures/masakhaner-dev-absolute-delta.png)
- [MasakhaNER 两顺序官方测试矩阵](figures/masakhaner-test-absolute-delta.png)
- [TyDiQA 原顺序内部与已有作者开发结果](figures/tydiqa-first-dev.png)
- [TyDiQA 两顺序完整作者开发矩阵](figures/tydiqa-author-dev-absolute-delta.png)
- [MTOP 官方开发完整矩阵](figures/mtop-dev-absolute-delta.png)
- [MTOP 官方测试完整矩阵](figures/mtop-test-absolute-delta.png)
