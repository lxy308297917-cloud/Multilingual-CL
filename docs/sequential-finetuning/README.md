# 多语言顺序微调实验进展

[多语言顺序学习 V2 实验汇总](v2/README_ZH.md)包含三个五语主基准的六组 baseline、阶段矩阵、Single 对照、热力图和配对区间。


汇总范围为首轮 XNLI 单卡试跑、MASSIVE 和 XNLI 的改版实验，以及截至 2026 年 11 月 22 日北京时间 10:00 的基准扩展和基线复验。共完成五个基准的顺序训练与阶段评估，另完成 Belebele 数据可行性审计。各基准自己的语言和顺序见下表。

## 基准、顺序与完成范围

| 基准及主指标 | 完成的顺序训练 | 最完整的评估覆盖 | 已有匹配对照 |
| --- | --- | --- | --- |
| **MASSIVE 1.1** 槽位微 F1 | DE → AR → VI → SW → UR，以及 DE → AR → VI → UR → SW。两序都有 SeqLoRA、Replay 和 Full-SFT `5e−5`。原序另有 Full-SFT `1e−5`、递减学习率 SeqLoRA、种子 43 的 SeqLoRA／Replay | 每条完整轨迹官方开发 2,033 题／语、官方测试 2,974 题／语。五语各 30 格，含 Base | Base、五语 Single LoRA、Joint LoRA、五语 Single Full-SFT 和 Joint Full-SFT。全参 Single 和 Joint 使用 `5e−5` |
| **XNLI** 三分类准确率 | 含英语 EN → DE → AR → SW → ZH 的 SeqLoRA 与 Replay。无英语 DE → AR → VI → SW → UR 的两条 SeqLoRA 分别用本地和英文训练指令 | 含英语五语官测 5,010 题／语，另用本地提示复测官方开发 2,490 题／语。无英语有每语 600 题的六语开发矩阵，官测确认覆盖 SW 5,010 题 | 含英语五语 Base、Single、Joint。无英语八语 Base／Single 开发预筛及 SW Base／Single 官测 |
| **MasakhaNER 2.0** 实体微 F1 | HAU → YOR → SWA → LUG，以及 HAU → YOR → LUG → SWA | 两序各 20 格，涵盖四语官方测试。另有官方开发子集，每语 384 句 | 四语 Base、Single LoRA、同预算 Joint LoRA |
| **TyDiQA-GoldP** 答案 F1 | TE → ID → RU → SW → AR，以及 TE → ID → AR → SW → RU | 两序各 30 格，覆盖五语完整作者开发集，每语 499–921 题。原序另有训练侧内部开发集 | 五语 Base、Single LoRA、同预算 Joint LoRA |
| **MTOP** 顶层平面槽位微 F1 | DE → FR → ES → HI → TH 的 SeqLoRA | 官方开发和测试各 30 格。官方测试每语 2,765–3,549 题，另评五语同题 Single | 五语 Base 和 Single LoRA |
| **Belebele** 四选一阅读理解 | 数据重建、五语 NLLB-200 试译、结构和语义筛查、独立审阅材料准备、上游评分器连通检查 | 重建后保留英文训练 66,595 题、开发 3,768 题。200 道源题用于五语试译，147 项优先译题完成结构化语义复查 | 当前停在独立双语验收，正式 Base／Single 和顺序训练待译题验收后开展 |

主模型是 Qwen3-4B Base。Single 每次从同一底座单独学一门语言。SeqLoRA 在阶段间继承适配器和 AdamW 状态。Replay 在相同总更新和词元预算内，以约 10% 的已学语言输入词元替换当前语训练样本。Joint 用同一批多语训练题混合训练后取终点。Full-SFT 更新全参。XNLI 含英语协议每语 10,000 条、625 步。MASSIVE、XNLI 无英语和 TyDiQA 每语 2,000 条、125 步。MasakhaNER 和 MTOP 每语 1,024 条、64 步。不同协议的训练题、提示、划分和阶段结果分别保留。

每个阶段矩阵中，行是检查点，列是评估语言。用同语同题计算本语训练前后差和学完本语后到任一后续阶段的差，并记录本语检查点是否是全程最高。表中“明确”以同题配对区间排除零为准，后续回落按单语可观测阶段数校正。末语计入习得，遗忘分母只包含有后续观测的语言。终点和反向迁移（BWT）均按语言等权。BWT 只取非末语终点减学完本语时的分数。题级区间反映固定模型的题目抽样波动。 [各基准提示、数据和评分](protocols.md)

## 首轮到补齐的时间线

1. **XNLI 首轮。** 单卡试跑核对显存、吞吐及适配器加载，随后完成含英语的 SeqLoRA、Replay、Joint、五个 Single 和正式评估。SW 的 SeqLoRA 本语阶段为 −2.59 个百分点，再学中文后为 +3.95。Replay 分别为 −1.82 和 +3.11。后续用本地推理指令复测旧权重，又训练无英语的双指令对照。无英语 SW 官测中，本地训练指令为 +2.12 后 −5.05，英文训练指令为 +8.62 后 −3.63。在原序全语官测，SeqLoRA 有 4/5 明确习得、1/4 同时明确回落。无英语官测的阶段确认目前限于 SW。 [原顺序官测矩阵](figures/xnli-original-test.png) · [无英语开发矩阵](figures/xnli-no-en-dev.png) · [SW 官测确认](figures/xnli-no-en-sw-test.png)
2. **MASSIVE 从单语预筛到两种正式顺序。** 12 语同题开发预筛的意图准确率、槽位 F1、整句匹配均提高。这里另有复用顺序训练数据的五个匹配 Single。按五个匹配 Single 的开发槽位 F1 从高到低排列后，得到 DE → AR → VI → UR → SW。两序的 SeqLoRA、Replay 与共享 Joint 均已在固定官方题组评估。两条 SeqLoRA 合看，AR、VI、SW、UR 四语至少一次明确回落，DE 多次出现后续正迁移。 [原序／换序官测矩阵](figures/massive-test-absolute-delta.png) · [12 语预筛](figures/massive-12-language-single.png)
3. **扩展到不同任务。** TyDiQA 先做五语内部开发预筛，再完成两序作者开发矩阵与 Joint。MasakhaNER 完成两个四语顺序、匹配 Single 与 Joint。MTOP 完成五语 Base／Single 预筛和官方开发／测试顺序矩阵，补充了一个没有 SW 的独立语义解析数据集。Belebele 模型筛查将三道人工已确认误译的 SW 题全标为保留，目前保留在译题审计。 [MasakhaNER 官测](figures/masakhaner-test-absolute-delta.png) · [TyDiQA 作者开发](figures/tydiqa-author-dev-absolute-delta.png) · [MTOP 官测](figures/mtop-test-absolute-delta.png)
4. **校准全参基线并复验回放。** MASSIVE 补齐两序 `5e−5` Full-SFT、匹配的高学习率 Single 和 Joint，以及原序 `1e−5` Full-SFT。完成递减学习率的 LoRA 对照、原序第二种子、普通与平行回放的等预算对照、共同有效输出诊断。逐题复核通过的最新批次共有 28 条方法／划分矩阵、800 个阶段评分格和 76 个基线评分格。 [全参学习率热力图](figures/massive-full-lr-test.png) · [种子 43 官测](figures/massive-seed43-test-absolute-delta.png)

## Single 与各方法终点

每行使用一个固定划分和主指标，单位为百分数。Base 是预训练底座，同语 Single 列为每语独立训练后测该语的等权宏平均。“／”依次表示原序和换序。同一行的不同方法可比较，跨行的不同任务指标分别阅读。

| 基准及划分 | Base | Single LoRA | SeqLoRA | Replay | Joint LoRA |
| --- | ---: | ---: | ---: | ---: | ---: |
| MASSIVE 官方测试，槽位 F1 | 13.56 | 53.81 | 61.05／61.78 | 61.81／62.09 | 63.85 |
| XNLI 含英语官方测试，准确率 | 67.23 | 78.52 | 79.81 | 79.23 | 78.95 |
| MasakhaNER 官方测试，实体 F1 | 0.00 | 54.57 | 61.38／65.36 | — | 68.89 |
| TyDiQA 作者开发，答案 F1 | 73.14 | 87.04 | 85.88／87.52 | — | 87.53 |
| MTOP 官方测试，平面槽位 F1 | 4.56 | 39.77 | 60.12 | — | — |

MASSIVE 全参 `5e−5` 的 Single 为 **44.26%**，Joint **49.86%**，顺序终点原序 **28.41%**、换序 **27.27%**。仅降低原顺序全参学习率至 `1e−5`，终点为 **62.66%**，比原序 SeqLoRA 高 **1.61** 个百分点。五个语言仍均有明确本语增益，非末语中 AR 和 SW 明确回落。原序 SeqLoRA 的递减学习率对照终点为 **60.16%**，比常数方案低 **0.89** 个百分点。 [逐语 Single 和基线表](single-and-baselines.md) · [基线热力图](figures/baseline-endpoints.png)

## 逐语证据和剩余空缺

| 顺序及划分 | 明确本语习得 | 本语阶段为峰值 | 同时明确习得和后续回落 |
| --- | ---: | ---: | ---: |
| MASSIVE 全参 `5e−5`，两个官测顺序各自 | 5/5 | 5/5 | 4/4 |
| MASSIVE 全参 `1e−5`，原序官测 | 5/5 | 3/5 | 2/4 |
| MASSIVE SeqLoRA，原序／换序官测 | 5/5／5/5 | 3/5／3/5 | 3/4／2/4 |
| MasakhaNER，原序／换序官测 | 4/4／4/4 | 3/4／3/4 | 2/3／1/3 |
| TyDiQA，原序／换序作者开发 | 5/5／5/5 | 3/5／3/5 | 2/4／3/4 |
| MTOP，官测 | 5/5 | 3/5 | 2/4 |

- **MASSIVE 的逐语形状最清楚。** 两个高学习率全参顺序各自四个非末语都回落，合看覆盖五语。部分大幅失分同时出现输出格式或跨度错误。以换序 UR 为例，槽位 F1 从 1.20% 学到 39.83%，再学 SW 后降到 3.13%。在两阶段共同有效的 1,000 道题上仍下降 12.06 个百分点。较强的 `1e−5` 顺序全参和 `5e−5` 的观察各自保留。
- **其他基准出现不同语言的遗忘和恢复。** MasakhaNER 的约鲁巴语原序终点回落 9.81 个百分点，豪萨语同期提高 8.66。TyDiQA 泰卢固语原序终点回落 8.91 个百分点，换序中阿语下一阶段回落 2.55 后部分恢复。MTOP 西语和印地语官测回落 10.78、4.20 个百分点，德语继续提高 15.89。两个 MasakhaNER 顺序合看回落覆盖两语，两个 TyDiQA 顺序合看覆盖三语。MTOP 泰语目前处于末阶段。
- **MASSIVE 回放略改善平均保留。** 原序两个种子的终点槽位 F1 分别提高 0.76、0.67 个百分点，旧语平均保留分别提高 0.94、1.07。SW 当期习得分别少 2.04、2.20 个百分点。SW 后续槽位保留差分别为 +0.38、+1.20，两区间均覆盖零。整句匹配保留差分别为 +1.78 和 +0.71，前者区间排除零。换序 UR 回放多保留 2.03 个百分点槽位 F1、2.86 个百分点整句匹配。
- **同源平行回放已完成初次受控比较。** 当前逐例监督下，与同标签异源回放相比，同源组的 SW 槽位 F1 差为 −0.30 个百分点，95% 区间 [−1.29, +0.69]。后续可固定全部训练例与批次，只切换显式一致性损失里的真实和打乱配对。
- **待补项。** MASSIVE `1e−5` 全参的同学习率 Single／Joint，MTOP 的 Joint 和换序。XNLI 无英语其余语言仍以开发结果为准。TyDiQA 上表是作者开发集。Belebele 等待独立双语验收。

## 图表与聚合结果

[全部图表索引](figures.md) · [逐阶段分数和配对区间（CSV）](stage-matrices.csv) · [逐语控制组（CSV）](baseline-scores.csv) · [单语预筛和正式分数](single-and-baselines.md) · [XNLI 历史矩阵](xnli-stage-results.md)

方法参考论文中的逐阶段矩阵与同预算对比。 [MASSIVE 全语言矩阵](https://aclanthology.org/2025.fieldmatters-1.2.pdf#page=11) · [MASSIVE 回放与学习率基线](https://aclanthology.org/2023.findings-acl.48.pdf#page=3) · [MTOP 逐语曲线](https://aclanthology.org/2023.acl-long.217.pdf#page=25) · [TRACE 逐轮表](https://arxiv.org/html/2310.06762#A4.SS2)
