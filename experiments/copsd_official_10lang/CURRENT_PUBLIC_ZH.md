# 十语言官方 COPSD：当前状态与结果

<!-- COPSD_LATEST_STATUS_BEGIN -->
> 2026-10-25 02:36：4B十语言独立20更新及13组开发配对完成；独立开发集已选SW、TE、ZH。三个匹配单语言参考与SW→TE→ZH、ZH→TE→SW两条顺序的训练均已完成验收。正式阶段评测正在四卡执行，完整遗忘矩阵尚未完成。
<!-- COPSD_LATEST_STATUS_END -->

## 当前4B研究：训练完成，正式评测进行中

更新：2026-10-25（Asia/Shanghai）。当前结果入口为[4B维护总表](FOUR_B_RESULTS_SCIENCE_20261024.PUBLIC.json)和[今日日报](../../docs/daily/OPD_公开科研摘要_2026.10.25.md)。1.7B结果与早期排查见文末历史链接，各自协议分别保留。

- 原2700题池的十语言独立20更新均完成，十语言开发配对及SW/TE/ZH额外seed43共13组已验收。开发题用于选语言，未用正式测试分数选择checkpoint或语言。
- 三个900题池匹配参考与四个后续顺序段共七个唯一训练段均完成；两条顺序各60更新、1920个不同英文原题。第一阶段直接复用相应匹配参考。训练成本约21.571 GPU小时，不含原2700池实验、排错、开发或正式评测。[训练验收与成本](SEQUENTIAL4B_ALL_TRAINING_ACCEPTED_SCIENCE_20261024.PUBLIC.json)
- Base七项正式输出已分别验收：三语言MGSM各250题、PolyMath各500题与英语GSM8K1319题。SW/TE/ZH的MGSM为13.6%/59.6%/96.0%，PolyMath为4.4%/14.4%/41.4%；英语94.54%。[主评测基线](SEQUENTIAL4B_BASE_MAIN4_SCIENCE_20261024.PUBLIC.json)、[PolyMath四档基线](SEQUENTIAL4B_BASE_POLYMATH3_SCIENCE_20261024.PUBLIC.json)

已完成的匹配单语言模型MGSM配对如下；每项250题、n1/8192/native16，同题同协议。三个独立匹配参考的九项主测试均已验收；它们不是顺序训练阶段矩阵，英语与PolyMath仍待补齐。

| 训练语言 | 评测语言 | Base→OPD正确数 | 差值（百分点） | 配对题目95%区间 |
|---|---|---:|---:|---:|
| SW | SW | 34→55 | +8.4 | [3.6,13.2] |
| SW | TE | 149→140 | −3.6 | [−9.6,2.4] |
| SW | ZH | 240→239 | −0.4 | [−2.8,2.0] |
| TE | SW | 34→55 | +8.4 | [4.0,12.8] |
| TE | TE | 149→152 | +1.2 | [−4.4,6.8] |
| TE | ZH | 240→241 | +0.4 | [−2.0,2.8] |
| ZH | SW | 34→46 | +4.8 | [0.0,9.6] |
| ZH | TE | 149→149 | 0.0 | [−5.6,6.0] |
| ZH | ZH | 240→241 | +0.4 | [−2.0,2.8] |

SW本语及TE训练后到SW的迁移观察到数学收益，其余已完成配对的区间均包含零；不据此宣称训练seed总体稳定性或普适收益。TE→SW的数学正确且思考区SW高置信29→45，但思考与最终区均SW高置信的正确数12→10，不能称全面语言收益。SW语言分析仍存在数学与最终回答语言保持的权衡，见[SW完整部分配对](SEQUENTIAL4B_SWMATCHED_MGSM_PARTIAL_SCIENCE_20261024.PUBLIC.json)。跨语言预训练能力保持不等于已经学完某语言后的遗忘。区间反映固定模型的题目重采样，不代表训练seed总体，也没有多重比较校正。

待完成：八个模型（Base、三个匹配参考、四个顺序后续阶段）的完整七项套件及语言分析，两条顺序的完整阶段矩阵、匹配预算比较、英语保持和成本汇总；原2700池十语言模型的完整正式报告亦未完成。已有后台调度继续执行，不能把训练结束当成这些交付已完成。

### 顺序阶段部分正式结果

正序SW→TE第二阶段的中文MGSM已验收：Base240/250、SW阶段后239/250、SW→TE后242/250（96.8%）。相对Base差+0.8个百分点，配对题目区间[-1.2,2.8]包含零；相对上一阶段获得5题、失去2题。中文尚未作为训练阶段，本项不是中文已学能力遗忘，也不能证明SW或TE保持。TE、英语和PolyMath等其余阶段测试仍需补齐；SW单项结果见下一段。参见[部分配对及阶段身份](SEQUENTIAL4B_FORWARD_S2_ZH_MGSM_PARTIAL_SCIENCE_20261025.PUBLIC.json)。

SW已学语言的第二阶段测试也已验收：SW阶段55/250（22.0%）→SW→TE后73/250（29.2%），获得32题、丢失14题，增益+7.2个百分点，题目配对区间[2.0,12.4]；旧语言遗忘值保留为**−7.2个百分点**，区间[−12.4,−2.0]。该项数学继续提升，同时思考区英语标签29→110/250、严格SW高置信199→111，数学正确且SW思考高置信44→40；数学正确且两区均SW高置信6→10。整体不能认定语言与数学同时保持，也不能推广为OPD防遗忘。实际父adapter衔接与逐题协议一致已验证，18条阶段TensorBoard事件保留了负遗忘值。[逐阶段SW数学与语言证据](SEQUENTIAL4B_FORWARD_S2_SW_FORGETTING_LANGUAGE_SCIENCE_20261025.PUBLIC.json)

逆序ZH→TE的已学中文结果为241→234/250（96.4%→93.6%），数学遗忘点估计**+2.8个百分点**，配对区间[0.0,6.0]仍包含零。两阶段均无截断或提取失败；英语思考top1均0，中文思考高置信249→248。11道新增失分均正常终止、提取答案且保持中文高置信思考。当前不能把该点估计称为稳定遗忘，也不能将它与不同基线的SW单项直接用于排序训练顺序。[中文逐阶段配对](SEQUENTIAL4B_REVERSE_S2_ZH_FORGETTING_SCIENCE_20261025.PUBLIC.json)

逆序ZH→TE在尚未训练的SW上为44/250（17.6%），Base34/250、前一ZH阶段46/250；相对两者的配对区间均包含零，不认定稳定迁移或下降，也不计算已学SW遗忘值。[未学SW部分配对](SEQUENTIAL4B_REVERSE_S2_SW_MGSM_PARTIAL_SCIENCE_20261025.PUBLIC.json)

逆序当前语言的新结果：ZH→TE的TE为130/250，低于Base及前一ZH阶段各149/250，也低于匹配TE单语言152/250；250条思考仍全部为泰卢固语高置信标签。相对前一阶段新增45道失分中23道截断（22道同时未提取答案），另22道正常结束但提取答案判错；不能将退步全归因于语言切换或长度。ZH→TE→SW的SW为66/250，高于Base34和前一阶段44；相对匹配SW单语言55的区间包含零。SW高置信思考183→168、英语思考13→68，数学正确且SW思考37→48、两区均SW且数学正确17→11，不能仅凭数学提升宣称语言保持改善。这两项是当前语言获得，不是旧语言遗忘。[完整配对与限制](SEQUENTIAL4B_REVERSE_SW_TE_ACQUISITION_SCIENCE_20261025.PUBLIC.json)

正序SW→TE的当前TE结果出现严重未完成问题：Base149/250、前一SW阶段140/250、匹配TE单语言152/250，当前仅14/250（5.6%）。234条触及8192上限，234条没有提取答案；相对前一阶段新增126道失分全部同时截断且无答案。250条思考仍为TE高置信，百分比补充分也仍14，不能把这一退步归因于英语切换或百分号评分。该阶段三语MGSM完整行为SW73、TE14、ZH242，不能只报告SW收益；英语、PolyMath和后续阶段仍待完成。[负结果与逐题分解](SEQUENTIAL4B_FORWARD_S2_TE_COLLAPSE_SCIENCE_20261025.PUBLIC.json)

### 评分与错误诊断

逆序中文的11道新增失分已审查：10道最终解释与基准预期存在数学或语义偏差，1道涉及百分比单位表达。原官方符号函数会把25%与纯数字金答案25判为不同；这是本项目MGSM数值标签的单位接口问题。已基于全部250道英文题固定4道百分比答案标签，并对17个已验收4B条件统一作CPU补充诊断，原分与输出保留。七个SW条件的逐题标签均不变；中文小幅差异对格式约定敏感。补充分数单列，不替代官方原分，现有语言联合计数仍按原评分定义。[诊断与规则](MGSM_PERCENTAGE_AND_REASONING_DIAGNOSTIC_20261025.PUBLIC.json)、[CPU源码](mgsm_percentage_diagnostic.py)

### 官方来源、环境与适配

- 固定源码：[cisnlp/COPSD@4f4410730e9b6775a30d20f4b6e141cba2e87590](https://github.com/cisnlp/COPSD/tree/4f4410730e9b6775a30d20f4b6e141cba2e87590)。4B模型revision为`1cfa9a7208912126459214e8b04321603b3df60c`；与历史1.7B分开。
- 当前环境实际包版本：Torch2.8.0、Transformers4.57.1、TRL0.26.0、PEFT0.17.1、vLLM0.11.0、math-verify0.8.0。源码、模型、输入和具体配置身份由每次冻结文件及验收记录绑定。
- 官方核心保留LoRA64/128、原始Base禁用adapter教师、在线学生rollout、英文题与参考的教师特权上下文、AdamW5e-6、有效batch32、完整2048训练预算、beta0和词表分量clip0.05。当前每段20更新沿用100步调度，是本项目预算适配，不能称严格官方100步复现。
- 工程适配包括两GPU分别承担训练/教师与在线vLLM学生服务、批量rollout、非破坏性合并学生权重传输、回答位置logits及mask归约内存优化、依赖/UUID兼容与恢复。教师仍为原始冻结Base；两GPU角色分工不等于数据并行。实际启用以冻结源码哈希及[阶段入口](sequential4b_stage_entry.py)为准，[历史补丁目录](patches/)不表示每个补丁在当前运行同时启用。
- 当前正式评测是本项目`project_formal4b_n1_8192_native16_v1`：n1、8192生成预算、native LoRA、并发16、temperature1/top_p0.95/top_k−1、逐题seed42与官方答案解析。它与官方pass@12、旧2048预算及未通过一致性门槛的合并并发32实验分别记录。

### 复现入口与产物

| 阶段 | 入口与证据 |
|---|---|
| 原2700池数据／单语言 | [数据准备](prepare_remaining_4b_pool.py)、[TE模板与本地绑定说明](reproduction_templates/te4b/README_ZH.md)，其他语言使用各自已发布入口 |
| 三语言选择／900池 | [选择协议](SEQUENTIAL4B_SELECTION_SCIENCE_20261024.PUBLIC.json)、[数据接口](sequential4b_data.py)；实际ID清单和冻结记录本地保留 |
| 匹配参考／顺序阶段 | [生成阶段配置](prepare_sequential4b_stage.py)、[运行](run_sequential4b_stage.sh)、[验收](collect_sequential4b_stage.py) |
| 七项正式评测 | [生成评测配置](prepare_sequential4b_evaluation.py)、[单项运行](run_formal4b_case.sh)、[完整套件验收](collect_sequential4b_evaluation.py) |
| 矩阵／匹配对照 | [遗忘矩阵](build_sequential4b_matrix.py)、[匹配预算比较](compare_sequential4b_matched_single.py) |
| 语言／日志／成本 | [语言分析](analyze_sequential4b_languages.py)、[TensorBoard](record_sequential4b_evaluation_tensorboard.py)、[运行成本去重](inventory_4b_queue_costs.py) |

以下命令针对已经按上述流程在本地生成的新冻结文件；先使用对应项目环境，并将变量绑定为自己的文件。准备模式只做CPU校验，不启动GPU任务。公开模板不包含本机完整绑定，不能直接冒充原运行身份。

```bash
COPSD_SEQUENTIAL4B_STAGE_PREPARE_ONLY=1 bash experiments/copsd_official_10lang/run_sequential4b_stage.sh "$STAGE_FREEZE"
COPSD_FORMAL4B_PREPARE_ONLY=1 bash experiments/copsd_official_10lang/run_formal4b_case.sh "$EVAL_FREEZE" mgsm_rev2_sw
```

实际运行由已检查资源归属和存储的队列提供GPU UUID及角色数，入口仍为相同脚本；不在这里自动启动或重复生成当前已存在结果。新机器须重建官方数据与模型、本地冻结文件和资源锁，具体要求见[源码配置索引](REPRODUCTION_FILES_20261023_ZH.md)。checkpoint保留adapter、optimizer、scheduler、局部global_step和RNG；逐题输出、训练生成与同步回执、配置、验收文件和TensorBoard保留在各run目录。仅观察到进程alive或文件存在不能替代结果验收。

---

## 历史记录

[历史公开状态与完整1.7B记录](HISTORICAL_PUBLIC_ZH_20261025.md)为归档前公开文件的逐字快照，原内容与文件身份保留。各阶段原始JSON、日报和Git历史继续可追溯。
