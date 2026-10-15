# Qwen3 单语言 OPD 与教师轨迹 SFT 对照执行方案

更新日期：2026 年 10 月 15 日。目标仓库：lxy308297917-cloud/Multilingual-CL。

## 1 当前目标与任务变更

本文件是当前任务依据。文件路径保留原名，便于现有 Codex 链接继续使用，但实验优先级已改为：**先做 SW 单语言 OPD，检查有效性，再补齐同一特权教师的 SFT 与固定前缀蒸馏对照；十语言筛选和三语言顺序训练暂缓。**

当前没有在本方案中声明任何服务器训练、数据验收或评测已经完成。执行 Codex 必须核对服务器实际状态、脚本和产物，以实际报告为准。

本次变化：
- 暂缓 Qwen3-14B 批量生成短英文解答、再翻译十语言的旧标签路线。
- 主 SFT baseline 改为：冻结的特权教师直接生成目标语言轨迹，审核后做普通交叉熵 SFT。
- OPD 与该 SFT baseline 使用同一教师快照、同一英文特权信息和可对齐的问题池。
- OPD 小规模验证不等待全部 SFT 标签完成；正式 A/B/C 对照须另在共同问题子集上训练。
- 原有产物、日志和实验分支保留，不删除、不覆盖为新协议结果；旧 14B 数据可作辅助对照，不混入新的主标签集。

## 2 给服务器 Codex 的任务指令

先读取本文件，再检查实际运行任务、仓库修改、已有脚本和数据版本。先汇报哪些任务已经启动、哪些可以安全暂停。只暂停可明确归属于旧十语言合成计划的队列任务，不终止其他项目进程，不对正在写 checkpoint 的进程粗暴中断。

当前执行顺序为：
1. 审计 SW 静态问题和英文参考信息，准备冻结的 Qwen3-4B 特权教师。
2. 核查教师生成与在学生前缀上的监督质量。
3. 实现/复用 COPSD 风格 OPD，完成 smoke，并重新从 Base 开始跑 SW 短程 OPD。
4. 用同一教师直接生成 SW 轨迹，形成可验收的 SFT 标签。
5. 在共同问题池完成 A 教师轨迹 SFT、B 教师轨迹分布蒸馏、C 学生轨迹 OPD。
6. 有实际收益后补种子和其他语言；暂不启动十语言批量合成、十语言训练或顺序 CL。

无需等待十语言标签齐全。缺少合格教师、长度配置或必要资源时，完成不依赖这些条件的代码与审计并报告具体阻塞，不能用未审核标签启动正式 SFT，也不能伪造验收或成绩。训练必须开启 eval 和 TensorBoard。

## 3 模型 代码与环境

| 项目 | 当前设置 |
|---|---|
| 学生初始模型 | Qwen/Qwen3-4B，支持 thinking/non-thinking 的原始后训练快照 |
| 特权教师 | 相同快照、相同 tokenizer 的独立冻结 Qwen3-4B 副本 |
| 当前语言 | SW 斯瓦希里语，项目代码 sw，COPSD split 为 swa |
| 主模式 | thinking=True，思考与最终解释都要求 SW |
| 更新范围 | FFT；不暗中切换 LoRA |
| 项目主仓库 | https://github.com/lxy308297917-cloud/Multilingual-CL |
| OPD 参考实现 | https://github.com/cisnlp/COPSD |
| SFT 实现 | 复用现有合格入口，或 Transformers Trainer + Accelerate |
| 数学评分 | 固定版本 Math-Verify 与必要的精确数值规范化 |
| 评测生成 | 现有可靠 Transformers/vLLM 后端，训练与推理模板需一致 |

这里 Base 指原始 Qwen3-4B 后训练模型，不是 Qwen3-4B-Base，也不混用 2507 Instruct/Thinking 快照。十语言范围以后仍可保留 SW/TE/TH/BN/ZH/JA/RU/DE/FR/ES，但当前只实施 SW。

首先读实际 checkout 的 AGENTS.md（若存在）、现有 handoff 和训练入口；记录 branch、commit、未提交修改，保留用户改动。模型、tokenizer、chat template、环境和数据固定 revision。已有 NPU/CUDA 环境优先复用，不盲目升级。

COPSD 官方主要使用 LoRA；其移除 adapter 获取固定教师的实现不能直接照搬到 FFT。必须验证教师参数不进入 optimizer、无梯度、前后权重不变化；共享学生权重的 stop-gradient 前向仍然不是固定教师。

FFT + 独立教师 + 全词表 KL 显存开销较大。优先 micro batch=1、梯度累积、gradient checkpointing、分设备部署教师和按回答位置分块计算 KL，避免缓存全序列学生/教师 FP32 大词表 logits。保持数学目标不变，必要时重计算并记录成本。资源不够应报告实际瓶颈，不擅自将 top-k 近似称作全词表蒸馏。

## 4 静态问题池的构建

### 4.1 来源与字段

优先使用 https://huggingface.co/datasets/yihongLiu/COPSD-PolyMath-TrainDataset 的 swa split。数据卡列出 problem、solution、Answer、problem_translated、source 等字段；必须实际核验 schema 和内容，不能按字段名猜测。

规范记录为：

`problem_id, source, question_sw, question_en, reference_en_original, gold_answer, split, source_revision, content_hash, source_quality_status`

每题的语义对象为：
`d_i = (q_sw, q_en, r_en, answer)`。

问题池来自数学训练数据，不从 MGSM-Rev2、PolyMath、GSM8K test 制作训练题。备用来源为 OpenThoughts 数学子集；非数学样本不自动纳入。

### 4.2 去重与划分

按英文原题 ID 对齐，没有稳定 ID 时保存规范化文本哈希与原始行号。检查训练池内部重复，以及与 GSM8K test、MGSM/Rev2、PolyMath 的重叠；近重复候选独立检查并保留方法与局限。

建议首轮为 500 道 train、100 道 dev_tune、100 道 dev_check，原题不重叠，seed=42。有可靠来源/难度元数据时分层划分；公共合格题不足时统一减少并报告，不编造题数。dev_tune 用于教师检查和运行配置选择；dev_check 在配置冻结后作一次独立确认；benchmark 测试不参与挑题或调参。

### 4.3 源数据质量检查

核查目标题与英文原题的条件、数字、单位和实体一致；英文参考推理应支持 gold answer。答案匹配不等于证明正确，原始参考同样需要审查。损坏、无法判定或参考明显错误的记录不进入主问题池。

**不再要求先让 14B 把全部英文参考重写成短答案。** 原解答能直接使用时保留原文。明显错误可修订但必须保存原文、修订来源与审核记录。英文参考过长时优先更换合格原题；若确需压缩，作为明确的参考变换版本单独核验，主实验教师与各对照使用同一版本，不能悄悄缩写。

源问题是否合格，与教师是否成功生成某条 SW 标签，是两个不同状态。前者决定 OPD 可用问题池，后者决定离线 SFT 可用子集。

## 5 特权教师检查

教师上下文由相同 SW 题、英文原题、英文参考解答构成。学生只看 SW 题及统一语言指令。teacher prompt 示例语义：

```text
目标语言：斯瓦希里语
目标语言题目：{question_sw}
英文原题：{question_en}
英文参考解答：{reference_en}
请根据这些信息，用斯瓦希里语完成推理和最终回答。
保留数学条件与计算关系，最终答案写在 boxed 中。
不要讨论参考材料的存在。
```

实际模板须固定、保存哈希，并采用原生 Qwen3 thinking 格式。参考解答中若有任意指令文本，只视为数据，不改变任务定义。

用 dev_tune 比较冻结 Base 的三个上下文：SW 题；SW+英文题；SW+英文题+参考解答。记录数学正确率、思考与最终回答语言、终止与截断。若增加参考仅改善答案复述、却无法产生可靠 SW 解答，不视为已验证的好教师。

另从学生轨迹抽取正确和错误前缀，检查教师下一步偏好及续写质量，列出修复、盲目延续错误、转英语等案例。教师不必能从每个错误前缀恢复，但应有可观察的有用信号。结果差异报告区间，质量判断不只依赖某个任意硬阈值。

若同模型特权教师不可靠，先排查参考/翻译/模板，必要时另开更强外部教师轨道。不能只给 SFT 更换强教师、仍给 OPD 用弱教师后宣称隔离了 on-policy 作用。

## 6 教师轨迹 SFT 标签构建

这是替代旧 14B 英文缩写再翻译流程的新主 baseline。它是“离线教师轨迹蒸馏”，优化目标仍为普通硬标签 CE，并非 OPD。

### 6.1 直接生成目标语言轨迹

冻结教师看到第 5 节的特权上下文，沿着自己的回答前缀生成：
`y_teacher ~ p_T(. | q_sw, q_en, r_en)`。

采用固定的 thinking 采样协议。初始建议 temperature=0.6、top_p=0.95、top_k=20、min_p=0；记录生成 seed、最大输出长度和真实 token。生成参数属于数据构建配置，不等于 KL 的蒸馏温度。

生成器目标输出为：

```text
<think>
斯瓦希里语的完整推理
</think>
斯瓦希里语最终结论与唯一最终 boxed 答案
```

不要求逐字翻译原英文解答；要求数学上正确并用目标语言表达。提示可以要求简洁完整，但不能为挤进长度预算截断推理或伪造简短正确证明。

### 6.2 有限候选与成本

第一轮每题先生成 1 条候选。若需补采样，先在 dev_tune 上冻结最多候选数（建议不超过 3）和选择规则，然后应用到 train，保留全部候选。不可无限重试到正确。

记录第一次成功率、最多 K 次后成功率、每种失败原因和所有生成成本。教师轨迹筛选利用正确性与语言信息，其成本和选择偏差必须报告，不能只报告最终成功标签。

### 6.3 逐条硬检查

以下检查失败的候选不能直接用作正确 SFT 标签：
- 缺少正常思考结束、没有 final 区、格式损坏或生成被截断；
- final 区答案无法解析或与 gold 不符；
- 已发现实质计算/推理错误、遗漏必要条件或翻译改变题意；
- 思考/解释主要不是 SW，或语言无法判定却被标成合格；
- 完整学生训练序列超过固定 2048。

对不确定数学或语言问题标 pending_review；明确错误标 rejected；修订需保留版本。格式、答案和长度检查是可自动化的必要条件，不等价于全部推理和语言正确。

### 6.4 独立审查与批次验收

每批保留原始候选、自动检查结果、审查者/工具、审核意见与采用理由。每语至少分层检查 30 条 train、20 条 dev，覆盖随机样本、长推理、难题和疑似英语夹杂；发现系统性错误应扩大审查并修复生成流程。

旧方案的“明显错误超过 10% 暂停”只是返工触发线，**错误率不超过 10% 不代表其他样本自动全部 accepted**。同一模型自我评价不等于独立审查；Codex 数学预核不等于人工 SW 语言验收。没有可靠母语审查时明确自动/回译检查的范围和残余局限，不能虚构人工审核。

Codex 需要在数据审计报告中写清 acceptance_policy：哪些检查自动、哪些抽查、待审如何处理、由谁作最终放行、批次何时可用于正式训练。accepted 可以表示在已声明协议下被接纳，不能宣称所有条目均已逐条人工验证。若关键语言质量始终无法建立可信检查，应报告阻塞而非永远只生成 pending。

正式 SFT Trainer 只读取冻结 accepted manifest。smoke 使用待审候选时只能验证接口，明确标记并重新从 Base 启动正式实验，不将 smoke 权重或成绩用于结论。

### 6.5 最终训练对与字段

SFT 样本为 `(q_sw, y_teacher)`，学生输入没有英文题、参考或 gold。仅对 assistant 回答与真实结束符计算 CE，system/user/padding labels=-100。

保存字段至少为：
`problem_id, question_sw, question_en, reference_en_original, reference_en_used, gold_answer, teacher_model_revision, teacher_prompt_hash, generation_seed, candidate_index, reasoning_sw, final_sw, raw_output, assistant_completion, final_answer_match, language_status, termination_status, student_full_tokens, review_status, review_method, review_notes, selected_candidate, split`。

模型输出和审核证据冻结后可复用，不随学生 checkpoint 重新生成标签。

## 7 SFT 与 OPD 长度协议

保持用户的 SFT 设置：完整序列固定 pad 到 2048，不使用动态 block size、不跨题 packing。包括模板、题目、思考、最终回答和结束符。教师生成长度上限不能误认为学生完整序列长度；用真实学生 tokenizer 核验。

原生 Qwen3 模板可能处理 assistant 思考内容，必须检查实际 token 序列和 loss mask，确保训练思考未被模板移除、没有重复 think 开头、真实结束符没有因 pad 同 ID 被屏蔽。

OPD 长度独立定义：
- 初始 student completion 上限 2048，参考 COPSD 的 rollout 预算；
- teacher 上下文必须容纳 SW 题、英文题、完整参考和学生回答前缀；
- 按真实模型窗口与后端 max_model_len 检查总长，记录 prompt/answer 各自预算；
- 不能把教师上下文强行截成 SFT 的 2048，不能静默丢弃英文参考或学生 prefix。

SFT 长度筛选会偏向短轨迹/短题，报告筛选前后题源与难度。若保留率太低，应先报告设计冲突并提出可审查的调整；不能偷偷放宽为 4096，也不能把长轨迹切成未说明的新样本。当前 OPD smoke/先导运行可以继续使用源质量合格问题池，不依赖 SFT 长度筛选完成。

## 8 OPD 在线训练数据与目标

每批取静态问题 `d_i`，学生只看 SW 输入，在线生成 `y_student`。冻结教师收到特权上下文并在同一学生前缀上计算下一 token 分布。两者回答 token 完全对齐，但 prompt 长度通常不同，应检查各自 offset 和 causal shift。

定义：
`p_t = p_student(. | q_sw, y_student_<t)`
`q_t = p_teacher(. | q_sw, q_en, r_en, y_student_<t)`

第一版全词表 reverse KL：
`L_OPD = sum_t mask_t * KL(p_t || q_t) / sum_t mask_t`。

教师输出 detach；rollout 采样过程不反传；学生在固定采样轨迹上重新前向计算分布并更新。包含思考与 final 的有效生成 token，排除输入与 padding。loss 下降本身不证明任务能力提升。

轨迹来自当前学生，错误轨迹也可提供监督；不默认只保留答案正确、SW 合格或长度短的学生轨迹。若加筛选或跳过异常样本，应单独定义变体并记录数量。EOS 终止与长度截断要区分；截断序列可对有效 token 计算 KL，但记录其缺少完整最终回答的风险，不伪造 EOS。

每个新 rollout batch 同步当前学生权重；如为工程吞吐采用少量轨迹复用，记录 policy version、复用次数与陈旧度，不声称严格逐更新 on-policy。教师始终同一冻结 Base，不随学生同步。

无需预先用 14B 制作 SW 标准答案才能运行这种 OPD。但静态英文参考的正确性与目标语言问题质量仍要审计。

## 9 SW 先导运行配置与检查

以下是项目起始值，不是已调优最优值，也不是论文 FFT 复现：

| 参数 | 默认 |
|---|---|
| optimizer | 标准 AdamW，记录实现与状态精度 |
| learning rate | 5e-6 |
| global batch | 32 个问题，micro batch 与梯度累积按硬件调整 |
| 训练更新 | 正式先导 100 个 optimizer step，按 20/50/100 检查 |
| weight decay | 0.01 |
| betas / eps | (0.9,0.999) / 1e-8 |
| scheduler / warmup | cosine / 5% |
| grad clip | 0.1 |
| precision | BF16，数值敏感 KL/log-softmax 稳定计算 |
| rollout / question | 1 |
| rollout sampling | thinking=True，temperature=1.1，top_p=0.95，top_k=20 |
| KL 温度 | 1.0，与 rollout temperature 分开 |
| rollout 上限 | 2048 completion token |
| 初始训练 seed | 42，有收益后补 43、44 |
| logging / eval | 每 5 step 日志，0/20/50/100 step 开发集生成评测 |
| 保存 | 与评测点一致，保存模型、优化器、sampler/RNG 和配置 |
| 初始化 | 每个正式独立 run 都重新从原始 Base 开始 |

rollout 温度 1.1 参考 COPSD/RP-OPSD 的训练配置；正式评测使用第 11 节较低温度，二者不可混淆。生成器标签采样温度也有独立记录。固定 500 题下 100×32 更新会重复曝光问题，报告实际次数与末批，不能描述成只训练一轮。

先用少量训练题完成 5—20 step smoke，检查 loss/梯度/教师冻结、mask、权重同步和 checkpoint 恢复。smoke 通过后新建正式 run 从 Base 开始，不接续 smoke 模型。若数值或生成异常，先诊断而非立即扩规模。

不沿用旧 15000 步、15.36M 输入预算或十语 SFT 两轮配置。若需学习率校准，在独立开发集上给予方法相同搜索预算并明确哪些是共享设置、哪些是方法专用，不能把固定单一学习率的失败等同于方法无效。

## 10 同教师 A B C 正式机制对照

| 组别 | 回答前缀来源 | 学习信号 |
|---|---|---|
| A 教师轨迹 SFT | 第 6 节 accepted 教师轨迹 | 硬标签 CE |
| B 教师轨迹分布蒸馏 | 与 A 完全相同的轨迹 | 与 C 相同的特权教师全词表 reverse KL |
| C 学生轨迹 OPD | 当前学生在线生成 | 同一特权教师全词表 reverse KL |

A/B/C 学生从相同 Base 开始。B 教师在其固定目标轨迹前缀上评分，C 在学生前缀上评分。A/B 的英文信息仅供教师生成/评分，不能放入学生输入。

A→B 比较轨迹相同条件下硬标签与分布监督；B→C 主要检查轨迹来源。A→C 是离线轨迹蒸馏与 on-policy 蒸馏的整体比较，不能只凭这一对归因于前缀。

问题池控制必须明确：
- OPD 先导 C_full 可以使用所有源质量合格 train 题，先回答是否有学习信号。
- 主 A/B/C 使用同一 accepted 教师标签对应的题目子集 Q_common；C_common 必须在该子集上另从 Base 训练。
- 不能直接拿 C_full 对比筛选后的 A/B，宣称隔离了 on-policy。报告 Q_common 占原问题池比例和筛选偏差。
- C 不因学生本次答错而删题；教师离线标签筛选与在线学生轨迹筛选不是一回事。

机制主对照匹配问题清单、shuffle/曝光次数、更新范围和尽可能一致的优化器更新安排。参考第 9 节作为共同起点，SFT 不被预设为该学习率已最优；必要时用相同开发搜索预算补强 baseline。

分别统计离线教师生成/筛选 token、SFT 监督 token、B/C 教师评分 token、学生 rollout token、有效 loss token、耗时与峰值显存。等问题曝光不等于等 token/算力。补充成本曲线或固定有效预算对照，不承诺所有预算同时一致。

若 A 获得更多候选选择或修订，计入制作成本。可以另加原 14B 短解答 SFT 作数据来源消融，但不能替代同教师主对照。

## 11 评测与成功判断

### 11.1 生成与指标

数学主评测固定 Qwen3 thinking：temperature=0.6、top_p=0.95、top_k=20、min_p=0、一次回答、max_new_tokens=4096，默认无额外 repetition/presence penalty。固定 generation seed=17；冻结后用 18/19 检查采样稳定性，报告平均单次正确率，不取三次任意成功冒充单次准确率。

训练 2048 completion、SFT 2048 total、评测 4096 new tokens 是三个预算，分别记录。按开发集预先进行必要长度敏感性检查；不能按测试成绩给不同模型选有利上限。

- dev_tune：教师检查、监控曲线、checkpoint 选择；检查 0/20/50/100。
- dev_check：配置和 checkpoint 冻结后的独立确认。
- MGSM-Rev2 SW：基础数学测试，不能用于筛训练题或选 best。
- PolyMath SW：完整四档扩展；官方数据/指标按固定 revision 的实际文件核验，困难题接近零也保留。
- 英语 GSM8K：与目标语言相同数学生成协议，检查能力变化。
- 英语 IFEval：单独冻结 non-thinking 任务协议，Base 与所有方法一致。

数学评分只从 final 区提取最终答案，不从未完成 think 中搜索正确中间数。预测提取失败计错，全题固定分母；gold 解析异常需修复或明确报告，不静默删题。校验本地数字、分数、负数、百分数、单位、多 boxed 等。

分别报告思考区与 final 区的 SW 合格率、英语切换、unknown、正常终止、截断、输出 token。去除公式/数字再做语言识别；无自然语言的结果不自动算语言合格。

### 11.2 判断标准

先回答 C_full 相对 Base 是否有实际值得继续的收益，再判断 C_common 相对 A/B 是否有优势。这是两个不同问题。

报告题目配对正确率变化与 bootstrap 区间、完整逐题输出、训练/生成 seed 和成本。首次单 seed 改善只称先导信号，后续补至少三训练 seed；bootstrap 不替代训练复跑。

loss 降低、输出更短、final 更常闭合可能解释分数变化，需与数学和语言指标一起分析。OPD 无收益时先排查教师信号、mask/offset、权重同步、学习率和长度限制，再决定是否停止。不得把“C 必须胜出”作为验收目标。

## 12 后续语言筛选与顺序计划

只有单语言数据、教师和实现链路明确后才扩大。候选范围仍是 SW/TE/TH/BN/ZH/JA/RU/DE/FR/ES，可以按成本先补两语，不强制立即十语全量。

筛选时同时看 Base、single 后绝对水平、提升、输出语言与教师可靠性；独立开发集选语言，正式 benchmark 不参与语言搜索。筛选方法是 SFT 还是 OPD 应明确：SFT 高提升不保证 OPD 高提升，不能混称同一结果。

选三语 L1/L2/L3 后，先开展 L1→L2→L3 与 L3→L2→L1：
- 每条顺序、每方法从原始 Base 开始，阶段接续学生权重，不拼接 single 模型。
- 每阶段重置 optimizer/scheduler 的默认约定对所有方法一致。
- 各阶段训练原题 ID 不重叠，同阶段方法用相同问题池；数据不足先扩充或统一减小，不能复用小池假称原题不重叠。
- B/C 第一版始终用同一冻结 Base 特权教师，学生轨迹随阶段更新。
- 每阶段评全部三语和英语，报告完整矩阵、获得、旧语言下降、最终平均、语言保持和成本。
- SFT/稀疏等同类比较固定每阶段有效监督 token；跨 SFT/OPD 另匹配问题曝光并报告计算/长度差异。
- 旧任务历史最佳分减当前分的遗忘可为负，保留正迁移，不隐藏原始矩阵。

on-policy 不自动保护旧语言。基线遗忘明确后再单独加入 replay、旧语言/英语保持约束或稀疏更新，每次一项，保留未加模块对照。replay 数据来自训练池，不取测试题；区分回放旧问题重新 rollout 与回放固定答案。预算与额外教师成本单独报告。

## 13 实施交付与验收

建议在 experiments/math_single_10lang/ 或新的清晰子目录组织，避免覆盖旧结果。组件名称由现有代码结构决定，至少交付：
- SW 静态问题审计、去重与 split manifest；
- frozen teacher 与学生 prompt、revision、长度配置；
- teacher trajectory 生成器、原始候选及 acceptance_policy；
- OPD trainer/launch 配置、冻结和回答位置对齐检查；
- SFT/B/C 配置与 Q_full/Q_common 清单；
- dev/test 生成、评分、语言与成本汇总；
- 中文进度报告，列出已运行/待运行/阻塞与真实路径。

每个 run 保存模型/数据/prompt 哈希、环境、命令、train/eval 日志、TensorBoard、optimizer/sampler/RNG 状态。恢复同 run 须检查版本一致；新方法不能意外 resume 另一个方法权重。

必要验证：学生提示无答案泄漏；教师冻结；teacher/student offset 与 causal shift 正确；loss mask 正确；数学评分和分母可靠；rollout 同步当前学生；checkpoint 恢复可复现合理进度。少量必要回归用例即可，不用实现镜像测试堆数量。

阶段验收以可审计产物为准，不以“跑出正结果”为准。候选生成数量、accepted 数量、完成训练数量和有效评测数量分开报告。任何待审内容的 smoke 不能冒充正式实验。

## 14 参考论文与实现阅读任务

| 参考 | 需复用或核查的内容 |
|---|---|
| Liu 等 2026 COPSD | 目标题+英文题/参考解答；学生轨迹上固定教师全词表 reverse KL；thinking 与语言前缀；LoRA 与本项目 FFT 的差异 |
| Wang 等 2026 RP-OPSD | 同一学生轨迹上有/无英文参考推理的教师视图；token gate 与 reference anchor；当前先不实现 gate |
| Heo 等 2026 多语言 OPD2 | 问题字段即可用于外部强教师 OPD；EN/KO/JA、两种模式；teacher 与其对应 base 的差分不能随意替换 |

COPSD/RP-OPSD 的静态数据不要求目标语言金标准推理标签。Heo 的外部教师路线仅保留问题字段，是另一种设定；不能因此删掉本项目 COPSD 特权教师依赖的英文参考。

本次同教师轨迹 SFT 是本项目为加强对照提出的设计，不声称完整照搬三篇论文的 baseline。不要把使用 Qwen3 等同于已复现其数值。

官方入口：
- COPSD：https://github.com/cisnlp/COPSD
- COPSD 问题池：https://huggingface.co/datasets/yihongLiu/COPSD-PolyMath-TrainDataset
- OpenThoughts：https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
- OPD2：https://github.com/naver-ai/opd2
- MGSM-Rev2：https://github.com/google-research-datasets/MGSM-Rev2
- PolyMath：https://github.com/QwenLM/PolyMath
- PolyMath 数据：https://huggingface.co/datasets/Qwen/PolyMath
- Math-Verify：https://github.com/huggingface/Math-Verify
- Qwen3 模型卡：https://huggingface.co/Qwen/Qwen3-4B

论文优先读取用户提供的原文；服务器无本地副本时从官方仓库定位，不猜本地文件路径。所有执行依据固定版本。

## 15 第一份进度报告应回答的问题

1. 旧十语言生成任务哪些仍在队列或运行，如何保留并调整？
2. SW 源质量合格问题多少，train/dev 有无污染，英文参考可直接用多少？
3. 特权教师与学生具体快照/模板是什么，是否有可靠教师信号？
4. OPD smoke 是否完成，教师冻结、回答对齐和权重同步如何验证？
5. C_full 正式先导是否开始，评测点与成本如何？
6. 同教师 SW 标签候选/accepted 各多少，验收由什么规则与证据完成？
7. Q_common 是否足够，A/B/C 何时能开展，哪些仍是阻塞？

优先推进已经授权且依赖满足的工作。不要只交计划，不要重复已核验的产物，也不要在未满足条件时同时启动全部后续训练。
