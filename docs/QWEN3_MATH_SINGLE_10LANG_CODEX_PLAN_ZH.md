# Qwen3 十语言数学 Single 微调与语言筛选执行方案

日期：2026 年 10 月 15 日。执行对象：服务器上的 Codex。当前目标是完成十语言独立数学 SFT，找出适合后续三语言顺序学习与 on-policy 蒸馏的候选语言。本文中的训练规模、超参数和筛选门槛是本项目的建议默认值，不是三篇参考论文的原始配置。

## 1 当前实验目标与执行范围

回答三个问题：同一小模型在不同语言上能否通过数学 SFT 获得能力；哪些语言在统一配方下提升稳定且保持目标语言输出；这些语言是否适合进一步检查教师监督和顺序遗忘。

本轮实施数据审计、参考解答准备、评测适配、十语言 single FFT、结果汇总。完整后续目标为教师检查、一语言 A/B/C、三语言顺序比较和必要的保持机制；详见第 10—11 节。每次训练独立从同一初始模型开始，不接续其他语言的 checkpoint。暂不启动十语言顺序训练，也不把稀疏更新、英语保护、OPD 同时混进第一轮。

这里的 Base 指未经本轮微调的 Qwen/Qwen3-4B 后训练模型，不指 Qwen3-4B-Base。single 的结果仅用于选语言和检查可学习性，不能证明 on-policy 有效或持续学习遗忘得到改善。

## 2 语言范围

十种语言全部保留，不能提前根据 benchmark 测试分数删语言。

| 项目代码 | 语言 | COPSD 发布数据 split | 本轮角色 |
|---|---|---|---|
| sw | 斯瓦希里语 | swa | 独立 SFT 候选 |
| te | 泰卢固语 | tel | 独立 SFT 候选 |
| th | 泰语 | tha | 独立 SFT 候选 |
| bn | 孟加拉语 | ben | 独立 SFT 候选 |
| zh | 中文 | zho | 独立 SFT 候选 |
| ja | 日语 | jpn | 独立 SFT 候选 |
| ru | 俄语 | rus | 独立 SFT 候选 |
| de | 德语 | deu | 独立 SFT 候选 |
| fr | 法语 | fra | 独立 SFT 候选 |
| es | 西班牙语 | spa | 独立 SFT 候选 |
| en | 英语 | 不包含在上述十个 split 中 | 原题对照和保持评测，不算目标语言阶段 |

MGSM-Rev2 与 PolyMath 均覆盖上述十种目标语言。代码必须维护显式映射，不要把 ISO 二字母代码、三字母 split 名和旧项目中的 FineWeb2 语言配置混用。

## 3 推荐代码库与实现方式

| 用途 | 代码库或工具 | 实施要求 |
|---|---|---|
| 项目主仓库 | https://github.com/lxy308297917-cloud/Multilingual-CL | 首先读服务器 checkout 的 AGENTS.md、README、handoff、现有 SFT 与评测入口；记录当前 branch、commit 和工作区修改。优先扩展已有数据、日志、评测接口 |
| SFT 训练 | Hugging Face Transformers Trainer，配合 PyTorch 和 Accelerate | 本轮使用标准完整参数 SFT；若现有入口支持 assistant-only loss 和 2048 长度则复用，否则新建小而明确的入口 |
| 后续自蒸馏参考 | https://github.com/cisnlp/COPSD | 读 trainer、collator、language_config、训练脚本；用于理解特权教师上下文和 rollout，不将其 OPSD trainer 直接当作普通 SFT trainer |
| 基础数学评测 | https://github.com/google-research-datasets/MGSM-Rev2 | 实际读取固定版本的 TSV 和替换题目记录，核验答案、数量和语言 |
| 扩展数学评测 | https://github.com/QwenLM/PolyMath 与 https://huggingface.co/datasets/Qwen/PolyMath | 使用官方题目、难度划分和评测代码；将生成后端与评分后端分开 |
| 数学答案解析 | https://github.com/huggingface/Math-Verify | 固定版本；基础数值题可配合 Decimal/Fraction 精确规范化，不只用浮点近似或字符串相等 |
| 模型生成 | Transformers；CUDA 上可用 vLLM 加速 | 先跑通 Transformers 作为对照；已有可靠 NPU 生成后端可复用，不因为后端差异改变提示词和采样协议 |

服务器 Codex 应先检查实际代码再决定复用位置。本文没有核验当前私有或服务器 checkout 的训练入口，不能假设仓库已实现上述功能。不要为这轮强制迁移到另一个大框架，也不要修改旧实验输出。

建议新增目录为 `experiments/math_single_10lang/`，保存配置、数据 manifest 和报告；训练权重放服务器持久存储的独立输出根目录。具体绝对路径由服务器现有磁盘布局决定，日志中写清楚。

## 4 模型与硬件

默认使用 **Qwen/Qwen3-4B，FFT，全参数更新，thinking=True**。使用支持 thinking/non-thinking 切换的原始 Qwen3-4B 后训练模型；不要误用 Qwen3-4B-Base、Qwen3-4B-Instruct-2507 或 Qwen3-4B-Thinking-2507，这些是不同快照和模式。十语言所有 Base、single 和后续顺序实验使用同一模型 revision、tokenizer revision、chat template 和 special token。

选择 4B 是为了在数学与目标语言输出上保留更好的可学习空间，并与 COPSD 多语言实现的 4B 路线衔接。它是建议主规模，并非已证明优于本项目的 1.7B。若 4B 在实际设备上无法稳定 FFT，采用 Qwen/Qwen3-1.7B 跑完整独立实验轨道，重新准备 token 长度统计与 Base 评测；不在十语中间换规模。旧 Qwen2.5 实验仅作历史背景，不能沿用其 Base 分数。

这轮仍是 single SFT 筛选；论文采用 Qwen3 不意味着同模型就复现了论文。COPSD 的 LoRA、特权教师、reverse KL、采样协议与本轮 FFT SFT 不同，逐项记录差异。

Qwen 官方模型卡明确建议 thinking 模式使用采样，避免 greedy 的退化和重复。主生成参数在第 8 节固定。训练完整序列仍为 2048，但思考与最终答案都计入；使用短而完整的目标语言推理标签。主评测生成上限 4096，这是与训练长度不同的独立预算，不能把固定训练 2048 理解成推理也只能生成 2048。4096 是本项目成本折中，并非完整长推理能力评估；报告截断率和必要的长度敏感性检查。

优先使用现有可运行的 CUDA 环境，例如 A800 80GB；NPU 可使用已验证环境。CUDA 与 NPU 最好分别成批，不让某语言固定落在一种后端以免混入硬件差异。记录 torch、transformers、datasets、accelerate、torch_npu/CANN（若使用）、vLLM 和评分器版本，不盲目升级已有环境。

CUDA 可用 FlashAttention 2；NPU 使用兼容实现，不强行安装 CUDA flash-attn。显存不足优先启用 gradient checkpointing 或降低 micro batch、增加梯度累积，保持全局 batch 不变。

## 5 数据来源与审计

### 5.1 优先使用的现成数据

优先检查 https://huggingface.co/datasets/yihongLiu/COPSD-PolyMath-TrainDataset 。数据卡列出十个语言 split，各 3000 条，并包含 `source`、`problem`、`solution`、`Answer`、`problem_translated`、`language`、`language_short`。这里的 PolyMath 名称指对应语言设置，不能因此认为 benchmark 测试题允许用于训练。

必须实际读数据：核验 `problem` 和 `solution` 的语言，确认英文原题与目标语言题目关系，检查 `Answer` 类型和可解析性。不能假设有现成的目标语言解答；数据卡没有提供独立的 `solution_translated` 字段。

采用 **COPSD 已发布 OpenThoughts 衍生数据作为首选问题池**，原始备用来源为 https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k 。备用数据只使用数学问题，保留原始来源和元数据；科学、代码、谜题不自动纳入。先审计发布数据的来源和质量，只有合格数据不足时才扩展原始问题池。

### 5.2 原题对齐与划分

以英文原题的稳定 ID 为单位划分，所有语言共享同一个划分。没有原始 ID 时，建立规范化英文题目哈希，并保留原文和原始行位置。规范化可以统一空白与 Unicode，不能删掉会改变题意的数字、符号和条件。不能假定十个 split 的第 n 行一定对应同一原题。

执行顺序为：原题对齐与去重 → 测试污染检查 → 建立公共原题池 → 固定 train/dev ID → 准备目标语言解答 → 长度与质量审计 → 冻结可用 ID 清单。替换不合格题时只从预先打乱的备用 ID 中取，不根据模型测评成绩选择题目。

建议第一轮规模：

| 划分 | 每种语言的原题数 | 用途 |
|---|---:|---|
| train | 1000 | 十语言使用同一组原题，训练两轮 |
| dev_tune | 200 | 共享配方校准与 checkpoint 选择 |
| dev_select | 200 | 固定配置和 checkpoint 后，语言筛选 |
| reserve | 剩余合格题 | 质量替换、后续扩大训练或另立顺序题目池 |

使用 seed=42；按来源、题型与粗略难度分层划分，有可靠标注就使用，无标注不能编造“难度一致”。总数不足时先报告原因并采用所有语言一致的较小规模，不能让十种语言悄悄使用不同原题池或不同题数。

train、两个 dev 集与 benchmark 测试题都应按英文原题检查重叠。至少检查 GSM8K test、MGSM/Rev2 原题及 PolyMath。准确去重后，再对近似匹配候选抽查；没有共同 ID 时保存方法与残余风险。禁止把公开测试题翻译后放入训练或开发集。

十语言 single 使用同一组训练原题，有利于比较语言可学习性；后续顺序实验则优先按阶段分配不重叠的原题 ID，控制重复学习同一道数学题的收益。两种设计用途不同。

### 5.3 目标语言解答准备

普通 SFT 学习的是目标语言的完整解答，不能把英文 `solution` 直接作为十种语言共同的 assistant 标签。

推荐两步：先针对英文题目、英文参考推理和金标准答案，建立短而完整的英文参考解答；再用同一模型、同一提示模板翻译为十种目标语言。保持推理步骤、条件、数学表达与最终答案，不要求逐字翻译长推理日志。所有语言必须使用同一英文解答作为语义源，避免分别自由解题造成难度和解答质量差异。

解答准备器使用服务器上可用且经小样本检查的较强多语言模型。默认由环境变量指定模型路径，先检查现有 Qwen3-8B 等可用模型是否能完成短解答和十语言翻译；对某语言检查不通过时，应换可靠生成器或人工修订。不要让 4B 学生未经检查生成自己的标签作为默认可靠监督。数据生成器与后续 COPSD 特权教师是两种角色，必须分别记录。

如没有合格的本地生成器且没有已授权 API，Codex 应完成其他数据和代码工作，报告缺少的模型或凭据，不擅自调用付费服务。生成器选择是主要尚待服务器确认的依赖。

统一回答结构：assistant completion 包含 `<think>目标语言的短完整推理</think>`，之后给目标语言的简短最终结论及一个 `\boxed{最终答案}`；公式和最终数字不需要翻译。`solution_target` 可以保存可读推理，另以 `reasoning_target`、`final_target`、`assistant_completion` 保存分段内容及实际训练 completion；标签中必须有正确闭合的思考结束符和可核验最终答案。生成一次后冻结，不能随每个学生 checkpoint 重新生成标签。

质量检查：

1. 对每条记录检查数字、单位、实体、条件及最终答案，解析并比对金标准；最终答案正确不代表推理过程正确。
2. 检查文本语言与英文夹杂，对低置信度、过短、公式占比过高样本标为待核查，不能自动视为语言保持合格。
3. 每种语言分层抽查至少 30 条 train 和 20 条 dev，涵盖长解答、复杂题、疑似英语夹杂和随机题。用独立于生成器的审查方式，结合原题；无法人工判断的语言保留核查限制，不把回译或同一模型打分称作人工验证。
4. 若明显题意改变、推理错误或语言错误超过抽查的 10%，先修订该语言标签并扩大抽查，暂缓其正式训练。该门槛是工程门槛，不是质量置信保证。
5. 确认 `\boxed{}` 标签没有仅复制正确答案而省略必要推理。验证完整解答与正确性，不能只看答案字符串。

保存字段至少包括：`source_id`、`problem_id`、`problem_hash`、`source`、`language`、`question_en`、`reference_en_original`、`reference_en_short`、`question_target`、`solution_target`、`gold_answer`、`split`、`generator_id`、`generator_revision`、`generation_prompt_hash`、`quality_status`、`input_tokens`、`supervised_tokens`、`full_tokens`。

### 5.4 训练数据是怎样得到的

SFT 训练对为“目标语言题目 → 目标语言参考解答”。它不是从 MGSM-Rev2/PolyMath 测试题生成，也不是学生 rollout。制作链条为：

1. 下载固定 revision 的 COPSD-PolyMath-TrainDataset，读取英文问题、英文原解答、gold answer 与已翻译的目标语言问题。
2. 按英文原题 ID 对齐十语，在任何模型测评之前去重、检查测试污染并固定 train/dev 划分。
3. 较强多语言数据生成器看到英文问题、英文原解答和答案，整理出短但完整、正确的英文推理。不要只截取长解答前 2048 token。
4. 同一短英文推理翻译为十语，检查目标问题与解答一致、数字/条件/答案不变，得到 `reasoning_target` 和 `final_target`。
5. 组装 Qwen3 assistant completion，核验完整 chat 序列 ≤2048，保存源文本、版本、哈希、质量状态和生成成本。
6. 学生训练时只看到目标语言问题并学习目标语言 completion；英文原文只在制作标签或未来特权教师上下文中出现。

例如，“有 3 盒，每盒 4 个苹果，共多少？”的英文信息用于生成/翻译监督；SW 学生的 user 是 SW 题目，assistant 是 SW 的“3×4=12”解释加 boxed 12。这个示例仅说明格式，不是发布数据的真实记录。

当前没有生成好的十语言 SFT 标签、训练权重或新模型成绩。方案上传只交付实验任务说明；服务器 Codex 的第一项实质工作就是数据审计与标签制作。参考数据卡本身不能证明英文原解答都正确，所有自动生成标签都需独立质量检查。

## 6 固定 2048 长度与训练格式

本轮 SFT **完整输入序列上限为 2048 token**，包括 system、user、assistant 解答、chat template 和结束符。不是回答最多 2048。固定 pad 到 2048，不使用动态 block size，不做跨样本 packing。

超过上限时，优先压缩语义源英文短解答再重新翻译，保持正确且完整。仍不能容纳的原题按十语言公共规则移除或用备用题替换。禁止硬截断解答尾部后训练，禁止为部分语言临时改成 4096。公共长度过滤会偏向较短题，报告被移除的来源、题型和各语言长度分布。

数据质量与公共长度审计后若凑不齐 1000/200/200，先使用统一的较小规模并记录，不反复生成直到掩盖某语言固有的数据质量问题。保存 P50/P90/P95/P99、最大长度、超长率、修订/剔除数。

使用 tokenizer 的原生 chat template，显式设置 `enable_thinking=True`。训练应使用与推理完全一致的 system/user 和 assistant generation prefix，再接经过核验的 completion 与原生结束符；不得假定通用模板会完整保留 assistant 中的 `<think>` 文本。某些模板会处理/移除历史思考，必须逐 token 检查实际序列，确认本轮训练推理标签仍存在。模板若已经输出 `<think>` 开头，completion 不重复插入；模板没有输出时按模型原生约定接入。训练与推理各保存至少五条真实序列样例。system 指令统一表达“使用指定语言逐步解题，最后给出 boxed 答案”；目标语言 user 只包含该语言题目。英文题目、英文解答和 gold answer 不得进入学生推理提示。

**Loss 仅覆盖 assistant 解答及正确的结束 token**，system/user 和 padding labels=-100。检查模板边界而非凭字符数估算 token mask。即使 pad_token_id 与结束 token 共用，也要依据 attention_mask 屏蔽 padding，保留真实结束符监督。训练时不要重复添加 BOS/EOS。

## 7 第一轮统一训练配方

| 参数 | 默认值 |
|---|---|
| 模型 | Qwen/Qwen3-4B，固定 revision，thinking=True |
| 更新范围 | FFT，全部参数 |
| optimizer | adamw_torch，FP32 optimizer 状态需实测并记录 |
| learning rate | 2e-5 |
| epochs | 2，固定数据曝光次数 |
| global batch | 16 个样本，含梯度累积与 world size |
| micro batch | 根据显存设置，优先 1 或 2 |
| betas / eps | (0.9, 0.999) / 1e-8 |
| weight decay | 0.01 |
| scheduler | cosine |
| warmup | 总优化器更新的 5% |
| max grad norm | 1.0 |
| precision | BF16；不支持时单独记录替代配置 |
| gradient checkpointing | 开启；训练 use_cache=false |
| max sequence length | 固定 2048 |
| packing | 关闭 |
| seed / data seed | 首轮 42；候选复跑 43、44 |
| logging | 每 5 个 optimizer step；TensorBoard 开启 |
| eval 与保存 | 开始前、每 0.5 epoch、终点 |

1000 条 train、两轮、global batch=16 时，预计约 126 个 optimizer step，实际由分布式 sampler、末批处理决定。训练器使用 epochs 或预先计算的步骤数，不能同时残留旧 `max_steps=15000` 导致覆盖 epoch 配置。每种语言验证实际有效样本曝光次数与 sampler 重复样本，避免分布式 padding 带来隐性差异。

先在 SW 和 ZH 上各做 20 步左右 smoke，检查损失、标签、保存和生成；smoke 权重不进入正式比较，正式跑重新从 Base 开始。若默认学习率稳定，就冻结并跑十语；若需校准，只在 dev_tune 上比较 1e-5、2e-5、5e-5 三个候选，预先按 SW/ZH 平均开发集表现选一个共享学习率，不能每语在测试集上挑最优。此轮偏向“统一配方下可学习性”，并不声称每个语言达到各自最优。

本轮标准 AdamW 结果不与旧自定义 FP32 动量优化器结果直接归因比较。后续比较 MoFO/稀疏投影时，要另建同 optimizer family 和同预算的密集控制。

### 7.1 预算解释

第一轮匹配原题、有效样本数量、原题顺序、epoch 和全局 batch，token 数因语言与标签长度不同而不同。记录每个 checkpoint 的累计非 padding 输入 token、assistant 监督 token、padding token、样本曝光次数、optimizer step、耗时和显存。

不将 pad 到 2048 后的总矩阵元素当作有效 token；不将旧项目的 15.36M 输入预算直接套入 OPD。若入选语言的监督 token 成本差异明显，应补一个固定有效监督 token 预算的可学习性对照，判断排名是否受解答长度影响。

### 7.2 Checkpoint 选择

保存 0.5、1、1.5、2 epoch 四个 checkpoint。用 dev_tune 的目标语言“正确且语言合格比例”选 best；并列时优先数学正确率，再优先更早 checkpoint。语言指标未通过校准前，用数学正确率选择并将语言状态列为单独的未解决项，不用不可靠识别器决定 best。

同时报告 best 和统一 2 epoch 终点。dev_select 只评 Base、best 与统一终点，不反过来改 checkpoint。保存开发集 CE，但不能只用 CE 最小替代生成解题效果。

## 8 评测设计

### 8.1 数据与用途

| 数据 | 评测对象与时机 | 用途 |
|---|---|---|
| dev_tune，200 原题的十语言版本 | Base、四个 checkpoint 的训练语言 | 选配方与 checkpoint |
| dev_select，200 原题的十语言版本 | Base、每语 best/终点 | 选后续语言 |
| MGSM-Rev2 | 语言及配置冻结后，Base 与十语 best/终点 | 正式基础数学结果，每目标语言 250 题 |
| PolyMath | 配置冻结后，Base 与十语 best，完整四档难度 | 更广数学泛化，按实际固定文件核验题量 |
| GSM8K test，英语 | Base 与十语 best | 英语数学能力变化 |
| 英语 IFEval | Base 与十语 best | 通用指令遵循变化 |
| 原题英文 dev_select 版本 | Base 与十语 best | 低成本英语保持开发指标 |

正式语言名单先由独立 dev_select 决定，再解锁 MGSM-Rev2/PolyMath 测试结果。这样可以研究“经过开发集筛选的三种语言”，而不是把测试集用作语言搜索工具。十语测试成绩仍完整报告，入选后的结论应明确只适用于这一筛选范围。

PolyMath 官方 README 描述四档、每档每语 125 题，即每语 500 题；上传的论文对题数描述有差异。执行时以固定版本数据文件为准，记录每档题数、去重数与总分母。完整官方指标按官方评分代码计算，另报告四档准确率及普通宏平均；不自己猜 DW-ACC 权重，不把自定义聚合称作官方指标。

### 8.2 统一生成协议

本轮数学主评测采用 **thinking 模式的采样单次回答准确率**：`enable_thinking=True`、`do_sample=true`、`temperature=0.6`、`top_p=0.95`、`top_k=20`、`min_p=0`（后端支持时）、`max_new_tokens=4096`、`num_return_sequences=1`。默认不加 presence/repetition penalty，冻结生成 seed。参数来自 Qwen3 官方建议，生成长度是项目成本预算。

dev_tune 固定 generation seed=17，用于选择配置和 checkpoint；dev_select 对 Base、best 和终点固定 generation seeds=17、18、19，每个 seed 每题一次生成，报告三次单次回答准确率的平均与差异，不用“任意一次正确”冒充单次准确率。比较使用同一组题目与生成 seeds。训练 seed 与生成 seed 分开记录；固定采样 seed 不保证跨后端完全一致。

正式数学测试默认 seed=17 的一次回答，入选三语可再补 18、19 作采样稳定性检查。thinking 模式不沿用旧 greedy 主协议。IFEval 是一般指令遵循评测，单独固定 `enable_thinking=False` 与其任务协议；Base 和所有 checkpoint 保持一致，不把思考文本直接交给格式严格检查器。GSM8K 英语数学使用与目标语言相同的 thinking 采样参数。

训练完整序列上限是 2048；推理 4096 只限制新增 token。推理上下文窗口须容纳输入与输出，按真实模型配置核验，不对题目错误截断到 2048 总长。入选模型和对应 Base 补充 2048/4096 的固定生成预算对照，检查提升是否主要来自更快终止。若开发集出现大于 10% 的“没有完成最终答案且达到长度上限”，先报告并在开发集上统一做 8192 敏感性检查；不按各模型成绩分别增减上限，也不根据正式测试结果改主预算。4096 有限预算可能低估长思考，结论应表述为该固定预算下能力。

每条题目保存完整 prompt、原始输出、token 数、结束原因、是否超长截断、提取答案、评分和语言识别状态。达到生成长度上限的回答仍进入分母，能提取正确答案就按评分规则处理，同时独立报告截断率。

统一目标语言解题指令对 Base 与微调模型都使用，明确要求思考和最终解释均用目标语言。主协议先不使用 COPSD 手写语言前缀；若作为论文复现补充加入，必须对 Base 与所有方法一致，另立协议。不要只给微调模型语言引导。MGSM-Rev2 使用本项目零样本协议，属于明确自定义推理设置；要对照某论文数值需重跑其相同版本和协议。

后续对齐 COPSD 时另增其采样 pass@12：固定温度、top_p、生成上限与采样种子，并记录成本；不能拿 pass@12 与单次回答准确率直接比较。

### 8.3 答案评分

先将思考区与 `</think>` 后的 final 区分开，主评分优先提取 final 区的最终 boxed 答案；没有闭合思考或没有 final 区则标记未完成，不能从思考中找正确答案充当 final。必要 fallback 只允许从明确最终答案区提取，并提前固定规则。没有可靠最终答案时记错，不能从中间计算任意找一个正确数字。检查多 boxed、分数、小数、负数、百分数、单位及本地数字字符。MGSM 数值答案使用精确规范化；PolyMath 使用官方数学评分流程。

以固定全部题目为分母。预测提取失败记错；gold 解析失败是评分器/数据错误，需要修复或明确报告，不能自动删题使准确率虚高。加入少量必要评分回归样例，包括本地数字、多答案与解析失败；记录 parse_failure_rate。

### 8.4 输出语言与不确定性

公式、数字、代码块和 boxed 最终数字从语言识别文本中剔除，分别检查思考区和最终解释区的自然语言，报告两部分语言合格率与英语切换；不能只检查最终结论掩盖英文思考。使用覆盖十语、经样例校准的识别器；不能只按文字系统识别（日中共享汉字，SW/DE/FR/ES 共享拉丁字母）。无自然语言、过短和低置信度输出单列为 unknown。

报告数学正确率、目标语言合格率、英语切换率，以及“答案正确且目标语言合格”的比例。公式正确但解释全英文不算目标语言输出合格；只有数字的回答可能数学正确，但目标语言指标应为不可判断。

dev_select 的 Base→single 提升以题目配对 bootstrap 给出 95% 区间（如 2000 次重采样），报告题级明细。bootstrap 衡量题目采样不确定性，不替代训练随机种子复跑。MGSM-Rev2 250 题每题对应 0.4 个百分点，不凭几题差异声称稳定胜出。

第一轮每语 best 只跑训练语言与英语；若需理解对其他候选语言的影响，可在冻结 checkpoint 后跑十语 dev_select 的 10×10 迁移矩阵。这是补充诊断，不算顺序遗忘证据。

## 9 怎样从十种语言中选三种

用户关注提升高，因此以 dev_select 的数学正确率提升为主要排序，并设输出语言与稳定性门槛。不必预先保留 SW；若 SW 不符合条件就如实报告。

建议筛选流程：

1. 先淘汰标签质量或评分器存在未解决问题的语言，区分数据不可用和模型难学习。
2. 使用 Base→best 的正确率提升排序，同时展示终点提升、联合正确率、训练后绝对水平、英语保持及 token 成本。不能只看相对百分比提升。
3. 以提升 ≥5 个百分点、语言合格率 ≥80% 作为优先候选门槛；这些只是预实验默认门槛，必须在看 dev_select 结果前冻结。未达到并不等于该语言没有研究价值。
4. 取前 4 个可用候选，追加 seed=43、44。三个 seed 都从相同 Base 开始，改变训练随机性；原题划分、标签、prompt 和共享配方不变。
5. 按三个 seed 的平均提升选前三，同时报告标准差、各 seed 分数和 best/终点差异。若差异接近噪声或均无法达门槛，暂不强行挑三种“有效语言”，先排查训练规模、教师标签或模型能力。
6. 有相近候选时，优先考虑可核查的标签、教师信号、训练后仍有提升空间与语言背景多样性；不能在看到正式 benchmark 后换语言来提高论文结果。

若模型在某语言训练后仍接近零分，很难研究其遗忘；若 Base 已高、提升小，也可能适合高资源保持对照，不能简单说它“没有用”。建议保存一套按提升选出的三语主设置，同时列出高初始能力语言作为后续补充候选。

候选复跑成本为十语首轮 + 四语各两个追加 seed，共 18 次正式训练；不含 smoke 和共享学习率校准。先在 smoke 测每步时间与评测吞吐，再估算总耗时，不承诺未知机器的小时数。

## 10 Single 筛选后的教师检查与一语言对照

本项目完整路线为：十语言 single 筛选 → 特权教师检查 → 一语言 A/B/C 对照 → 三语言顺序 A/B/C → 必要时加入保持机制。后续阶段按启动条件推进，不将实验设想记为已完成结果。

### 10.1 教师检查

入选三种语言后，从训练侧独立开发题取每语约 100—200 题，先检查同一冻结原始 Qwen3-4B 在三个上下文下的行为：仅目标语言题目；目标语言题目加英文题目；再加英文参考解答。英文参考信息仅用于教师，不能泄漏到学生输入。

比较正确率、思考与最终回答语言、终止率，并核查教师在学生错误前缀上是否提供有用的下一步监督。只会复述参考答案不足以证明教师能指导学生轨迹。补充对学生分布的 token 级检查及人工/独立审查案例，不仅看教师另生成完整答案的表现。

第一版特权教师使用固定的原始 Qwen3-4B，学生与教师 tokenizer 相同。教师检查无可靠优势时，先排查上下文和翻译质量；必要时另立更强外部教师路线，记录其成本与变化，不把它称作同模型自蒸馏。标签生成器与特权教师是不同角色。

### 10.2 一语言 A B C

先在入选语言中选择一个做机制实验，使用独立于 single 标签调优的开发检查。选择依据预先固定，例如在入选三语中取 single 平均提升最大的语言；不能根据 A/B/C 测试结果再选有利语言。

| 组别 | 训练回答前缀 | 学习信号 | 学生输入 |
|---|---|---|---|
| A 普通 SFT | 固定目标语言参考解答 | 硬标签 CE | 目标语言题目 |
| B 固定前缀蒸馏 | 与 A 相同的目标语言参考解答 | 冻结特权教师分布 | 目标语言题目 |
| C on-policy 蒸馏 | 当前学生自己生成的回答 | 与 B 相同的冻结特权教师分布 | 目标语言题目 |

三个组各自从相同原始 Qwen3-4B 开始，不从 single best checkpoint 接续。A/B 使用同一套经检查的目标语言参考解答；英文解答只能进入 B/C 教师上下文。A→B 同时改变监督形式和特权信息，不完全隔离软标签作用；B→C 主要检查前缀来源。

B/C 固定同一教师、英文信息、KL 方向、温度、FFT 更新范围和问题池。第一版用全词表 reverse KL：
`KL(p_student || p_teacher)`，仅对有效回答 token 计算，包含实际思考和最终回答；教师分布 detach，rollout 采样不反向传播，更新学生分布。不能 B/C 各用一种 KL，也不能将 sampled-token 或 top-k 近似称作全词表 KL。

第一版不同时引入 token gate、稀疏参数更新、replay 或英语保护。先查明 on-policy 前缀本身的贡献。B/C 的具体蒸馏温度和 rollout 参数在看开发结果之前冻结，区别蒸馏温度与生成 temperature；rollout 暂沿用 Qwen3 thinking 采样参数。将损失按有效回答 token 归一化并记录长度效应。

### 10.3 C 的一次更新

1. 当前学生只看到目标语言题目，在线生成目标语言思考与回答。
2. 冻结教师看到该题、英文问题、英文参考解答以及同一学生回答前缀。
3. 学生和教师在相同回答位置预测下一 token，计算分布差异。
4. 根据回答 mask 聚合 KL，更新学生。
5. 下一批题目使用更新后的学生重新生成轨迹。

教师不需要另生成完整答案给学生照抄。学生轨迹无需全部正确；错误前缀正是教师监督需要检查的状态。不能预先生成一次轨迹后在整个训练中重复使用，再把它称作严格 on-policy。若后端存在 rollout 权重同步延迟，记录版本差和同步规则。

### 10.4 预算与启动顺序

先做 20 个优化器更新左右的 smoke 验证对齐与 loss，再分别重新从 Base 做固定问题曝光次数的短程 A/B/C。三组匹配同一问题清单和有效曝光次数，B/C 进一步匹配教师调用与更新次数；分别统计参考监督 token、学生生成 token、教师评分 token、有效 loss token、优化器更新、峰值显存和耗时。单一 token 指标不能代表 SFT/OPD 计算相同，正式比较需补计算预算或成本曲线。

SFT 的 2048 完整序列预算不直接套到 OPD：独立规定 student prompt、completion、teacher prompt+英文解答+学生前缀的窗口；参考代码的 2048 completion 不等于本项目的 2048 total。阶段启动前冻结长度协议并验证模型上下文容纳，超长不能静默截断英文特权信息或学生 prefix。若为了成本选择更短 rollout，B/C 及相应长度敏感性对照须明确记录。

FFT 自蒸馏需单独冻结教师副本。共享学生权重的 stop-gradient 前向仍会随学生更新而变化，不能称为固定教师。官方 COPSD 的移除 LoRA adapter 固定教师方式不能直接照搬到 FFT。

一语言结果至少有多训练 seed 和开发集配对比较，同时看正确率、目标语言、终止和成本。C 相对 B 有稳定且实际有意义的收益后，推进完整三语 on-policy 顺序实验。B/C 接近时先研究教师信号，不强行宣称 on-policy 优势；普通 SFT 顺序诊断仍可开展。一次小样本不显著也不等于没有作用，应结合区间和成本判断是否值得扩展。

## 11 三语言顺序微调与保持机制计划

### 11.1 顺序初始化与阶段连接

从 single 入选语言记作 L1/L2/L3。先运行两条不同起点的顺序：L1→L2→L3 与 L3→L2→L1。英语只用于保持评测，不计一个训练阶段。

每条顺序、每种方法都重新从原始 Qwen3-4B 开始；后续阶段只接续该方法自己的上阶段学生权重，不能将三个 single 模型拼接或从某个语言的 single best 起步。

| 方法 | 第一阶段 | 第二阶段 | 第三阶段 |
|---|---|---|---|
| 顺序 SFT | Base → L1 SFT | 接续权重 → L2 SFT | 接续权重 → L3 SFT |
| 顺序固定前缀蒸馏 | Base → L1 B | 接续权重 → L2 B | 接续权重 → L3 B |
| 顺序 on-policy | Base → L1 C | 当前学生 → L2 C | 当前学生 → L3 C |

建议每阶段重置优化器状态和学习率调度器，只接续权重；所有方法一致。另一条顺序同理。优化器延续可另作消融，不能有的方法延续、有的重置。

B/C 第一版整个序列固定同一原始 Qwen3-4B 教师，特权题目/参考解答随题目变化。C 每阶段 rollout 来自当前接续学生，不能复用 single 阶段轨迹。更换教师快照、EMA 教师、阶段教师都属于后续独立消融。

### 11.2 阶段数据与预算

按英文原题 ID 分配各阶段互不重叠的问题池，匹配来源、题型、粗略难度和标签质量。同一阶段各方法使用相同池和数据顺序。语言与问题池绑定方式预先冻结；第二条顺序移动同一语言及其问题池，有助于检查顺序影响。共享原题重复三语训练只作补充对照。

执行前重新审计可用公共原题数量，禁止把“1000 条 single 问题池”直接当作足够支持三个互不重叠的 1000 条阶段池。若数据不足，统一减小阶段池或扩充独立训练来源，并保持 dev/test 排除规则。入选语言的 single 控制需补测与正式阶段池、预算对应的独立适配，旧 single 仅作筛选参考。

正式 SFT/稀疏/保护方法主比较固定每阶段非 padding 有效监督 token 预算，并按相同 token 间隔评测；既报告预算终点，也报告预算内按开发集规则选择的 best。每阶段预算由先导学习曲线和各语言长度审计确定，在看正式测试之前冻结，不沿用旧 15.36M/15000 步。语言长度不同会导致样本曝光次数不同，必须一并报告。

跨 SFT/B/C 主机制对照另以相同问题曝光次数为控制，报告全部 token 成本与耗时，并补预算敏感性。固定问题曝光对照和固定有效监督 token 对照是不同协议，分表标识，不能声称两者同时完全相等。B/C 尽量匹配教师评分 token/更新成本并检查 rollout 长度带来的差异。

### 11.3 阶段评测与结果

Base、第一/第二/第三阶段结束后都评全部三种目标语言与英语，保存完整阶段×语言矩阵。开发集用于 checkpoint 选择；正式 MGSM-Rev2/PolyMath 按冻结设置评测。沿用思考/最终回答语言和结束检查，以及英语 GSM8K/IFEval；有资源时补旧项目通用能力评测。

报告：
- 当前语言相对该阶段开始前的获得；
- 每个旧语言相对其学习后 checkpoint 的下降；
- 旧语言历史最佳阶段分数减当前分数的 forgetting，并保留负值表示正迁移；
- 最终三语言平均正确率及联合语言正确率；
- 英语能力变化、终止/截断率、训练和生成成本。

所有指标明确百分比或百分点，保留原始矩阵。OPD 当前语言更强不等于旧语言忘得更少；B 与 C 差异才更直接检验学生轨迹的作用。只有两条顺序不能主张任意顺序有效，后续补不同起点、训练 seed，再考虑第四种语言。

### 11.4 若仍遗忘，加入保持机制

on-policy 不自动保护旧语言。基线三组显示明显遗忘后，每次只加入一个模块，保留未加模块的原方法：

| 补充实验 | 检查的问题 | 预算要求 |
|---|---|---|
| SFT + replay 与 C + replay | 旧问题回放在两种监督下是否都改善保持 | 相同 buffer 原题、回放比例和总曝光/计算统计 |
| C + 旧语言或英语保持约束 | 额外保持信号是否改善获得与遗忘的取舍 | 独立报告校准/教师调用成本 |
| C + 稀疏更新，对照同设置密集 C | 参数选择是否帮助保持 | 相同 optimizer family、目标数据和监督协议 |

replay 可以从每阶段训练池按预先固定规则保存小缓冲（例如每旧语言 50—100 题作为待校准起点），不能取 benchmark 测试题。区分回放旧问题并重新 rollout 与回放固定参考解答，两者监督不同。回放比例在开发集上决定后冻结，并建立总预算对照；不能偷偷增加数据然后全归因方法。

RP-OPSD 的 reference anchor 可作后续 token 监督参考，但冻结原始 Base 语言锚定不等于保护已学旧语言数学能力，必须用阶段矩阵验证。MoFO/SSU 或现有稀疏方法加入时，明确其参数更新机制与 OPD token 监督机制的不同。

### 11.5 后续任务交付与状态

当前实施范围仍是十语 single。后续启动按以下条件记录：

| 阶段 | 启动条件 | 交付 |
|---|---|---|
| 教师检查 | single 候选与数据质量明确 | 教师上下文比较、错误前缀案例、可靠性结论 |
| 一语言 A/B/C | 教师信号可靠、位置/mask/长度核验通过 | 多 seed 比较、成本和是否扩展判断 |
| 三语顺序 A/B/C | 语言名单和预算冻结；C 有值得扩展的证据 | 两条顺序完整阶段矩阵与遗忘报告 |
| 保持机制 | 基线存在需要解决的获得/保持问题 | 单模块消融、同预算与成本对照 |

Codex 可以在 single 执行期间预先准备后续配置、数据接口和评测矩阵，但不因本文件列出计划就同时启动全部训练。阶段条件不满足时记录证据与下一步，不能填造结果。后续新授权或用户明确启动时，先读取实际 single/教师报告，再将这些建议落实为冻结运行配置。

## 12 三篇参考论文的阅读任务

读取用户上传的本地 PDF；以下文件名应由服务器提供实际路径，不能凭名称猜服务器位置。

| 论文 | 必读内容 | 对本项目的用途 |
|---|---|---|
| Liu 等 2026，Crosslingual On-Policy Self-Distillation for Multilingual Reasoning | 方法、数据、语言提示、冻结教师、全词表 reverse KL、LoRA、推理附录 | 后续 C 组的基础实现与英文特权上下文；区分论文复现与本项目 FFT 控制 |
| Heo 等 2026，On-Policy Delta Distillation for Multilingual Math Reasoning | OPD 与 OPD2 奖励、teacher 及其对应 base、语言切换分析 | 更强外部教师路线与增量蒸馏参考；不可直接当作 COPSD 自蒸馏同一设置 |
| Wang 等 2026，RP-OPSD Reasoning-Pivot-Guided On-Policy Self-Distillation for Multilingual Reasoning Transfer | 双教师视图、PRS gate、forward KL、冻结 reference anchor、消融 | 后续探索“哪些 token 值得接受特权推理监督”，区分 token 权重与参数稀疏更新 |

关键区别：COPSD 是学生轨迹上的跨语言特权教师蒸馏；OPD2 利用 post-trained teacher 与该 teacher 对应 base 的差分信号，不能随意用学生 Base 替代 teacher base；RP-OPSD 对有/无英文参考推理的教师视图求差异，用 gate 路由监督，并用冻结 reference 做语言锚定。RP-OPSD 的 token 路由不是 MoFO/SSU 的参数筛选，语言锚定也没有直接证明解决旧语言遗忘。

论文中 single 数学收益不等于顺序持续学习证据。Codex 应整理“论文做了什么—本项目复用什么—新增验证什么”的简短对照，重点核查冻结教师与 stop-gradient 教师区别、KL 方向、题目数和生成上限。

代码参考入口：COPSD https://github.com/cisnlp/COPSD ；OPD2 https://github.com/naver-ai/opd2 。RP-OPSD 仓库链接从上传 PDF 和论文原始页面核验后记录，不能凭名称推断。现阶段不要求实现三篇方法全部细节。

## 13 Codex 的交付物与执行顺序

建议在主仓库形成以下新增组件，名称可根据现有组织调整：

| 组件 | 功能 |
|---|---|
| audit_math_data.py | 检查 schema、十语原题对齐、去重、答案和测试污染 |
| build_math_sft_data.py | 固定原题划分、短英文解答、目标语言翻译、质量与 2048 审计 |
| train_math_single.py | FFT assistant-only SFT、eval、TensorBoard、token 与样本计数 |
| eval_math.py | 统一生成、答案解析、语言检查、逐题输出 |
| summarize_math_single.py | Base/best/终点比较、配对区间、语言排名与成本 |
| configs/math_single_10lang.yaml | 全部默认值、语言映射、数据和模型版本 |
| scripts/run_math_single_10lang.sh | 队列训练、失败停止、跳过已验收任务和断点恢复 |
| DATA_AUDIT_ZH.md | 数据来源、剔除、质量检查、长度与污染结果 |
| SINGLE_RESULTS_ZH.md 与 CSV | 十语言结果与语言推荐；保留完整候选表 |
| HANDOFF_MATH_SINGLE_ZH.md | 实际运行命令、环境、输出路径、失败项与下一步 |

执行阶段：

1. 检查仓库、环境、硬件、磁盘和可用模型；记录 commit，保留用户现有改动。
2. 下载固定 revision 的训练问题池和评测数据，完成对齐、划分与污染审计。
3. 在小样本上验证解答生成/翻译器，生成并冻结十语言标签，完成质量与长度检查。
4. 实现或复用训练与评测，做 SW/ZH smoke；检查 labels、结束 token、推理 prompt 无答案、checkpoint 可加载。
5. 冻结共享配方，跑十语言 seed=42，期间开启开发集评测和 TensorBoard。
6. 用 dev_select 整理候选，复跑前四种语言 seed=43、44，冻结三语名单。
7. 跑正式 MGSM-Rev2、PolyMath 和英语保持评测，输出十语全表、候选三 seed 表和逐题数据。
8. 写交接报告，给出第 10—11 节后续各阶段的启动条件是否满足，以及下一阶段的具体任务与缺口。后续蒸馏与三语顺序按条件另行执行。

断点恢复需验证同 run 的模型、数据 manifest、prompt、配置和 sampler 状态；恢复同一 run 可以继续，独立语言不能 resume 前一种语言。汇总脚本只纳入验收通过的完整评测任务，失败项显示原因，不填零冒充有效结果。

必要验证聚焦三个问题：assistant loss mask 和 padding 是否正确；英文/答案是否泄漏学生输入；评分与分母是否正确。验证数据划分、十语原题顺序、训练初始化和 checkpoint 读取。不要用单纯重复实现的测试堆数量。

## 14 验收标准与仍需环境确认的事项

成功交付需要：十语同 Base、同原题和统一训练配方；参考解答可靠且未被硬截断；开发集选择与正式测试分离；每次训练有 eval/TensorBoard/成本统计；报告 Base、best 和终点；候选复跑与语言保持检查完成；模型和逐题输出可复查。

可以直接采用上述默认模型、1000/200/200 规模、FFT、2e-5、两轮和 Qwen3 thinking 采样协议开展工程工作，不必逐项重新询问。服务器侧需要确认的实际依赖只有可用训练设备/模型路径、合格的标签生成器和剩余磁盘空间。资源受限可先完成十语数据与评测，再按队列运行；不能声称尚未完成的训练已成功。

若标签生成器无可靠能力、公共合格样本不足、教师信号无优势或候选提升不稳定，应给出已完成工作、具体证据和最小下一步，不硬凑三语言结论。第一轮的价值是筛出可重复的能力获得，再进入蒸馏和顺序保持。

## 15 官方参考链接

- MGSM-Rev2：https://github.com/google-research-datasets/MGSM-Rev2
- PolyMath 数据：https://huggingface.co/datasets/Qwen/PolyMath
- PolyMath 代码与指标：https://github.com/QwenLM/PolyMath
- COPSD 代码：https://github.com/cisnlp/COPSD
- COPSD 十语言训练问题池：https://huggingface.co/datasets/yihongLiu/COPSD-PolyMath-TrainDataset
- 原始 OpenThoughts：https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
- Math-Verify：https://github.com/huggingface/Math-Verify
- Transformers Trainer：https://huggingface.co/docs/transformers/en/main_classes/trainer
- Chat template：https://huggingface.co/docs/transformers/en/chat_templating
- 项目仓库：https://github.com/lxy308297917-cloud/Multilingual-CL

这些链接需固定到执行时的具体 revision 和版本；后续更新不应静默改变正在比较的实验。
## 16 交给服务器 Codex 的任务指令

请读取本文件并完成本轮十语言 Qwen3-4B single FFT 数学 SFT。先检查当前 checkout、AGENTS.md（若存在）、现有入口、硬件和可用数据生成器；保留已有修改。按第 13 节实施数据审计、目标语言标签制作、smoke、十语训练与筛选评测，训练开启 eval 和 TensorBoard。不要先运行顺序 CL 或 OPD。缺少合格数据生成器时先完成可独立完成的代码和审计，明确报告阻塞，不擅用付费 API。模型或显存导致失败时报告实际错误和最小修复，不能暗中切换 LoRA、模型规模或长度。交付实际运行命令、冻结配置、数据 manifest、逐题输出和进度报告；未跑的任务不能报告完成。

本次 GitHub 交付是任务文档，不包含已实现的新训练脚本。服务器 Codex 需实现/复用脚本后才能运行。后续三篇论文从官方论文入口读取；用户上传 PDF 若已放服务器则优先读本地原文，无本地副本时按第 15 节链接及官方仓库定位论文，不猜文件路径。

Qwen3 官方模型卡：https://huggingface.co/Qwen/Qwen3-4B 。规模备用：https://huggingface.co/Qwen/Qwen3-1.7B 。
