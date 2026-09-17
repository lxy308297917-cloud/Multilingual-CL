# 多语言持续预训练实验统一规范

本文件是本项目的 Codex 持久化工作规范，也是人工阅读的唯一总入口。凡涉及 SSU、多语言持续学习、持续预训练、语言神经元或 multilingual CL，开始操作前必须完整读取本文件，再读取 configs/multilingual_cl_v3_2_lowresource_bn_te_sw.json。该 JSON 是当前活动轨道的机器可执行参数源；报告默认使用中文。

## 唯一代码与数据位置

- 主代码：`/root/autodl-fs/ssu-project/ssu-main`
- 当前正式配置：/root/autodl-fs/ssu-project/ssu-main/configs/multilingual_cl_v3_2_lowresource_bn_te_sw.json
- 旧 v2 配置与 DE→ES→SW 结果为只读先导档案，不得覆盖或混入 v3 正式统计。
- 当前 v3.2 低/中低资源实验产物：/root/autodl-fs/ssu-project/experiments/multilingual_cl_v3_2_lowresource_bn_te_sw；临时数据与筛选检查点位于 /root/autodl-tmp/multilingual_cl_v3_2_lowresource_bn_te_sw。
- 已完成的 v3.1 RU→AR→SW 产物与 configs/multilingual_cl_v3_task.json 均转为只读对照，不得覆盖或混入 v3.2 正式统计。
- `/root/ssu-github/ssu-main` 是迁移前旧副本，不得作为正式运行入口，也不得把新改动只写在那里。

## 固定环境

训练与评测采用按角色固定的两个环境，不允许临时换环境后把结果混入同一正式轨道。

| 角色 | 唯一解释器或入口 | 已锁定版本 |
|---|---|---|
| 训练、PPL 及离线 GlotLID 诊断 | `/root/miniconda3/envs/cl/bin/python` | Python 3.10.20；PyTorch 2.5.1+cu121；Transformers 4.57.6；Datasets 4.8.5；Accelerate 1.7.0；fasttext-wheel 0.9.2 |
| Lighteval 任务评测 | `/root/miniconda3/envs/cl_eval_latest/bin/python` 与 `/root/miniconda3/envs/cl_eval_latest/bin/lighteval` | Python 3.10.20；PyTorch 2.5.1+cu121；Transformers 4.57.6；Datasets 5.0.0；Accelerate 1.13.0；Lighteval 0.13.1.dev0 |
| 基础模型和 tokenizer | `/root/models/Qwen2.5-1.5B-Instruct` | Qwen2，28 层，hidden size 1536 |
| GPU 基线 | 两张 NVIDIA vGPU-32GB | 驱动 560.35.03；每张 32760 MiB |

规则：

- 禁止用训练环境直接导入或运行正式 Lighteval 任务；该环境安装的是不兼容旧版 Lighteval。
- PPL 评测使用训练环境，因为它由项目自有的 `eval_multilingual_cl_ppl.py` 执行。
- 正式运行中禁止 `pip install`、升级、降级或修改环境。确需变更时必须建立新协议版本和新输出轨道，不覆盖旧结果。
- 所有评测必须经 `scripts/evaluate_cl.py` 进入；所有训练必须经 `scripts/train_cl.py` 进入。内部实现文件不是用户入口。
- 自定义任务只从 `evaluation/custom_multilingual_tasks.py` 加载，不修改环境的 site-packages。

## 阶段定义

- 阶段A，也称先导筛选：复用旧检查点，训练设置可能与正式协议不同，仅用于低成本寻找候选 benchmark，不得作为正式方法对比。
- 阶段B，也称正式实验：所有 Seq-FFT 和 Single 模型按完全统一协议重新训练；正式下三角、Single 上界、显著性和论文结论只来自阶段B。
- 阶段A与阶段B必须在 CSV、JSON 和报告中保留独立 `phase` 字段，严禁合并计算。

## 后续 Codex 的强制读取与启动流程

- `/root/AGENTS.md` 是工作区路由文件；它强制所有后续 Codex 在项目操作前读取本文件和正式 JSON。
- configs/multilingual_cl_v3_2_lowresource_bn_te_sw.json 是当前唯一机器可读参数源。语言顺序、数据模式、方法、学习率、训练预算、seed、环境版本和 benchmark 集合不得在其他脚本中另设默认值。
- 正式任务优先从 `scripts/run_experiments.sh` 启动。其 `preflight` 会校验冻结协议、配置 SHA-256、控制器解释器、训练环境、Lighteval 环境、模型与任务文件。
- `train` 与 `evaluate` 子命令在真正启动前自动执行同一 preflight；底层 Python 入口还会再次校验冻结字段。
- 每个新训练的 `EXPERIMENT_INFO.json` 以及每个新评测的 `job.json` 都记录 `active_protocol_id`、配置绝对路径和 `config_sha256`，用于追溯。
- 禁止通过 `CL_LANGUAGE_ORDER`、`CL_DATA_MODE`、`CL_METHOD` 或命令行参数绕过冻结协议。若实验设计确需变化，先更新 JSON、协议版本/ID和独立输出轨道，再运行 preflight。

推荐的每次启动顺序：

```bash
cd /root/autodl-fs/ssu-project/ssu-main
scripts/run_experiments.sh preflight
scripts/run_experiments.sh train --training-mode single --gpu 0
scripts/run_experiments.sh evaluate --training-mode single --gpu 1 --full-eval
scripts/run_experiments.sh summarize
```

## 当前 v3.2 协议以 configs/multilingual_cl_v3_2_lowresource_bn_te_sw.json 为唯一事实源：BN→TE→SW；MASSIVE、XL-Sum、TyDiQA、FLORES 双向 MT 为主评测。Single 筛选冻结为 5e-5、最多2000 steps、Clean-MT30、seed42，保存500/1000/2000步；三个语言只能统一选择1000或2000步。selection_status=pending_single_gate 时严禁启动 Seq。数据与检查点位于独立 v3.2 轨道；v3.1 RU→AR→SW 已完成结果只读保留。

## v2 阶段B冻结协议（只读先导档案）

以下参数以 `configs/multilingual_cl_v2.json` 的 `active_*` 字段及顶层训练字段为权威来源。若文档与 JSON 不一致，先停止新任务并核对，不得自行猜测。顶层 `language_order` 永远表示当前活动轨道；未来五语言方案只写在 `planned_five_language_order`，不得据此自动启动。

- 模型：Qwen2.5-1.5B-Instruct
- 语言顺序：`deu_Latn,spa_Latn,swh_Latn`，即 DE → ES → SW
- 数据模式：`mono`
- 方法：`seq_fft`
- 学习率：`5e-5`
- scheduler：cosine
- warmup：5%
- 每阶段：1000 optimizer steps
- batch：8
- sequence length：512
- gradient accumulation：1
- 每阶段固定 token positions：4,096,000
- seed：42
- 快速 benchmark：PPL、MT 双向、SUM、Belebele、X-CSQA、SIB-200
- MMLU 与 G-MMLU 仅在快速矩阵完成后运行。

当前活动协议 ID：`phase_b_seq_fft_mono_de_es_sw_lr5e-5_seed42`。若要改变语言顺序、学习率、训练预算、数据模式、模型、seed、环境或任务定义，必须提升 `protocol_version`，建立新输出轨道，并保留旧结果；不得在当前轨道中直接替换。

正式命令模板以本文件上方的 `scripts/run_experiments.sh` 四个子命令为准。除 GPU 编号、训练模式和安全评测分片外，不在命令行重复填写已冻结参数；这样可以避免命令文本与 JSON 漂移。

## 结果一致性与验收

- 每个正式单元必须同时保存 `job.json`、原始 result 或 metrics、逐样本 details 和 `DONE`。存在 `FAILED` 或 `FAILED.json` 时不得宣称完成。
- 控制器按 `DONE` 幂等跳过已完成单元。服务器或终端连接短暂中断后，先检查 PID、GPU 子进程和标记；进程仍存活时绝不重启。
- 两张 GPU 对同一阶段并行评测时，必须用 `--eval-languages` 指定互不重叠的语言分片；不得启动未分片的第二控制器。
- 同一轨道的模型、tokenizer、任务定义、seed、数据模式、训练预算和任务哈希必须一致。
- 重复结果不得静默覆盖。汇总器按文件修改时间显式选择，并在 `duplicates` 中保留记录。
- 报告必须给出原始下三角、Single 原始上界、BWT、平均遗忘、单调下降比例和 paired bootstrap 95% CI。正迁移、无变化和遗忘都保留。
- MT 原始表、BWT 和平均遗忘使用语料级 chrF++；paired bootstrap 使用匹配样本的逐句 chrF++ 差值均值。两者方向应一致，数值不要求完全相同。
- PPL 同时保留 PPL 与平均 NLL；paired bootstrap 和严重遗忘门槛使用逐样本平均 NLL。
- 多选指标严重遗忘门槛：acquisition 到 final 至少下降 5 个百分点且 paired CI 上界小于 0。
- MT 或 SUM 严重遗忘门槛：下降至少 3 chrF++，或相对下降至少 20%，且 paired CI 上界小于 0。
- PPL 严重遗忘门槛：平均 NLL 增加至少 0.15。
- benchmark 必须先由 Seq-FFT 筛选，再评测 SSU、首尾整层和 GCSU，禁止根据 My Method 的结果事后挑选指标。

## 每次 Codex 工作前后检查

开始前：

1. 读取本文件和正式 JSON 配置。
2. 用 `ps` 与 `nvidia-smi` 检查现有任务，避免重复占卡。
3. 检查目标轨道的 `TRAINING_DONE`、`DONE` 和失败标记。
4. 明确本次结果属于阶段A还是阶段B，不复用含义不明的旧结果。

结束或汇报前：

1. 刷新中文汇总报告和 tidy CSV。
2. 检查预期下三角是否完整、任务哈希是否一致、是否存在失败标记。
3. 报告原始分数，再报告派生统计，不只给平均值。
4. 明确当前仍在运行的单元和下一步，不把等待中的实验写成完成。

最近一次环境核对日期：2026-09-06。环境或冻结协议发生任何合法变更时，必须同步更新本文件和协议版本。


## 独立 Stage-A：Igbo LAPE 稀疏微调（2026-09-11）

当前用户授权的新工作是 `ig_lape_sparse_v1`；原 PLND 保护实验和 v3.2 Sequential 均不启动。它不改变 v3.2 的 pending_single_gate 状态。唯一新配置为 `configs/ig_lape_sparse_v1.json`，完整协议见 `analysis/IG_LAPE_SPARSE_V1_PROTOCOL.md`。通过 `CL_CONFIG=configs/ig_lape_sparse_v1.json scripts/run_experiments.sh run` 使用统一入口。输出在 `/root/autodl-fs/ssu-project/experiments/ig_lape_sparse_v1`，恢复状态和单个临时物化模型在 `/root/autodl-tmp/ig_lape_sparse_v1`。不删除历史 checkpoint。

5%/10% 是全模型 FFN 候选通道比例；最终 Igbo 更新参数分别为 19,021,824（1.232211%）和 44,255,232（2.866802%）。每个范围配对逐层数量相同且与 LAPE 不重叠的 Random；先跑四个 seed42 模型。仅按冻结下游 gate 决定是否将一个配对扩展 seed43、44，最多8个模型。PPL/NLL不参与成功判定。

启动或接续前检查 `controller.json` 中 PID、GPU、`controller_logs/`、`failures/` 和 `EXPERIMENT_DONE`，进程存活禁止重复启动。训练、全套评测、恢复及源代码身份验收均由该轨道控制器执行。当前执行进度以日志和DONE为准，不以文档快照替代。

## 当前用户任务：Igbo可信baseline（2026-09-14）

用户当前目标为Base与既有FFT/SSU统一复评，以及七个黑字方法的统一重训。LAPE保持暂停；v3.2 Sequential不启动。统一评测入口scripts/baseline_eval.sh仅传完整模型路径，目前采用configs/ig_baseline_eval_v3.json（family继承ig_baseline_eval_v1），输出experiments/ig_baseline_eval_v3。GPU0、1执行互不重叠的任务/seed分片，GPU2、3预留训练；实际已扩容至4张32GB GPU。此资源变化不修改旧v3.2冻结JSON。新训练配置configs/ig_baseline_train_v1.json共同预算已冻结，逐方法release单独放行；FFT与首尾层已启动，其余方法不得视为已启动。进度以ig_baseline_suite_v1记录和实际PID为准。EN3/Target3各由阅读、摘要及对应输出方向翻译的相对Base变化等权平均，PPL和MMLU/G-MMLU不纳入。

2026-09-14最新用户要求覆盖旧多seed计划：所有正式评测仅seed42，每任务一次；不做不必要的重复或解码对照。争取北京时间2026-09-15早上完成。旧目标描述中的生成三次已被此要求替代。当前入口说明见BASELINE_EVAL_ZH.md。用户授权的4对历史相同权重硬链接去重已完成，保留全部路径及内容，证据ig_baseline_suite_v1/AUTHORIZED_DEDUP_DONE.json。

## 最新暂停指令（2026-09-14）
用户已暂停当前baseline目标，仅授权阅读SSU官方代码和评测方式。禁止自动恢复暂停进程或启动新实验；详见SSU_OFFICIAL_EVAL_AUDIT_ZH.md及USER_PAUSED.json。此前全量迁移建议撤回：SSU官方摘要和翻译本来采用明确构造的500条子集。

## 用户恢复与适度对齐（2026-09-14最新）
用户已恢复原baseline目标，并允许保留合理且已积累结果的自有任务设置。继续固定v3 seed42，不因官方差异任意重跑。GSM8K保留8-shot CoT并明确与SSU官方5-shot不同。摘要500条已核实ID/顺序/文本全匹配官方规则（Qwen tokenizer）。翻译源401，抽样来源一致性待核验；现有500条固定跨模型使用。训练和评测原进程已恢复，禁止重复启动。

## 全局延期 MMLU / G-MMLU（2026-09-14 最新用户指令）

所有模型的 MMLU、G-MMLU 均先搁置，等所有模型的其他 benchmark 完成后再考虑，不自动恢复。已完成结果保留；未完成任务不计为完成。当前队列执行六项语言核心任务、IFEval、GSM8K 和单独 PPL 诊断，评测仍固定 seed42。调度状态见 experiments/ig_baseline_suite_v1/DEFER_KNOWLEDGE.json 和 nonknowledge_queue.py。只传模型路径的当前入口为 scripts/reproduce_baseline_eval.sh，它遵守延期标记；延期期间不要绕过该入口调用旧 baseline_eval.sh。此次仅调整调度，不修改冻结任务与评分协议。


## 2026-09-17 当前双任务与正式 GSM8K 协议（覆盖上文过时指引）

- 最新用户授权为独立单语言SW训练/测评与IG baseline统一复评；不启动v3.2 Sequential、LAPE、MoFO、MADLAD、TE或TH。上文v3.2配置作为原轨道档案；实际当前参数分别读取configs/ig_baseline_train_v1.json、ig_baseline_eval_v3.json、sw_fineweb2_aya_v1.json及ig_gsm8k_harness048_v1.json。不得套用旧CL顺序或旧预算。
- GSM8K正式使用lm-evaluation-harness 0.4.8的gsm8k、5-shot、chat template及fewshot_as_multiturn；seed42，temperature0.8/top_p0.8/top_k40/repetition_penalty1.1/do_sample=true，生成上限沿用harness默认。旧自定义8-shot结果只作历史，禁止混入当前比较；上文“保留8-shot”已被用户后续指令覆盖。
- GSM8K专用环境为/root/autodl-tmp/envs/gsm8k_harness048/bin/python，冻结身份见experiments/ig_gsm8k_harness048_v1/protocol_lock.json。不得为保持本文件旧的双环境表而换用其他解释器。
- MMLU/G-MMLU继续全局延期；每任务seed42一次，禁止为匹配历史数字而重跑或选最好一次。评测中断保留attempt记录，仅重跑未完成单元；完成结果须通过DONE身份和产物哈希验证。
- 组合入口为experiments/reproduction/ig_eval.sh与sw_eval.sh，仅传完整HF模型路径；详见该目录README_ZH.md。组合端到端验收尚待全部底层正式任务结束后验证，不能仅凭脚本存在宣称交付完成。IG六项语言/IFEval/PPL与新版GSM8K分属冻结轨道，组合调用不修改各自协议。
- 四GPU调度以实际PID和子进程为准。当前GPU0/1完成IG非知识评测，GPU2/3跑剩余GSM8K；expand_on_release.py等待IG完成，再保留活动子任务自然结束后切换四卡GSM8K队列。新selection_controls_v1五项对照已获用户授权并冻结配置，等待原正式队列完成后启动；不能混入本轮baseline结果。
- 交付证据索引：experiments/reproduction/DELIVERY_EVIDENCE_INDEX_ZH.md。动态进度：experiments/TWO_TASK_PROGRESS_ZH.md。缺项及协议分离原则见reproduction/current_missing_results.json与current_raw_and_relative.csv。PPL/NLL始终单独诊断，EN3/Target3仅阅读、摘要与对应输出方向翻译。


## 2026-09-17 最新用户覆盖：SW数据配方优先
用户取消selection_controls_v1的随机层/中间层消融，改做SW全部FFT的C多任务SFT、B纯Aya SFT、A纯FineWeb2 CPT，优先级C→B→A；D旧9:1混合模型只作历史参考。各新组15.36M非padding输入token、完整SFT样本≤512、seed42、lr5e-5，训练中固定开发集loss和TensorBoard，不据测试分数挑checkpoint。C/B正式配置configs/sw_data_recipes_v1_C_multitask_sft.json及_B_aya_sft.json；独立产物experiments/sw_data_recipes_v1，A准备中。C输入配比Aya40%、阅读20%、摘要20%、英→斯10%、斯→英10%。阅读使用固定第三方TyDiQA镜像，官方GCS403，须披露来源限制；摘要XL-Sum train，同分布监督泛化单列。新训练统一经scripts/train_cl.py路由run_sw_data_recipes.py，不改旧评测代码。GPU0启动C，GPU1旧IG Top20 IFEval/PPL完成后启动B；GPU2/3保留旧IG GSM8K。已停止旧五控制实验和四卡GSM扩展等待器，禁止自行恢复。MMLU/G-MMLU继续延期。


## 2026-09-17 最新暂停：先长度统计，再确认新配方
用户要求512仅作为候选长度，先交付未按长度筛选的原始去重样本P50/P90/P95/P99、512/1024/2048覆盖率、输入与监督token及重复次数，再决定长度/预算/配方。sw_data_recipes_v1的C/B训练和A/B/评测自动队列已SIGSTOP暂停，保留内存及现有产物；不得自动SIGCONT或另启新训练。原IG GSM评测继续。暂停证据experiments/sw_data_recipes_v1/USER_PAUSED_FOR_LENGTH_AUDIT.json。原512数据池保留为旧准备版本，不作本轮长度决策的数据母池；当前长度统计直接读取原始Aya、TyDiQA镜像、XL-Sum、MAFAND文件。此前OPUS-100提案已纠正为MAFAND。


### 2026-09-17 剩余IG评测四卡收尾调度
新SW配方训练仍SIGSTOP暂停，但GPU0/1分别剩约9.3/8.5GB显存，已由backfill_paused_training_gpus.py安排HFT/LoTA的原冻结GSM8K评测；GPU2/3现有Top20/新SSU继续不中断。原调度器PID2011被有意SIGSTOP，仅停止领取新任务，子worker继续；严禁在补位HFT/LoTA完成前手动SIGCONT该调度器，否则可能重复领取。补位监督器PID14835会在两个子任务退出后恢复原调度器，原队列通过DONE跳过它们并继续首尾层。真实状态见experiments/ig_gsm8k_harness048_v1/BACKFILL_STATUS.json与backfill_controller.log。这不授权恢复新训练或改变2048长度方案。


## 2026-09-17 Aya扩容与三个新数据配方
用户要求优先扩大Aya数据池、制定新计划完成A纯CPT/B纯Aya/C多任务SFT。计划experiments/sw_data_recipes_v2/PLAN_ZH.md；已启动本地Aya train候选扩容（expand_aya.py），保留全部人工Aya/Dolly/CoQA，SODA/Wiki各最多50000候选。首选统一2048、15.36M输入token、单样本最多3次；最终来源配比和训练配置须按扩容统计冻结。旧512训练仍不恢复，新v2训练尚未放行；原IG正式评测继续。


## 2026-09-17 用户正式授权：SW v3原方案数据配方
用户已批准执行C→B→A三个独立FFT模型：C为Aya35%/独立英文阅读翻译30%/NLLB双向各10%/XL-Sum摘要15%，B纯Aya，A纯FineWeb2。唯一新配置configs/sw_data_recipes_v3.json，实现scripts/sw_recipes_v3/README_ZH.md，产物experiments/sw_data_recipes_v3和autodl-tmp/sw_data_recipes_v3。不能用TyDiQA/MAFAND或旧v2配方替代。预算每组15.36M非padding输入token；完整SFT、最多3次；依据实际容量选共同512/1024/2048。C阅读用固定Belebele独立英文六源脚本及NLLB3.3B译本，固定200题双语内容审核未完成或错误>5%不得放行，自动检查不冒充人工审核。至少50GiB训练盘空间、三组数据/配置/源哈希及TRAINING_RELEASE.json通过后才启动；目前准备阶段不等于训练完成。旧v1暂停模型和队列不恢复，v2计划被本轮取代；不启动额外稀疏/随机层/多语言实验，原IG评测继续。训练/评测分别经scripts/train_cl.py与scripts/evaluate_cl.py新路由，底层原评测协议不改。D仅历史参考。主观察SW Belebele，C-B/B-A配对文章bootstrap和Holm，EN3/Target3不含PPL；MMLU/G-MMLU继续延期。执行动态状态看新轨道CONTROLLER_STATUS.json，不按旧状态推断进程。


## 2026-09-17 用户改用官方原生SW阅读（最新覆盖）
用户因NLLB译制阅读质量不足，明确要求按开源原生数据完成。v3翻译及控制器已停止，证据保留，不再恢复。独立新协议configs/sw_data_recipes_v4.json、scripts/sw_recipes_v4，实验产物experiments/sw_data_recipes_v4。按用户给出的TyDiQA细则：官方google-research-datasets/tydiqa的secondary_task/train Swahili，固定revision da78f23f9119363459acbaf46bf89426ff26c259；不翻译、不生成答案，首个原序有效字符偏移答案；按文章title及内容组拆分5%开发集。C计划Aya55%/阅读10%/NLLB双向各10%/XL-Sum15%，阅读三遍容量不足则训练前差额转Aya；A/B及15.36M预算、FFT、seed42、评测与C→B→A优先级不变，三组统一2048完整样本。启动前交付NATIVE_READING_ACCEPTANCE_ZH.md/CSV及格式化样本mask，哈希与完整性验收、50GiB空间检查后放行。旧目标文字中禁止TyDiQA已被此最新明确用户要求覆盖。不得把自动偏移/掩码检查称为全量人工内容核验。旧译文和历史训练模型不删除。
