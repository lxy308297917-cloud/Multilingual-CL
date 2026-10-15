# SW OPD 当前状态

更新时间：2026-10-15 14:06（Asia/Shanghai）。执行依据：`docs/QWEN3_MATH_SINGLE_10LANG_CODEX_PLAN_ZH.md`，方案版本 `efe9b146d7c418ce8696368bbb6891ab36c29573`。

SW 源题 30 train、10 dev_tune、10 dev_check 分层样本的 Codex 模型预审现已完成，逐题记录在 `logs/math_sw_opd_20261015/SW_SOURCE_REVIEW_*_CODEX.jsonl`，汇总及文件哈希在 `SW_SOURCE_REVIEW_50_CODEX_SUMMARY.json`。50 条中 14 条有明确问题或待修订项：train 10/30、dev 4/20；其中 10 条涉及英文参考推导或证明，5 条涉及 SW 条件或术语，1 条两类重叠。典型例子包括参考把两两和的 `2G+60=121` 写为 `3G+60=121`、把 `n=1` 的质数特例判为合数、SW 将“倍数”写成“因子”。这是抽样发现，不是全池错误率；其余 36 条也只是模型预审未见明显问题。**正式源题批次验收仍未完成，Q_full=0**，14 条争议题在修订或剔除前不得混入正式问题池。此项是源题审核，不能与教师 SFT 轨迹的 0 条 accepted 混为一谈。

为避免只在 4 道题上反复做工程烟测，另冻结 `SW_OPD_PILOT_MODEL_REVIEWED_36_V1.json`：上述 36 条按原 split 为 20 train、8 dev_tune、8 dev_check；对应训练/dev ID 文件、50 条合并审核与 14 条修订/剔除队列都保存在同一日志目录，哈希已记录。它仅可用于标明局限的**探索性 OPD 先导**，并未把 Codex 预审提升为正式源题或母语验收。`run_sw_opd_dual_fsdp_pilot20.sh` 预备 20 step、512-token rollout、2048-token dev_tune 生成 eval 与 TensorBoard，训练与评测题目在启动前逐条交叉核对数学、SW 题意预审和候选 split；保留 8 条 dev_check 不参与配置选择。入口已通过语法及 CPU 身份预检，真实 GPU 运行尚未开始。该先导的 BF16 AdamW、batch 4 和 512-token rollout 与正式 100-step 协议不同，不能充作正式结果。

12:10 结果更新：同快照冻结 4B 教师两题×三上下文已全部完成，三组都答对两题，生成 token 总计 8,945，教师采样权重检查不变；但 **六条 thinking 都明显为英语**，仅 final 为 SW，不能据此验收 SW 教师轨迹。SW 强制思考前缀试验中，一题缺 `</think>`、另一题答对但 SW 表达明显重复和生硬，故也未放行。两条 Base 学生 512-token 轨迹上，同快照特权教师全词表 reverse KL 分别为 0.1301、0.0150，参数无梯度；这只证明对齐和分布差异，不证明有用的 SW 训练信号。原始记录分别在 `logs/math_sw_opd_20261015/teacher_probe2/`、`teacher_prefix_probe2/`、`teacher_prefix_score2/`。更强 14B 特权教师作为**独立诊断备选**的两题探针已手动排队，不能与原定同快照 4B A/B/C 混称。

SW 候选 split 已按预先固定 SHA256(seed42) 规则从旧 v8 不相交划分中取 500 train、100 dev_tune、100 dev_check；排除 4 条已知算式矛盾和 24 条英文参考语言检测不确定，见 `logs/math_sw_opd_20261015/SW_SPLIT_CANDIDATE_V1.json`。这是待审候选，正式 Q_full 仍为 0；另保存 30/10/10 道分层源质量审查清单 `SW_SOURCE_REVIEW_SAMPLE_V1.jsonl` 和 `ACCEPTANCE_POLICY_V1.json`，没有虚构人工审核。

源题分层样本前 10 条已完成 Codex 数学预审：8 条未见明显数学错误，2 条需修订或剔除（一道把两两和求总价的 `2G+60=121` 错写成 `3G+60=121`，一道三角函数最优性证明缺推导）。记录在 `SW_SOURCE_REVIEW_FIRST10_CODEX.jsonl`；这是模型预审，不是人工或 SW 母语认证，不能据此放行批量源题。SW 教师轨迹候选生成入口 `build_sw_teacher_trajectories.py` 已实现，显式核验源身份/状态、输出结构、官方 gold、SW 语言筛查与学生完整 2048 token 长度，并记录逐条成本；它尚未生成或验收标签。

又预审源题样本 5 条，其中 3 条英文参考推导有明确错误或自相矛盾：等差数列中间公式写反、几何证明推出与题设相矛盾的长度等式、Fermat 数证明把 `n=1` 的质数特例判为合数。记录在 `SW_SOURCE_REVIEW_NEXT5_CODEX.jsonl`。累计 15 条预审中 5 条需修订或剔除；这是样本中的实际问题数，不能外推为全集错误率，也不能凭其余 10 条模型预审直接验收源题。教师会读取英文参考，必须先处理这些缺陷。

现有特权上下文教师探针中的两条 dev_tune 输出已按新 SFT 规则单独审计：完整学生序列分别为 1,272、1,120 token，均答对且正常终止，但 thinking 为英语，因此两条都自动拒绝，accepted=0。源题本身仍待审，作为另一个未放行状态。审计 `logs/math_sw_opd_20261015/SW_TEACHER4B_DIRECT_CANDIDATE_AUDIT_V2.jsonl`；V1 曾把源题待审误列为硬拒绝，保留并由 V2 纠正。两条仅为已有探针的回溯候选，不计作正式批量标签合成。

14B 外部特权教师两题诊断已在 GPU 0、7 正常完成：2/2 答对，2/2 thinking 为英语，2/2 final 候选为 SW；生成 2,577 token、提示 1,920 token、生成计时 189.1 秒，见 `teacher14b_probe2/result/`。仅增加模型规模未解决这两题的 SW 思考语言问题。4B 关闭 thinking、直接写最终步骤解答的独立诊断也完成：2/2 答对、终止正常，正文语言检测 2/2 判 SW；但原文多处机械重复和明显不自然词句（例如 `Mafanikao`、反复 `Kwa kuchangia`），不满足人工可读的 SW 轨迹要求，不能按自动语言检测放行。该诊断生成 1,611 token、提示 1,928 token、生成计时 59.3 秒，见 `teacher4b_no_think_probe2/result/`；它不改变 thinking=True 的主 OPD 协议。

为排查教师指令语言，额外在一条已完成 Codex 源题预审的易题 dev_tune 上使用全 SW system/user 指令，仍提供原英文题与参考、同一冻结 4B 教师；该诊断在 CPU 上生成 512 token、用时 190.4 秒。输出从开头持续以英语思考，未出现完整 final、达到生成上限；不计数学正确率或标签验收，见 `teacher4b_sw_system_cpu_dev1/`。这是单题模板诊断，说明简单改用 SW 指令并未立即解决英语思考，不能据此证明所有模板无效，也不能将未完成答案当作答错。提示哈希、原文及成本已保留。

同题同 SW 指令的外部 Qwen3-14B CPU 诊断也结束：512 token 原文中的思考仍明显为英语，虽在正文算出 36，但尚未输出正常 final 与 boxed 答案，故数学完整正确率和标签验收均不能按成功计；生成 546.3 秒、提示 575 token，见 `teacher14b_sw_system_cpu_dev1/`。这进一步排除“只把指令翻成 SW 或只增大模型”作为现成修复，不对未测试的模板作普遍结论。该外部教师不能与主同快照 4B OPD/SFT 对照混用。

同一冻结 4B 教师和两道 dev 题的 ZH 接口诊断也完成：2/2 数学正确，thinking 与 final 原文均为中文，生成 3,015 token、提示 1,896 token、生成计时 115.5 秒，见 `teacher4b_zh_probe2/result/`。原自动语言检测器把含数学标记的中文 final 错判越南语；移除公式后，两条 final 的汉字占汉字/拉丁/假名字符比例分别 0.950、0.947，thinking 为 0.846、0.981，见 `ZH_TEACHER4B_LANGUAGE_AUDIT_V1.json`。这说明当前 SW 问题更像目标语言生成能力差异，不能据此断言所有语言教师接口失效。ZH 完整学生序列分别 2,067、1,154 token；前者超出 2048 硬限制，后者仍待源题与语言独立复核，正式 accepted=0，见 `ZH_TEACHER4B_LABEL_SCREEN_V1.json`。

**实际 4B OPD 单卡工程烟测已完成**：`opd_1gpu_sgd_smoke_v1/` 有 5 个新学生 rollout/权重更新，各 64 个生成及全词表教师评分位置，总有效 loss token 320；冻结教师抽样权重不变，每步非零梯度且 BF16 学生 306—311 个参数张量实际变化，GPU 峰值 16,991.8 MiB。0/5 步开发题短生成、逐步 KL、TensorBoard 均保存；两次 128-token 短评测均无完整 final（0/2），**不能当正式数学效果**。教师 CPU 评分累计 134.4 秒，学生 rollout 14.2 秒。该入口使用 FP32 CPU 主权重的纯 SGD、短 rollout、待审源题，运行代码对应 GitHub 提交 `822c29519c89d07a073c84353ae2ec63004d4cc4`；运行中曾出现无 attention mask 警告，后续代码已补显式 mask，旧结果不改写。它不是正式 100-step AdamW OPD；4 卡 FSDP 工程烟测及正式 eval/checkpoint 尚未运行。

## 已完成与仍在运行

- 旧十语言 14B 标签队列 `mathlabels-v8-reviewed4` 尚未取得第二张 GPU、没有启动模型；已按新方案手动停止并释放本账号保留的 GPU。四个 8192-token Base 长度复测队列也已停止，均未运行。旧数据、日志和已完成的英语 Base 三 seed 与 IFEval 结果完整保留，属于旧协议基线。
- 复用旧 v8 原题对齐、benchmark 排除和划分作为 SW 候选池。`logs/math_sw_opd_20261015/SW_STATIC_TRIAGE_V1.jsonl` 保留 1000 train、200 dev_tune、200 旧 dev_select 题的 SW/EN/参考/gold 与内容哈希；24 条英文参考的语言检测不确定，4 条出现已知独立算式矛盾。这些只是自动初筛，源推理和 SW 题意仍待验收，当前正式 Q_full=0。
- 两道旧 dev_tune 题的英文推理及 SW 题意已作 Codex 预核，作为 **教师接口探针**，不代表正式数据通过人工验收。新增冻结 Qwen3-4B 教师三上下文脚本 `sw_opd_teacher_probe.py`，对 SW 题、SW+英文题、SW+英文题+原参考使用相同逐题 seed 和官方 PolyMath 评分，记录原始输出、语言、token、耗时及参数无梯度/采样权重不变检查。队列 `swopd-teacher-probe2` 已在 GPU 0 完成，结果目录 `logs/math_sw_opd_20261015/teacher_probe2/result/`；质量结论见文首。

## 阻塞与接续

- OPD 4 卡 FSDP 工程烟测入口 `train_sw_opd_fsdp_smoke.py` 与 `run_sw_opd_fsdp_smoke.sh` 已准备：4B 学生在线采样、独立冻结的同快照 CPU 教师全词表 reverse KL、5 step 短 rollout、0/5 step 冒烟生成 eval 与 TensorBoard，逐 rank 记录 token/耗时/显存。训练题 4 条为 Codex 数学预审通过的待审源题，dev 2 条为原教师接口探针题；身份与拆分预检通过。此入口使用 BF16 AdamW 状态、短 rollout 和未正式验收源题，**只用于工程烟测，不是正式 100 step FP32 AdamW 先导或 A/B/C 结果**。随机初始化微型 Qwen3 的两进程 CPU FSDP 完整工程链路已通过：不同 rank 生成长度、冻结教师评分、全词表 KL、反向传播、梯度裁剪和 AdamW 更新；原始脚本及输出摘要保存在 `logs/math_sw_opd_20261015/SW_OPD_FSDP_CPU_PROBE_SOURCE.py` 与 `SW_OPD_FSDP_CPU_PROBE_DONE.json`。4B 单卡短程 OPD 已运行；四卡队列 `swopd-fsdp-smoke-v1` 于 13:15 手动启动，目前仅保留 GPU 0、等待另外三张经核实空闲 GPU，真实 FSDP 训练尚未开始，正式优化器仍未验证，不能把短程结果当正式 OPD 效果。
- 新协议正式同教师 SW 轨迹合成尚未启动，回溯探针候选 2、accepted 0；旧 14B 翻译标签不可作为 A 组主数据。源题审查、SW 语言审核和 2048 完整序列检查仍是放行条件。
- 14B 外部教师诊断与 4B 关闭 thinking 诊断都已完成，退出码 0；前者只能作为独立强教师轨道证据，后者只能作为生成模式诊断，均不能替代同快照 4B、thinking=True 的正式 A/B/C 对照。
- 4 卡 OPD 工程烟测队列已启动但真实训练尚未开始；当前只有 GPU 0 核实空闲且由本账号队列保留，其他卡有用户计算进程。之前 GPU 7 在 14B 任务运行期间出现另一用户的计算进程，未干预对方任务；两题探针仍正常完成。本账号后续新任务必须重新逐卡检查和遵守四卡总上限，不把旧空闲快照当作当前资源状态。
- 单卡烟测中 CPU 教师评分 320 token 用时 134.43 秒，短序列均值约 0.420 秒/token。若机械外推到 100 step×batch 32，四路并行、每题 512/1024/2048 token，**仅教师评分**约 1.99/3.98/7.97 天；这是 5 条 64-token 样本的粗略容量估算，不是正式运行时长，实际长上下文与 CPU 争用可能显著改变吞吐。机器可读计算在 `logs/math_sw_opd_20261015/SW_OPD_CPU_TEACHER_COST_EXTRAPOLATION_V1.json`。因此另准备独立 GPU 分片教师的 `train_sw_opd_dual_fsdp_smoke.py` 与启动脚本，不修改已排队的 CPU 教师烟测。两进程微型 Qwen3 的 CPU 双 FSDP 教师/学生前向、不同回答长度、KL、反向和 AdamW 更新通过；日志为 `SW_OPD_DUAL_FSDP_CPU_PROBE.log`。真实 4B 显存与吞吐**尚未验证**，新入口未启动、不能称正式训练。
- SW 来源审核后先做教师前缀信号与长度/显存检查，再做 5–20 step OPD smoke；通过后从原始 Base 另起 C_full，开启 0/20/50/100 的开发集 eval 和 TensorBoard。A/B/C 仅在共同 accepted 题池准备好后运行。

## 资源与成本

已完成的 4B 三上下文教师生成 8,945 token，14B 外部诊断 2,577，4B 关闭 thinking 诊断 1,611，4B ZH 诊断 3,015；对应提示 token 分别保留在各自 `DONE.json`，不混作同一协议。先前两条短学生轨迹共 1,024 rollout/教师评分位置，不计更新；单卡短程 OPD 另有 320 个有效 loss token、5 个工程更新及 TensorBoard。**正式协议**有效 loss token 和 optimizer step 仍为 0。
