# SW OPD 当前状态

更新时间：2026-10-15 12:38（Asia/Shanghai）。执行依据：`docs/QWEN3_MATH_SINGLE_10LANG_CODEX_PLAN_ZH.md`，方案版本 `efe9b146d7c418ce8696368bbb6891ab36c29573`。

12:10 结果更新：同快照冻结 4B 教师两题×三上下文已全部完成，三组都答对两题，生成 token 总计 8,945，教师采样权重检查不变；但 **六条 thinking 都明显为英语**，仅 final 为 SW，不能据此验收 SW 教师轨迹。SW 强制思考前缀试验中，一题缺 `</think>`、另一题答对但 SW 表达明显重复和生硬，故也未放行。两条 Base 学生 512-token 轨迹上，同快照特权教师全词表 reverse KL 分别为 0.1301、0.0150，参数无梯度；这只证明对齐和分布差异，不证明有用的 SW 训练信号。原始记录分别在 `logs/math_sw_opd_20261015/teacher_probe2/`、`teacher_prefix_probe2/`、`teacher_prefix_score2/`。更强 14B 特权教师作为**独立诊断备选**的两题探针已手动排队，不能与原定同快照 4B A/B/C 混称。

SW 候选 split 已按预先固定 SHA256(seed42) 规则从旧 v8 不相交划分中取 500 train、100 dev_tune、100 dev_check；排除 4 条已知算式矛盾和 24 条英文参考语言检测不确定，见 `logs/math_sw_opd_20261015/SW_SPLIT_CANDIDATE_V1.json`。这是待审候选，正式 Q_full 仍为 0；另保存 30/10/10 道分层源质量审查清单 `SW_SOURCE_REVIEW_SAMPLE_V1.jsonl` 和 `ACCEPTANCE_POLICY_V1.json`，没有虚构人工审核。

源题分层样本前 10 条已完成 Codex 数学预审：8 条未见明显数学错误，2 条需修订或剔除（一道把两两和求总价的 `2G+60=121` 错写成 `3G+60=121`，一道三角函数最优性证明缺推导）。记录在 `SW_SOURCE_REVIEW_FIRST10_CODEX.jsonl`；这是模型预审，不是人工或 SW 母语认证，不能据此放行批量源题。SW 教师轨迹候选生成入口 `build_sw_teacher_trajectories.py` 已实现，显式核验源身份/状态、输出结构、官方 gold、SW 语言筛查与学生完整 2048 token 长度，并记录逐条成本；它尚未生成或验收标签。

又预审源题样本 5 条，其中 3 条英文参考推导有明确错误或自相矛盾：等差数列中间公式写反、几何证明推出与题设相矛盾的长度等式、Fermat 数证明把 `n=1` 的质数特例判为合数。记录在 `SW_SOURCE_REVIEW_NEXT5_CODEX.jsonl`。累计 15 条预审中 5 条需修订或剔除；这是样本中的实际问题数，不能外推为全集错误率，也不能凭其余 10 条模型预审直接验收源题。教师会读取英文参考，必须先处理这些缺陷。

现有特权上下文教师探针中的两条 dev_tune 输出已按新 SFT 规则单独审计：完整学生序列分别为 1,272、1,120 token，均答对且正常终止，但 thinking 为英语，因此两条都自动拒绝，accepted=0。源题本身仍待审，作为另一个未放行状态。审计 `logs/math_sw_opd_20261015/SW_TEACHER4B_DIRECT_CANDIDATE_AUDIT_V2.jsonl`；V1 曾把源题待审误列为硬拒绝，保留并由 V2 纠正。两条仅为已有探针的回溯候选，不计作正式批量标签合成。

## 已完成与仍在运行

- 旧十语言 14B 标签队列 `mathlabels-v8-reviewed4` 尚未取得第二张 GPU、没有启动模型；已按新方案手动停止并释放本账号保留的 GPU。四个 8192-token Base 长度复测队列也已停止，均未运行。旧数据、日志和已完成的英语 Base 三 seed 与 IFEval 结果完整保留，属于旧协议基线。
- 复用旧 v8 原题对齐、benchmark 排除和划分作为 SW 候选池。`logs/math_sw_opd_20261015/SW_STATIC_TRIAGE_V1.jsonl` 保留 1000 train、200 dev_tune、200 旧 dev_select 题的 SW/EN/参考/gold 与内容哈希；24 条英文参考的语言检测不确定，4 条出现已知独立算式矛盾。这些只是自动初筛，源推理和 SW 题意仍待验收，当前正式 Q_full=0。
- 两道旧 dev_tune 题的英文推理及 SW 题意已作 Codex 预核，作为 **教师接口探针**，不代表正式数据通过人工验收。新增冻结 Qwen3-4B 教师三上下文脚本 `sw_opd_teacher_probe.py`，对 SW 题、SW+英文题、SW+英文题+原参考使用相同逐题 seed 和官方 PolyMath 评分，记录原始输出、语言、token、耗时及参数无梯度/采样权重不变检查。队列 `swopd-teacher-probe2` 已在 GPU 0 完成，结果目录 `logs/math_sw_opd_20261015/teacher_probe2/result/`；质量结论见文首。

## 阻塞与接续

- OPD 4 卡 FSDP 工程烟测入口 `train_sw_opd_fsdp_smoke.py` 与 `run_sw_opd_fsdp_smoke.sh` 已准备：4B 学生在线采样、独立冻结的同快照 CPU 教师全词表 reverse KL、5 step 短 rollout、0/5 step 冒烟生成 eval 与 TensorBoard，逐 rank 记录 token/耗时/显存。训练题 4 条为 Codex 数学预审通过的待审源题，dev 2 条为原教师接口探针题；身份与拆分预检通过。此入口使用 BF16 AdamW 状态、短 rollout 和未正式验收源题，**只用于工程烟测，不是正式 100 step FP32 AdamW 先导或 A/B/C 结果**。随机初始化微型 Qwen3 的两进程 CPU FSDP 生成已验证（不同 rank 生成长度亦可），但 4B 真机烟测尚未运行；不能把算法检查或旧 SFT 烟测当 OPD 结果。
- 新协议正式同教师 SW 轨迹合成尚未启动，回溯探针候选 2、accepted 0；旧 14B 翻译标签不可作为 A 组主数据。源题审查、SW 语言审核和 2048 完整序列检查仍是放行条件。
- 14B 外部教师两题诊断 `swopd-teacher14b-probe2` 仍在手动启动的一次性双卡队列等待第二张经核实空闲的卡；其结果仅用于判断另开强教师轨道是否值得做，不能替代同快照 4B 主对照。
- 当前外部教师队列已核实 systemd `ActiveState=active`、`MainPID=1746364`，保留 GPU 0 等第二张卡。暂不同时启动需要 4 卡的 OPD 烟测队列，避免两个队列各自占住部分卡后互相等待；待诊断完成或明确调整优先级再手动启动工程烟测。
- SW 来源审核后先做教师前缀信号与长度/显存检查，再做 5–20 step OPD smoke；通过后从原始 Base 另起 C_full，开启 0/20/50/100 的开发集 eval 和 TensorBoard。A/B/C 仅在共同 accepted 题池准备好后运行。

## 资源与成本

已完成的 4B 三上下文教师生成 8,945 token；两条短学生轨迹共 1,024 rollout token、1,024 个教师评分位置及 1,024 个学生评分位置，全词表 KL 计算不计作 OPD 更新。正式 OPD 有效 loss token 和 optimizer step 仍为 0。14B 诊断队列按本账号最多四卡与 UUID 选卡规则等待，不占用其他用户的任务。
