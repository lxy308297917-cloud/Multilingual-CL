# SW OPD 当前状态

更新时间：2026-10-15 11:50（Asia/Shanghai）。执行依据：`docs/QWEN3_MATH_SINGLE_10LANG_CODEX_PLAN_ZH.md`，远端 main `efe9b146d7c418ce8696368bbb6891ab36c29573`。

## 已完成与仍在运行

- 旧十语言 14B 标签队列 `mathlabels-v8-reviewed4` 尚未取得第二张 GPU、没有启动模型；已按新方案手动停止并释放本账号保留的 GPU。四个 8192-token Base 长度复测队列也已停止，均未运行。旧数据、日志和已完成的英语 Base 三 seed 与 IFEval 结果完整保留，属于旧协议基线。
- 复用旧 v8 原题对齐、benchmark 排除和划分作为 SW 候选池。`logs/math_sw_opd_20261015/SW_STATIC_TRIAGE_V1.jsonl` 保留 1000 train、200 dev_tune、200 旧 dev_select 题的 SW/EN/参考/gold 与内容哈希；24 条英文参考的语言检测不确定，4 条出现已知独立算式矛盾。这些只是自动初筛，源推理和 SW 题意仍待验收，当前正式 Q_full=0。
- 两道旧 dev_tune 题的英文推理及 SW 题意已作 Codex 预核，作为 **教师接口探针**，不代表正式数据通过人工验收。新增冻结 Qwen3-4B 教师三上下文脚本 `sw_opd_teacher_probe.py`，对 SW 题、SW+英文题、SW+英文题+原参考使用相同逐题 seed 和官方 PolyMath 评分，记录原始输出、语言、token、耗时及参数无梯度/采样权重不变检查。队列 `swopd-teacher-probe2` 手动启动，等待一张核实空闲 GPU；结果目录 `logs/math_sw_opd_20261015/teacher_probe2/result/`。当前没有教师质量结果。

## 阻塞与接续

- OPD FFT 的独立冻结教师全词表 reverse KL 训练入口尚未实现和烟测；100 step C_full 先导、eval 与 TensorBoard 尚未启动，不能把旧 SFT 烟测当 OPD 结果。
- 新协议同教师 SW 轨迹候选和 accepted 均为 0；旧 14B 翻译标签不可作为 A 组主数据。需先确定 `acceptance_policy`、源题审查、SW 语言审核和 2048 完整序列检查。
- SW 来源审核后先做教师前缀信号与长度/显存检查，再做 5–20 step OPD smoke；通过后从原始 Base 另起 C_full，开启 0/20/50/100 的开发集 eval 和 TensorBoard。A/B/C 仅在共同 accepted 题池准备好后运行。

## 资源与成本

当前教师探针只排队，GPU 生成 token、教师评分 token、OPD 有效 loss token 均为 0。所有 GPU 在启动前均显示已有计算进程，队列按本账号最多四卡与 UUID 选卡规则等待，不占用其他用户的任务。
