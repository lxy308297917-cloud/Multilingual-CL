# COPSD 复现来源与适配边界

本项目以官方 `cisnlp/COPSD@4f4410730e9b6775a30d20f4b6e141cba2e87590` 为起点。4B 实验属于**官方代码基础上的项目适配**；1.7B AfriMGSM 发布协议结果另行记录。

## 源码证据

- 原始目录为不含 `.git` 的源码导出。重新取得指定 commit 的官方归档后，19 个源码、配置和文档文件逐字节一致；只缺两个 `.DS_Store` 系统元数据文件。
- [现有完整补丁](patches/formal_ddp_complete_from_4f441073.patch) 在临时副本以零 fuzz 应用后，精确重建当前 `multilingual_opsd_train.py` 和 `multilingual_opsd_trainer.py`。没有修改运行中的源码。
- 补丁涉及缓存路径、dtype 表达、关闭训练 cache、DDP 设备映射、checkpoint 接续和 DDP 教师 unwrap。核心损失函数 AST 的既有一致性证据仍与当前源码哈希对应；这不等于整个运行过程或教师的数学监督必然正确。
- 七个三语言研究训练配置的 42 项项目冻结依赖、2 项安装依赖均与锁定哈希一致。两项 TRL 文件也与已安装 `trl-0.26.0` 的 distribution RECORD 一致。此前 main 已核实 40 项，缺少的 RU 数据准备和 TE 显存验证报告随本次补齐。
- [逐文件哈希、环境版本与核验范围](REPRODUCTION_SOURCE_AUDIT_20261026.PUBLIC.json)。该核验覆盖三语言研究的七段训练，原2700题池十语言的最终交付仍继续。

## 发布脚本与实际设置

对照的是指定 commit 的 `multilingual_scripts/run_all_opsd_4b_3000.sh`。

| 项目 | 官方发布脚本 | 本次4B研究 |
| --- | --- | --- |
| 训练语言 | BN/SW/TE/TH/ZH/ES/RU/JA | 上述八语，加 DE/FR |
| 更新预算 | 100 | 每段20；scheduler horizon仍100 |
| 有效 batch | 4 ranks × micro4 ×累积2 =32 | 1训练rank × micro1 ×累积32 =32 |
| vLLM | colocate | server：1卡训练／教师，1卡在线rollout |
| LoRA | r64、alpha128 | 保持；实际 dropout0.05 |
| 学习率／梯度裁剪 | 5e-6／0.1 | 保持 |
| 训练生成／上下文上限 | 2048／20000 | 保持 |
| 采样温度／top-p／top-k | 1.1／0.95／20 | 保持 |
| beta／词表分量裁剪 | 0／0.05 | 保持 |
| 正式共同评测 | 发布评测另列 | MGSM-Rev2、PolyMath、英语GSM8K；n1/8192/native16 |

当前两卡是两种角色，**不等于两卡 DDP，也不要求四卡才能运行一个 OPD 任务**。原始 Base 权重冻结，教师在训练卡禁用 adapter 取得分布；学生在线轨迹不需要预先制作目标语 SFT 标签。顺序阶段接续同一 adapter，重置 optimizer/scheduler；完整评测记录实际 token 与成本。

训练每2步记录 TensorBoard、每5步保存；数学 eval 由独立开发／正式评测流程执行并导出 TensorBoard。没有把它写成“显式开启了 Trainer 内置 eval_strategy”。完整矩阵和6项对照已完成，[结果与成本](SELECTED4B_COMPLETE_SEQUENTIAL_RESULTS_20261026.PUBLIC.json)保留全部原分、语言与终止统计。

## 环境与入口

实际训练和正式生成使用 Python3.10.20、Torch2.8.0、Transformers4.57.1、TRL0.26.0、vLLM0.11.0、Accelerate1.11.0、PEFT0.17.1、math-verify0.8.0、TensorBoard2.21.0；细目见源码核验报告和已有环境冻结文件。GlotLID 分析使用独立既有环境和固定模型哈希。

- 独立单语适配入口：[run_single4b_compact.sh](run_single4b_compact.sh)、[可移植TE配置模板](reproduction_templates/te4b/README_ZH.md)。
- 顺序训练入口：[run_sequential4b_stage.sh](run_sequential4b_stage.sh)，读取已锁定的阶段配置。
- 正式评测：[run_formal4b_case.sh](run_formal4b_case.sh)，保持已有 freeze、模型与采样协议。
- 汇总：[summarize_selected4b_final.py](summarize_selected4b_final.py)，读取已验收的源证据，输出到新的文件路径。
- [JSONL兼容层](jsonl_physical_line_compat.py)仅修复CPU验收对合法 Unicode 行分隔符的误切，原始数据、数学评分及冻结源码不变。[适配证据](JSONL_PHYSICAL_LINES_COMPAT_20261026.PUBLIC.json)。

复现时通过 `COPSD_CACHE_ROOT` 配置本地缓存目录。配置、模型、数据、checkpoint 的实际路径由本机渲染；版本、数据哈希、阶段预算与协议应与对应研究记录一致。官方复现、工程适配、项目扩展分别解释，不能将这里的20更新/n1结果称为官方100步/pass@12严格复现。
