# 初始化seed与成本记录：新v2入口

固定官方COPSD核心、LoRA／固定原始Base教师／在线rollout／原beta0分量裁剪损失／AdamW／有效batch32／2048生成预算均保留。新入口仅补齐初始化前seed、成本记录和只打印堆栈的超时观测，不更换算法。

## 四个运行文件

| 文件 | 职责 |
|---|---|
| `run_seeded_single_with_dev.sh` | 同一GPU依次Base开发100题、全新100步训练、最终开发100题 |
| `run_seeded_cost_batch_17b.sh` | 冻结身份预检、原训练参数、独立模型目录与显存采样 |
| `seeded_cost_batch_entry.py` | 创建模型／PEFT前应用实际seed；沿用已验证的批量rollout和正常退出流程 |
| `cost_instrumentation.py` | 原样返回官方loss，读取实际shifted labels并给student／teacher forward计时 |

没有修改旧BN／TE／JA的冻结入口或配置。旧同seed短跑的初始LoRA不同而初始策略相同；该初始化缺口保留为历史边界，不能宣称旧实验逐位可复现。新v2与旧运行分别标记，模型规模、原始Base revision、tokenizer和共同原题池保持一致。

## 验证边界

`check_seeded_cost_cpu.py`在微型CPU Qwen3／PEFT上，先改变进程RNG状态再比较：延迟设置Trainer seed不能固定LoRA初值，提前设置seed可固定LoRA初值；零B初始化仍产生同一Base输出。成本读取前后的原官方loss、梯度与torch RNG完全相同，mask计数正确。该检查不是完整GPU模型的逐位重放证据。见 `seeded_cost_cpu_20261020.PUBLIC.json`。

真实forward事件钩子的接口已经在独立两更新教师信号短跑中验证；新成本入口仅保留mask计数与forward计时，不进行额外整词表分布分析。首个ZH正式运行已核对初始策略等于原始Base，并产生有限loss的真实microstep；checkpoint成本文件仍待实际保存验收。没有把CPU预检当成100步完成。

## 保存与恢复

正常100更新实际3188题次，跨epoch末组20题。每个checkpoint与根目录保存`COST_SEGMENT_PROGRESS.json`，结束保存`COST_SEGMENT_DONE.json`和私有逐调用计数。记录有效回答loss位置、mask位置、教师／学生forward次数、CUDA事件秒数、总处理及有效attention位置。forward计时不包含rollout、KL、反传、optimizer或保存；完整训练时钟及实际生成token另行保留。

续训的成本文件只涵盖该进程segment，父任务ledger单独保留，不假称覆盖迁移前未观测的token。此前旧运行缺少精确有效token／单独teacher计时的部分，仍标为未单独记录，不能用新短跑倒填旧训练成本。

每个step开始与结束重新设置600秒堆栈打印超时，训练结束或异常退出时取消。超时只dump Python堆栈，不杀进程、不改变系统ptrace配置、不自动重试；取消运行队列仍遵守AGENTS.md授权要求。

## 当前启动范围与输出

只手动启动ZH新链路；RU／DE／FR／ES已改用相同新入口并CPU预检，尚未启动。Base与最终checkpoint的开发协议同为`project_dev_single_vllm_v1`，各100题，仅adapter／输出目录不同。遇错整个链路停止，不自动重试；不覆盖已有结果。

新checkpoint使用`models/checkpoint/copsd_vllm_seeded_cost_single_v2/RUN_ID`，开发结果在`logs/copsd_official_10lang/development_vllm/`，训练日志与GPU采样在项目日志目录。TensorBoard沿用Trainer对应run目录。全部任务合计最多四张GPU，以队列分配的UUID为准，不手设局部GPU号覆盖选卡。

ZH已验收checkpoint5：160次loss调用、295,590有效回答位置、teacher／student各160次forward，分别39.20／36.50 CUDA秒。`reconstruct_single_sampling.py`从本机可信checkpoint参数和固定SeedableRandomSampler重建前160个原题ID，学生／教师总处理与attention位置四个实际总量均吻合；原题ID与指纹清单仅保存在本机私有文件。这是重建清单，不冒称逐microstep直接记录。见 `zh_seeded_cost_checkpoint5_20261020.PUBLIC.json`。

Base、SW100与TH100的英语GSM8K1319题0-shot正式保持评测已另行手动排队，使用冻结协议，只作结果报告，不用于挑语言或checkpoint；不占第五张卡。
