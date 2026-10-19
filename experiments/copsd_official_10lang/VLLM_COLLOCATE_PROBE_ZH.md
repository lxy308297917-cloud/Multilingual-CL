# 官方 vLLM 在线训练接入检查

2026-10-19。官方基准版本固定为 `4f4410730e9b6775a30d20f4b6e141cba2e87590`。

## 已取得的证据

独立生成测速中，同卡 batch2 的 Transformers 为32.46 token/s，vLLM为201.85 token/s。约6.22倍仅适用于该生成测试，未覆盖教师前向、蒸馏反传、在线权重同步及两套模型同卡显存。完整结果见 `rollout_benchmark_results_20261019.json`。

接入前发现官方 `_move_model_to_vllm` 对学生调用 PEFT merge/unmerge，而冻结教师通过禁用同一模型的 adapter 获得。用真实 Base BF16 权重和 TH checkpoint70 的 FP32 LoRA，在 CPU 实测：

| 实际矩阵 | 元素数 | 一次合并拆分后变化元素 | 最大绝对变化 |
|---|---:|---:|---:|
| 第0层 q_proj | 4,194,304 | 8,909 | 0.0001220703125 |
| 第0层 down_proj | 12,582,912 | 17,870 | 0.00006103515625 |

这是两块权重的舍入证据，尚未测量整个模型的输出分布影响；不能据此判断方法效果。当前 SW／TH 正式训练使用 HF 生成，不执行 vLLM 合并同步，因此不受此路径影响。官方参考源码和当前运行中的冻结代码均保留。

## 必要适配及验证范围

`lora_sync_weights.py` 对每块 base weight 创建临时副本，按照 PEFT 相同的 FP32 delta 加法导出 BF16 合并权重，只更新 vLLM 推理副本。原始 base 不执行加法或减法。支持范围限定为本轮普通线性 LoRA，单个 adapter，无 DoRA、量化或 modules_to_save；不支持的配置直接报错。

- 上述两块真实矩阵：导出张量与官方 PEFT merge 的张量逐位相同，源 base 逐位不变。证据为 `lora_teacher_roundtrip_cpu_20261019.json`。
- 小型 PEFT 模型额外验证导出名称、所有导出张量及未被 LoRA 作用的参数，与官方合并结果一致；导出不修改源参数。此项只证明参数路由，不代替真实 GPU 同步检查。
- 额外在训练开始显式同步当前学生。发布回调只在 step end 同步，新初始化零 B 可用原 Base，但未来加载已有 adapter／顺序阶段时必须先同步，再生成。

以上为本项目的工程适配，未修改官方损失、英文特权教师上下文、思考前缀、AdamW 或目标语言数据。不能把修正后的路径直接称为未经修改的官方实现。

## 长梯度累积中的 sleep 生命周期适配

本轮为了24GiB同卡运行启用 sleep，但发布脚本默认未启用这一可选分支。已核实 vLLM0.11 的实际源码：level2 会丢弃权重和KV；level1把权重备份到CPU、丢弃KV。官方可选路径每个micro step后level2休眠，下一micro step仅唤醒KV，权重只在优化器更新后的同步时重载，因此不能直接用于本轮32次梯度累积。

待运行probe保留初始化时的level2，并在第一次生成前完整加载当前学生权重；之后每次生成后使用level1，下一次只唤醒仍在休眠的tag。普通micro step唤醒权重和KV；更新边界若同步已唤醒权重，则只唤醒KV，避免重复映射。权重备份不执行CPU模型前向，没有增加独立CPU教师。

CPU已调用安装版本的真实 `ExecutorBase.sleep/wake_up` 方法验证以上状态转换，worker RPC为模拟，不能证明GPU内存复制成功。记录见 `vllm_sleep_lifecycle_cpu_20261019.PUBLIC.json`。GPU probe会在第2和第34个micro step核对唤醒后的推理权重与当前同步指纹完全相同，再接受两次更新结果。此兼容适配仅用于排队中的probe，当前正式SW／TH及固定官方参考均未改变。

## 已准备的下一次 GPU 检查

入口 `run_vllm_colocate_probe.sh NEW_RUN_ID`，必须由单卡手动队列提供 UUID；公开设置在 `vllm_colocate_probe_20261019.PUBLIC.json`；实际启动读取的 `vllm_colocate_probe_20261019.CONFIG.json` 含精确数据／源码指纹，仅保留本机。自动审批拒绝发布这些内部指纹，公开证据仅含统计与验证结论。源码／模型／SW数据哈希、入口解析和 shell 语法已通过 CPU 检查。**尚未启动 GPU probe，尚未验证该训练路径能在24GiB完成。** 单卡队列 `copsd-sw-17b-vllm-colocate-probe-v1` 已手动启动监控，目前等待资源，GPU probe 尚未开始。不取消正在运行的 SW 或 TH。新增probe与三个待开发集评测均使用默认账号四卡上限；当前SW加四卡TH的既有临时五卡授权不扩展给新任务。

- 同一1.7B模型、SW官方适配数据，从 Base 独立开始。
- micro1、累积32、有效 batch32，保留2048生成上限和完整官方裁剪蒸馏损失，运行2次真实更新，64个micro step。
- colocate、TP1、显存比例0.4，启用官方已有 sleep mode；这是显存适配，需要实测。
- 校验每次生成对应当前已同步 adapter；对比三次同步的推理权重指纹，确认更新后确实变化；逐次核对全部冻结参数不变。
- 开启 TensorBoard，每步保存 checkpoint；记录完整长度吞吐、KD反传、同步及额外诊断哈希的成本。probe 的学习率调度总长度为2步，与正式100步的调度区分，不把probe当作效果对照。
- 成功、失败均保存独立记录，不自动重试。完成2次更新仍不等于 checkpoint 恢复已验证，也不等于数学效果有效。

通过这一步后才能决定后续语言是否采用 vLLM 正式训练。Base／single／顺序模型的评测参数仍需保持一致，已有 HF 开发集分数不与改动后端的分数直接混比。
