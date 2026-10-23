# TE4B checkpoint5 恢复与logits存储修复

原训练完成7更新后，在第8更新的第229个microstep发生OOM；checkpoint5完整保存。原目录、日志、失败队列和诊断失败记录保留。BaseTE200开发评测已完成42/200，不能单凭Base结果判断OPD效果。

## 存储修复与验证

官方compute_loss先从完整输出中切出响应logits，再删除完整输出；切片仍可保留整段题干与响应的底层存储。`compact_response_logits.py`严格只把两处响应logits赋值改为`clone()`，使删除完整输出后可以释放未用存储。它不改蒸馏方向、dtype、causal shift、mask、教师上下文、词表、分量裁剪或2048训练预算；官方源文件未修改，原始参考入口保留。

CPU微型Qwen3的FP32/BF16、完整/部分mask四个用例，损失与全部LoRA梯度逐位相同。CUDA短16位置对照控制了LoRA dropout随机流和kernel确定性，损失相同、最大LoRA梯度差0；真实失败轨迹2048位置则使用普通kernel策略完成前反向，保留checkpoint5的optimizer状态，峰值allocated显存约17.95GiB。该实样本验证使用可用checkpoint5，不是假称恢复丢失的step7权重；这是独立模型前反向预检，不能当作完整Accelerate Trainer的峰值或逐位续训保证，也不保证未来所有题目长度。

首个GPU诊断未固定dropout随机流，第二个固定随机流但遇BF16非确定性梯度差异，第三个控制两者后通过；所有记录与成本保留。短对照只用于等价性，正式训练长度始终2048。

## 恢复语义

使用`te4b_resume5_entry.py`、`te4b_resume5_job.py`与`run_te4b_resume5.sh`，由Transformers恢复checkpoint5的LoRA、optimizer、scheduler和RNG，不从Base重训。实际恢复已核验：模型对应旧step5、优化器及调度器在5，首次后续32题的学生/教师prompt tensor哈希与原序列跳过160次曝光后对应位置一致。vLLM新进程采样流不宣称逐位复现。

保留初始5更新，再执行15更新至global20；有效batch32、100步调度、原始Base教师和数据池不变。`release_te4b_resume5_checkpoint20.py`只在完整完成后放行新adapter；`evaluate_te4b_resume5_followup200.py`与`collect_te4b_resume5_followup200.py`复用已验收BaseTE/EN，运行相同native16/n1/8192/seed42开发评测。`te4b_resume5_series.py`只观察已运行的恢复队列并启动一次结束评测，不重启训练、不做自动故障重试。

## 成本与配置边界

最终逻辑训练曝光为160+480=640；原失败尝试预生成256轨迹，加上恢复后480轨迹，实际生成最多按736条计费，而不是只报640。失败训练、已丢弃更新及三次GPU诊断分别计入实际成本；BaseTE计一次，既有Base英语不重复生成或计费。成功loss与失败尝试、实际计算与逻辑预算分开记录。

公开科学设置及验证范围见`TE4B_RECOVERY_PROTOCOL_20261024.PUBLIC.json`。完整冻结配置含本机路径与状态绑定，保留本地；公开源码是通用入口，不能将科学协议摘要直接当作其运行配置。已有去机器信息模板仍是原始参考版本，不应与恢复身份混同。恢复入口依赖原实验的有效checkpoint5及匹配数据/资源锁，并不适用于任意失败实验。

目前未产生训练模型的完整TE/EN配对效果，不将工程恢复当作方法有效，也不据此启动三语言顺序实验。
