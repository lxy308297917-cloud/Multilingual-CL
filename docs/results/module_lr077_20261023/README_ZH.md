# S-All-LR077：更新强度对照

状态：原始参考产物与公共固定代码已核验；真实最长样本与4→8保存恢复预检通过，4293步正式训练正在GPU2运行；配置和判定在出分前冻结。本页不包含新模型的能力结论。

本轮仅将原S-All完整学习率曲线乘0.7714，峰值由5e-5变为3.857e-5；4293步、相同Base、输入1,536,351/监督599,469 Token、seed42、有效batch2、2048长度、每步名额77,070,336及原矩阵配额/评分/20步刷新不变。BF16权重、FP32 Adam动量；embedding/head/Norm/bias冻结。完整日程仍42465步、warmup2124，使用其原前缀。

以S-Attn实际累计BF16更新路径213.379786为参考，固定±10%范围[192.0418074,234.7177646]；净位移和分模块变化另列。未匹配时只称降低学习率对照，不调参重训。

沿用旧512条生成/511条评分，旧模型缓存复用；新模型完整评测。Attention相对新对照沿用旧联合门槛与来源配对10000次bootstrap；宽区间是证据不足，不等于证明劣势。数学和正式测试不执行。

[冻结协议](PROTOCOL_FROZEN.json)、[冻结判定](DECISION_RULES_FROZEN.json)、[代码差异](engine.py.patch)、[参考服务器复核](REFERENCE_REUSE_VERIFIED.json)、[公共代码复核](PUBLIC_REFERENCE_VERIFIED.json)。原优化器与评分代码逐字节保留，engine唯一算术变化为学习率缩放；train/evaluate中的其他变更为独立路径、标识和预算保护。

新增GPU总预算3小时，包含预检0.10、训练2.05和评测0.85小时；同型3090历史S-All训练1.436小时、评测约0.444小时，预估整体约2小时。仅此一组，不自动增加第二学习率、seed、子空间或层选择。

新检查点统一入口`/data0/users/lixiaoyang/models/checkpoint/module_lr077_20261023`，实际存放本账号既有私有存储`/data1/lixiaoyang/module_lr077_20261023`。日志在`/data0/users/lixiaoyang/logs/module_lr077_20261023`。原正式科学检查点保留；4/8步冗余技术载荷清理逐项保存于TECHNICAL_CLEANUP_DONE.json。
