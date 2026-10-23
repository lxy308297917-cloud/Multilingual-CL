# S-All-LR077更新强度对照

状态：真实数值与恢复预检通过，唯一4293步训练正在运行；尚无新任务分数结论。

[中文报告](docs/results/module_lr077_20261023/README_ZH.md) · [冻结协议](docs/results/module_lr077_20261023/PROTOCOL_FROZEN.json) · [冻结判定](docs/results/module_lr077_20261023/DECISION_RULES_FROZEN.json) · [代码](docs/results/module_lr077_20261023/code)

仅将原S-All完整学习率曲线乘0.7714。每步77,070,336名额、相同数据前缀、seed42、BF16权重/FP32动量和4293步不变；预检、训练、评测合计上限3 GPU小时。路径匹配要求S-Attn参考213.379786的±10%；未匹配不追加学习率重训。数学和正式测试未运行。
