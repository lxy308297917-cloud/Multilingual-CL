# S-All-LR077更新强度对照

状态：完整4293步、冻结端点评测和服务器原始产物核验已完成。

[中文报告](docs/results/module_lr077_20261023/README_ZH.md) · [冻结协议](docs/results/module_lr077_20261023/PROTOCOL_FROZEN.json) · [冻结判定](docs/results/module_lr077_20261023/DECISION_RULES_FROZEN.json) · [代码](docs/results/module_lr077_20261023/code)

仅将原S-All完整学习率曲线乘0.7714。每步77,070,336名额、相同数据前缀、seed42、BF16权重/FP32动量和4293步不变；预检、训练、评测合计上限3 GPU小时。路径匹配要求S-Attn参考213.379786的±10%；未匹配不追加学习率重训。数学和正式测试未运行。

[五个问题的直接回答](docs/results/module_lr077_20261023/INTERPRETATION_ZH.md)

累计路径186.937635，未达到预定路径匹配范围；Attention相对新对照的联合门槛未通过。解释与限制见完整报告，不自动启动新学习率、seed或子空间实验。
