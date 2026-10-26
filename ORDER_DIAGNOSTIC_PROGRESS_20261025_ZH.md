# LR077训练顺序敏感性诊断

状态：CPU身份与数据流核查、两份2146步checkpoint评测及4293步端点比较完成。

[中文报告](docs/results/order_diagnostic_20261025/README_ZH.md) · [分项结果](docs/results/order_diagnostic_20261025/RESULTS.json) · [冻结协议](docs/results/order_diagnostic_20261025/PROTOCOL_FROZEN.json)

不新增训练，冻结256题×两模型，共512条新回答，2GPU小时上限。中间点仅诊断，不选最佳checkpoint。原始输出和权重保留服务器；这里仅聚合结果与脱敏代码。
