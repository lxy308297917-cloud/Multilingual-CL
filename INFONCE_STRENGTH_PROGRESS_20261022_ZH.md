# 增强InfoNCE最终实验：已完成

4293步D1、16批训练侧强度校准、完整冻结开发评测与独立1000对表示诊断已完成；原始服务器模型、优化器和输出身份已核验。

[完整中文报告](docs/results/infonce_strength_20261022/README_ZH.md) · [冻结协议](docs/results/infonce_strength_20261022/PROTOCOL_FROZEN.json) · [逐项验收](docs/results/infonce_strength_20261022/REQUIREMENTS_AUDIT.json) · [实现](docs/results/infonce_strength_20261022/code)

关键结论：当前模型独立EN↔SW检索从D0的8.45%升为D1的63.00%，但主指标EN→SW翻译为27.024，未超过同池翻译CE的27.621；差值95%区间为[-2.498,+1.350]。全部冻结门槛未通过，停止当前固定原型，不追加扫描。

这是单seed开发筛查，未运行正式测试或数学排名。与Attention/MLP等元素预算实验分别报告。
