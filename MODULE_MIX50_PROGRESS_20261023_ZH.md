# S-Mix50：模块名额与训练顺序验证

状态：四份短训与冻结诊断完成。

[中文报告](docs/results/module_mix50_20261023/README_ZH.md) · [冻结协议](docs/results/module_mix50_20261023/PROTOCOL_FROZEN.json) · [数据缺口](docs/results/module_mix50_20261023/DATA_STATUS.json) · [脱敏代码](docs/results/module_mix50_20261023/code) · [公开范围](docs/results/module_mix50_20261023/PUBLICATION_SCOPE.json)

新增LR077-s43、Attn-s43、Mix50-s42、Mix50-s43。每步总名额77,070,336，Mix50在Attention/MLP各分38,535,168；最多四卡，整轮上限16GPU小时，无正式测试、数学或额外参数扫描。

新翻译确认按跨语言共同原文档排除历史开发池及训练/近重复后仅剩1个可用FLORES dev文档，不足128。不换题源、不动正式测试。新确认仅阅读/指令256题，原EN→SW升级标准不可完整验收。历史开发池曝光不等于当前模型训练泄漏。

[一页中文结论](docs/results/module_mix50_20261023/INTERPRETATION_ZH.md)

Mix50两训练顺序平均EN→SW相对LR077为-2.002 chrF++，来源配对95%区间[-2.896,-1.118]（旧开发题）。Attention旧题优势未在第二顺序复现；建议停止当前静态Mix50扩展，不直接进入子空间。完成4096条新回答，成本9.8535 GPU小时。新翻译确认仍缺失，原升级标准不可完整验收。
