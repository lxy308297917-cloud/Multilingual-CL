# 顺序阶段的 adapter 接续接口

`sequential_adapter.py` 仅负责把上一阶段的 LoRA 权重加载到当前模型已有的唯一 `default` adapter。它不创建独立新 adapter，不 merge 到 Base，不调用 Trainer checkpoint resume，也不加载旧 optimizer／scheduler。检查上一阶段 global step、LoRA 配置、全部 adapter 键／形状／dtype／有限值、加载后的逐位权重、Parameter 对象身份及冻结 Base 指纹。指纹仅用于本机私有运行证据。

`check_sequential_adapter_cpu.py` 已在随机初始化的微型 Qwen3 上实际完成：

1. 一次真实 AdamW 更新后保存 adapter，加载到相同原始 Base 的唯一 adapter。
2. 学生 logits 与上一阶段一致，禁用 adapter 后教师 logits 与原始 Base 逐位一致。
3. 新 optimizer 状态为空，scheduler 从0开始；同一后续 batch 的更新与显式重置 optimizer 的独立参考逐位一致。
4. 后续更新后再次确认禁用 adapter 的教师 logits 不变。

结果见 `sequential_adapter_cpu_20261020.PUBLIC.json`。这是合成 CPU 工程验证，不是1.7B顺序训练、GPU rollout 同步、checkpoint恢复或数学效果证据；没有占用GPU或选择三语言，没有修改运行中的训练源码。

顺序训练入口尚未接入此接口。正式接入还需：固定三语言／数据池和每阶段预算；新阶段重新建立 Trainer optimizer、scheduler及global step；首次生成前同步已接续的学生并与上一阶段最终 adapter／推理权重核对；独立验证原始Base身份与禁用adapter的教师路径；各阶段保留完整恢复状态，并评全部入选语及英语。继续沿用同一adapter、语言与原题池绑定一起反转，两条顺序从相同原始模型开始。未满足语言选择条件时不启动顺序实验。

## 新阶段step0的初始同步参考桥

已有运行时按当前Trainer的global_step在参考JSON中查同步记录。新阶段会重置optimizer／scheduler和global_step到0；直接传上一阶段完整proof，会错误地选到上一阶段**初始**step0，不能验证接续的最终学生。

`build_stage_initial_reference.py`读取已完成父阶段最终checkpoint及其最终同步，保留父step并仅将该最终adapter／推理指纹映射为新阶段local step0。实际加载仍由`load_stage_adapter`完成，随后必须由运行时首次同步与这份参考逐项核对。它本身不加载模型或创建optimizer，不改变官方loss。生产模式要求相同order_id、前一stage_index及相同model revision；没有这些上下文的旧single只允许显式engineering-only输出。

已在真实完成的SW20文件上执行CPU构建：新step0的两种指纹与父最终20相等、与父初始0不同，原proof未变。私有路径与指纹保留本机；[公开范围报告](sequential_initial_bridge_CPU_20261020.PUBLIC.json)。这是接口准备，不是实际1.7B GPU接续或三语言顺序结果。后续入口应在官方Trainer完成唯一LoRA创建后、创建新optimizer前加载父adapter；首次rollout前再校验映射后的学生同步。

## 遗忘矩阵的计算约定与CPU验证

[forgetting_metrics.py](forgetting_metrics.py)接收Base及三个已完成阶段的完整正确数／分母矩阵，每行必须包含全部三个目标语及英语，每种评测语言的分母跨阶段一致；缺阶段、缺分数或变化的分母会拒绝，不能用0填缺失值。输出保留原计数及包含Base的完整准确率矩阵，并区分当前语相对前一阶段的能力获得、相对Base的变化、旧语从学完时起的下降及英语相对Base的保持变化。

预先固定遗忘定义：语言i在阶段k的遗忘为其**学完阶段i至前一阶段k−1**的历史最佳准确率减去当前k阶段准确率；不把当前阶段放入历史最大值，不包含Base，保留负数表示正迁移。刚学完的第三语不人为记0后混入旧语遗忘平均值；最终三语平均准确率包含全部三个目标语，英语单列。

[CPU检查](check_forgetting_metrics_cpu.py)使用手工可核算的合成矩阵验证了负遗忘、历史最佳与学完时下降的区别、英语及新语平均边界，并拒绝缺失矩阵。结果见[合成CPU验收](forgetting_metrics_CPU_20261020.PUBLIC.json)。这仅证明给定计数的计算接口，不证明真实模型／输入／提示／评分协议一致，也不是实际顺序训练或遗忘结果；真实来源须由外层逐题验收完成。尚未选择最终三语言或启动顺序训练。
