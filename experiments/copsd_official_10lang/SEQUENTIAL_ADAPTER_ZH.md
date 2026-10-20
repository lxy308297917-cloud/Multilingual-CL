# 顺序阶段的 adapter 接续接口

`sequential_adapter.py` 仅负责把上一阶段的 LoRA 权重加载到当前模型已有的唯一 `default` adapter。它不创建独立新 adapter，不 merge 到 Base，不调用 Trainer checkpoint resume，也不加载旧 optimizer／scheduler。检查上一阶段 global step、LoRA 配置、全部 adapter 键／形状／dtype／有限值、加载后的逐位权重、Parameter 对象身份及冻结 Base 指纹。指纹仅用于本机私有运行证据。

`check_sequential_adapter_cpu.py` 已在随机初始化的微型 Qwen3 上实际完成：

1. 一次真实 AdamW 更新后保存 adapter，加载到相同原始 Base 的唯一 adapter。
2. 学生 logits 与上一阶段一致，禁用 adapter 后教师 logits 与原始 Base 逐位一致。
3. 新 optimizer 状态为空，scheduler 从0开始；同一后续 batch 的更新与显式重置 optimizer 的独立参考逐位一致。
4. 后续更新后再次确认禁用 adapter 的教师 logits 不变。

结果见 `sequential_adapter_cpu_20261020.PUBLIC.json`。这是合成 CPU 工程验证，不是1.7B顺序训练、GPU rollout 同步、checkpoint恢复或数学效果证据；没有占用GPU或选择三语言，没有修改运行中的训练源码。

顺序训练入口尚未接入此接口。正式接入还需：固定三语言／数据池和每阶段预算；新阶段重新建立 Trainer optimizer、scheduler及global step；首次生成前同步已接续的学生并与上一阶段最终 adapter／推理权重核对；独立验证原始Base身份与禁用adapter的教师路径；各阶段保留完整恢复状态，并评全部入选语及英语。继续沿用同一adapter、语言与原题池绑定一起反转，两条顺序从相同原始模型开始。未满足语言选择条件时不启动顺序实验。
