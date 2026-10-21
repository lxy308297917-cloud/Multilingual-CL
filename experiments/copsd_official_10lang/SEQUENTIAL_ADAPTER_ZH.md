# 顺序阶段的 adapter 接续接口

`sequential_adapter.py` 仅负责把上一阶段的 LoRA 权重加载到当前模型已有的唯一 `default` adapter。它不创建独立新 adapter，不 merge 到 Base，不调用 Trainer checkpoint resume，也不加载旧 optimizer／scheduler。检查上一阶段 global step、LoRA 配置、全部 adapter 键／形状／dtype／有限值、加载后的逐位权重、Parameter 对象身份及冻结 Base 指纹。指纹仅用于本机私有运行证据。

`check_sequential_adapter_cpu.py` 已在随机初始化的微型 Qwen3 上实际完成：

1. 一次真实 AdamW 更新后保存 adapter，加载到相同原始 Base 的唯一 adapter。
2. 学生 logits 与上一阶段一致，禁用 adapter 后教师 logits 与原始 Base 逐位一致。
3. 新 optimizer 状态为空，scheduler 从0开始；同一后续 batch 的更新与显式重置 optimizer 的独立参考逐位一致。
4. 后续更新后再次确认禁用 adapter 的教师 logits 不变。

结果见 `sequential_adapter_cpu_20261020.PUBLIC.json`。这是合成 CPU 工程验证，不是1.7B顺序训练、GPU rollout 同步、checkpoint恢复或数学效果证据；没有占用GPU或选择三语言，没有修改运行中的训练源码。

2026-10-21：`sequential_stage_entry.py`与`run_sequential_stage.sh`已接入此接口，复用固定官方核心及现有批量vLLM运行时。它在官方Trainer完成唯一LoRA创建后、建立optimizer前加载父adapter；新阶段global step／optimizer／scheduler重置，首轮rollout必须匹配父最终策略桥。阶段内恢复则加载本阶段有效checkpoint及其优化状态，强制新输出目录，保留旧文件。研究模式还要求冻结的独立开发选择记录、语言绑定题池及前一阶段七套评测完成标记；当前最终三语言未确定，外层七套评测编排与矩阵收集已补齐代码，仍待真实研究阶段验收，未启动研究顺序实验。

## 新阶段step0的初始同步参考桥

已有运行时按当前Trainer的global_step在参考JSON中查同步记录。新阶段会重置optimizer／scheduler和global_step到0；直接传上一阶段完整proof，会错误地选到上一阶段**初始**step0，不能验证接续的最终学生。

`build_stage_initial_reference.py`读取已完成父阶段最终checkpoint及其最终同步，保留父step并仅将该最终adapter／推理指纹映射为新阶段local step0。实际加载仍由`load_stage_adapter`完成，随后必须由运行时首次同步与这份参考逐项核对。它本身不加载模型或创建optimizer，不改变官方loss。生产模式要求相同order_id、前一stage_index及相同model revision；没有这些上下文的旧single只允许显式engineering-only输出。

已在真实完成的SW20文件上执行CPU构建：新step0的两种指纹与父最终20相等、与父初始0不同，原proof未变。私有路径与指纹保留本机；[公开范围报告](sequential_initial_bridge_CPU_20261020.PUBLIC.json)。这是接口准备，不是实际1.7B GPU接续或三语言顺序结果。后续入口应在官方Trainer完成唯一LoRA创建后、创建新optimizer前加载父adapter；首次rollout前再校验映射后的学生同步。

## 遗忘矩阵的计算约定与CPU验证

[forgetting_metrics.py](forgetting_metrics.py)接收Base及三个已完成阶段的完整正确数／分母矩阵，每行必须包含全部三个目标语及英语，每种评测语言的分母跨阶段一致；缺阶段、缺分数或变化的分母会拒绝，不能用0填缺失值。输出保留原计数及包含Base的完整准确率矩阵，并区分当前语相对前一阶段的能力获得、相对Base的变化、旧语从学完时起的下降及英语相对Base的保持变化。

预先固定遗忘定义：语言i在阶段k的遗忘为其**学完阶段i至前一阶段k−1**的历史最佳准确率减去当前k阶段准确率；不把当前阶段放入历史最大值，不包含Base，保留负数表示正迁移。刚学完的第三语不人为记0后混入旧语遗忘平均值；最终三语平均准确率包含全部三个目标语，英语单列。

[CPU检查](check_forgetting_metrics_cpu.py)使用手工可核算的合成矩阵验证了负遗忘、历史最佳与学完时下降的区别、英语及新语平均边界，并拒绝缺失矩阵。结果见[合成CPU验收](forgetting_metrics_CPU_20261020.PUBLIC.json)。这仅证明给定计数的计算接口，不证明真实模型／输入／提示／评分协议一致，也不是实际顺序训练或遗忘结果；真实来源须由外层逐题验收完成。尚未选择最终三语言或启动顺序训练。

## JA题池导出必须沿用已验收的数据修复

原题池清单固定于2026-10-19，引用JA修复前的2900条输入；已确认的70738字符重复前缀位于pool1。以后若JA入选，直接从原输入导出该池会重新引入坏记录。

`prepare_sequential_pools.py materialize`已支持`--ja-repaired-input`和`--ja-repair-manifest`。显式传入已有修复版及其原验收清单，核对原输入与修复版SHA、2900条ID顺序、唯一一条差异、该行除`problem_ja`外全部字段相同，且差异必须严格等于删除已确认的70738字符重复前缀。原池ID、语言绑定规则、英文题目／参考／gold与独立开发集隔离均保持。包含已知坏记录的JA原输入在缺少这些修复参数时拒绝导出；不改写官方源文件或旧池清单。

已将三个实际JA池临时导出并与现有修复版逐条比较：967／967／966、并集2900、池间零重叠，且未修复源被拒绝、没有写出不安全输出。无GPU、模型更新或新增标签；临时文件测试后删除。见[实际数据导出验收](sequential_JA_repaired_pool_export_CPU_20261021.PUBLIC.json)。这没有选择最终三语言或启动顺序训练，后续仍须完成adapter接续的实际训练入口和GPU验证。

## 实际Trainer接续CPU验证与GPU工程任务

`check_sequential_trainer_cpu.py`已经运行实际Transformers Trainer：微型随机Qwen3上父阶段一次AdamW更新，子阶段通过上述同一加载接口接续；观察真实Trainer新建优化器为空、scheduler／global step为0。再更新一次，与保留同一父模型但显式新建Trainer的独立参考全部参数逐位一致；学生logits接续一致，禁用adapter后的原Base教师不变。结果见[实际Trainer CPU验收](sequential_Trainer_integration_CPU_20261021.PUBLIC.json)。这是CPU合成工程证据，尚未验证真实1.7B GPU rollout或恢复完整性。

已手动启动`copsd-stage-handoff-engineering-v1`等待队列：复用已完成SW20，仅新增2个本地更新／64题次，仍为原2048长度、AdamW及官方损失。它单独验证接续、优化器重置和初始生成策略同步，不是新语言选择或三语研究结果；父single与工程池可能重叠，不声称题目独立。冻结设置见[工程任务清单](sequential_handoff_engineering_frozen_20261021.PUBLIC.json)。GPU任务完成后仍须核验真实checkpoint、优化状态、策略同步、冻结Base、完整64次损失调用与父进程exit0；目前仍在排队。

## 七套评测编排与真实结果收集（2026-10-21）

`sequential_evaluation.py`与`run_sequential_evaluation.sh`复用现有冻结`run_formal8192.sh`：每个Base／阶段必须包含三种入选语各自MGSM-Rev2与PolyMath，加英语GSM8K，共七套。仅接受SHA固定的独立开发选择记录；不会从正式成绩选语。当前预设仍为n=1／8192／并发16，若以后采用性能测试的新并发或合并权重后端，须另建冻结协议并对Base及全部模型一致评测，不能混用旧分数。

接口按`plan → prepare → run → collect`执行，不自行启动GPU队列或重试。`plan`保存准确的checkpoint、模型／输入／源码身份和显式缓存列表；缓存必须是同一实际adapter、预算和协议的完整输出，不能按分数选择。`prepare`仅检查一次共同环境与七套输入，不反复冷启动Python。`run`调用原正式评测入口；新输出不得已存在，失败后保留目录、另建恢复计划并显式复用已完成阶段。七套输出完成不等于父队列退出；`collect`须确认指定队列exit0及冻结plan一致，逐题核对输入、seed、token／正确数和PolyMath四档计数后，才在当前stage目录写出供下一阶段读取的`STAGE_EVALUATIONS_DONE.json`。

已用真实保存的SW MGSM Base／两个SW20输出在CPU上执行收集核验，共750条，正确数8／13／14与此前独立阶段报告一致。错adapter、错误输入SHA、错语言、不完整七套集合，以及把正式报告当作开发选择证据均被拒绝，见[实际输出CPU验收](sequential_evaluation_real_outputs_CPU_20261021.PUBLIC.json)。这没有创建生产选择记录、启动GPU、生成新回答或证明真实七套阶段管线成功。

`build_sequential_matrix.py`接收Base＋三个阶段的四份完成标记，再复核其实际plan、checkpoint链、共同输入／prompt／seed／评分配置和训练题池。它分别生成MGSM／PolyMath的完整矩阵，调用已有遗忘计算器保留负遗忘；同一英语GSM8K列在两个矩阵中展示，成本不重复加倍。Base结果可以由正反顺序共享。该收集器目前仅完成语法／CLI检查，真实三阶段尚不存在，不能声称端到端验收通过。语言保持、匹配题池single对照和完整恢复成本账目仍须另行完成；输出中的最终进程段成本明确不冒充恢复前后的总成本。
