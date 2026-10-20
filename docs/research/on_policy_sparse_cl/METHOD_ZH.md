# 方法草案：On-Policy Risk-Aware Sparse Plasticity（OP-RASP，暂名）

日期：2026-10-09。**这是可评议的研究方案与 CPU 核心原型，不是完成的论文、真实模型训练结果或已确认原创方法。**

## 1. 研究问题及假期证据

问题：在学生真正生成到的状态上，能否按“候选更新的目标收益与旧能力风险”在线选择固定数量的参数，使模型逐阶段积累能力？

历史 SW D2 已显示：15% 动态英语保护的双向翻译/英语阅读在两训练顺序有正向配对证据；固定 mask、EMA mask 未通过开发门槛；LP 名额集中导致两轮 EN→SW 明显落后同预算均匀。我们因此保留逐矩阵精确名额和跨模块覆盖，不再直接使用 Base 上一次探测的模块 LP。以上是动机，不能推出新方法必然有效。

旧方法用目标一阶动量和英语梯度平方决定选位，训练数据前缀固定，英语信号主要影响排名。上一候选“固定前缀上加 Base KL”不是 on-policy。本方案同时改变状态采样来源与选位信号；必须通过正交消融拆开验证。

## 2. on-policy 的严格含义

每次更新前，从当前学生 theta_t 在不含答案的任务提示 x 上按完整 categorical 分布、温度1采样 y。教师只在评分时看到 demonstration c，并在**相同学生答案前缀 y_<u**上给出监督分布。保留历史能力时，也由当前学生在合法旧任务提示上采样，再由固定参考评分。

- 只动态改 mask，或把学生生成缓存长期复用，不能叫严格 on-policy。
- 学生看到参考答案后生成，是信息泄漏；只有教师允许看 demonstration。
- 生成动作不等于正确答案；不直接把所有生成当成正确硬标签。
- prototype 每次参数更新必须重新生成，版本不匹配拒绝。之后若允许复用，另称 near/on-policy-window，并量化陈旧度；不能静默混用。
- 原型只支持温度1、无top-k/top-p截断，严格对应定义的策略。其他采样分布需要改目标并记录。

## 3. 教师、参考与局部目标

新任务教师 Q_k 使用阶段开始模型的冻结副本，并看到该样本 demonstration；学生不看 demonstration。固定教师是首个简单候选，**不是 SDFT 的逐项复现**，其 EMA 教师等设计必须在相应基线中匹配。Qwen1.5B 能否有效利用示范是前置实验，不默认成立。

旧任务使用两个概念不同的参考：原始 Base 保留英语通用能力；上一阶段 checkpoint 保留最近学得能力。旧提示来自训练侧记忆池，任务等权后各取固定名额。需要保存旧提示，故本方案**不是 memory-free**。多阶段历史任务不能只取最近一个任务，否则可能漏掉更早技能。

在当前学生采样的答案位置上，首版用局部 forward KL：

```text
L_new = E[current-student new prefixes] mean KL(Q_k || P_theta)
L_old = task-balanced E[current-student old prefixes] mean KL(R_task || P_theta)
L_update = L_new + lambda * L_old
```

KL按完整词表、有效答案位置计算，先按样本平均再按任务平均。教师和参考 logits detach，提示位置/padding/EOS后填充不计入；EOS本身计入。不同教师提示长度需要显式对齐相同答案位置。

这里对生成前缀占用分布 stop-gradient，得到局部 semi-gradient surrogate。**不是完整序列KL的无偏梯度**；不包括状态分布变化的score-function项，也不是PPO/GRPO。Forward/reverse KL都在原型中支持，首轮固定forward；不得把不同方向称为同一算法。参考SDFT论文的理论反向KL与实际前向KL实现存在区分。

## 4. 关键候选贡献：在实际更新坐标上选择

取独立梯度 g_new、g_old，基于 g_new+lambda*g_old 及当前FP32动量构造含weight decay的AdamW候选更新 d_i。然后估计：

```text
u_i = -g_new_i * d_i
r_i = max(g_old_i * d_i, 0) + 0.5 * f_i * d_i^2
score_i = u_i - beta * r_i
```

u_i是新任务局部预计损失下降，r_i是保守的不利一阶变化加非负对角敏感度代理。旧梯度可能改善旧任务：这里只扣不利部分，忽略有利抵消，这是刻意的保守启发式，须与保留signed风险的变体区分。

f_i为旧on-policy轨迹的sampled sequence log-probability score梯度平方均值。它是噪声较大的经验对角代理，**不是经过校准的真实Fisher/Hessian上界**。教师=学生时锚定KL梯度为0，这个非零敏感度代理可以避免首步完全无风险信号；有效性必须实测。

对每矩阵j精确选择floor(0.15*N_j)个最高score元素；仅在这些位置写回d_i。候选仍覆盖196个Attention/MLP矩阵，其他权重冻结。不同矩阵没有跨组抢名额，避免模块覆盖崩塌。原型每步刷新；和旧20步刷新比较时须补“相同刷新周期的原排序”控制。

这是可分离score下的top-k选择，不是求解真正风险约束优化。强制精确名额时可能选入负收益或高风险坐标；损失曲率的跨参数项、Adam路径依赖、BF16舍入、梯度噪声都可能破坏预测。不能据这个式子宣称“不遗忘保证”。

原型维护所有候选的动量；未选参数及其weight decay不写回。每步计划名额不等于实际非零写回、历次掩码并集、最终BF16差异或FLOP节约。未来记录以上四种数量，以及实际扰动范数。

## 5. 一次更新的伪代码

```text
freeze teachers/references for this stage
sample new trajectories from CURRENT student without demonstrations
sample old trajectories from CURRENT student on old training prompts
score exact same generated prefixes with demo-teacher and old references
compute independent g_new, g_old and empirical old sensitivity proxy
form actual proposed AdamW deltas including weight decay
choose exact per-matrix top-k by new utility minus old risk
write only selected deltas; log utility/risk/counts
advance student version; discard trajectories; regenerate for next update
```

“先生成—后评分—再选位”的耦合是候选主线，而非把on-policy当作选参策略的名字。

## 6. 与先例的区别及创新不确定性

- [SDFT](https://arxiv.org/abs/2601.19897)：已有示范条件化教师、学生轨迹及持续学习；不能作为本项目原创点。
- [Dense Supervision, Sparse Updates](https://arxiv.org/abs/2606.13657)：已有OPD的稀疏几何观察及发现子网的masked training；“首次结合稀疏与OPD”不成立。
- [iSDFT](https://arxiv.org/abs/2609.24646)：已有信息预算及固定Base锚定；“首次控制teacher影响/加Base KL”不成立。
- [On-Policy Replay](https://arxiv.org/abs/2605.29495)：已有最新checkpoint生成旧提示回放；采样旧任务也不能单独称创新。
- [Learning without Overwriting](https://arxiv.org/abs/2610.05200)：已有特殊推理图模型下OPSD稀疏更新的理论分析；不能移植其保证到此AdamW启发式。

尚可检验的贡献假设：**更新时刻在新旧学生轨迹上、对实际AdamW更新估计风险，并在精确逐矩阵预算下选择可塑性支持**，是否比“OPD＋现有稀疏排序”更好的能力/成本Pareto取舍。必须进一步核对signed-gradient、Fisher/EWC、GAST等相关选择先例；当前文献检索不是穷尽性新颖性证明。

## 7. 范围与失败条件

第一步仅监督翻译/指令；D2的70%无标签CPT不直接有demonstration-conditioned教师。若扩展D2，CPT保留CE、监督部分用OPD，另建新协议，不能声称完整固定D2输入token与旧结果严格相同。

当前实现是CPU核心，不包括Qwen训练适配、生成hints模板、可验证反馈、实际CL数据流、精确clip/scheduler顺序、完整resume或GPU入口。只有teacher质量、代理稳定性和同预算强对照通过，才能升级实验。
