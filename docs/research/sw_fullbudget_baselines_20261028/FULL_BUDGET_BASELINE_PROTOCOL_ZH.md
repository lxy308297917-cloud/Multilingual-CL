# SW完整预算基线协议（准备与技术预检阶段）

正式结果尚未产生。六份训练均从同一Qwen2.5-1.5B-Instruct Base重新开始，另有Base统一评测。本轮不训练N/N_abs、表示对齐或访问修复方法。

主流为Aya80%+EN→SW10%+SW→EN10%，按主输入Token。完整计数15,360,935输入、5,984,371监督、84,930原始样本呈现，42,465次更新；源索引无重复，但跨方向/同内容组并非互不重叠。原始lossmask、2048长度、微batch1累积2、样本均值CE、seed42顺序相同。

共同峰值5e-5、warmup2124、覆盖全部42465步的cosine，非短训日程前缀。BF16权重、FP32Adam动量；betas.9/.999、eps1e-8、wd.01、clip1。无超参数扫描或测试选checkpoint。

## 参数预算

HFT15、SSU、U/Element、P/Protected严格统一196,608,000元素/步，分母为196个Attention/MLP矩阵1,310,195,712元素，实际15.0060024%。HFT15是历史整矩阵预算匹配适配，不是原HFT50%；SSU是低英语重要性整列静态选择，整数列数匹配公共预算；U/P按矩阵参数量分配公共整数名额、每20步按动量刷新，P仅以英语平方梯度敏感度改变评分。共同矩阵外embedding/head/Norm/bias冻结。

SSU在clip前清零被屏蔽坐标梯度，避免被保护坐标影响全局梯度裁剪；写回和衰减只限选中列。HFT只有选中矩阵参与反向。U/P候选矩阵全部推进动量一次，但仅选中元素写回和衰减。

MoFO复核官方每步动量阈值及全候选weight decay次序；在共同196矩阵范围内使用名义15%自适应分量，ties可能超额、全候选衰减可能掩码外写回，明确标记为196矩阵适配，不能声称全部写回严格同预算。Dense保留全部唯一参数（包括tiedembedding/head及Norm/bias）更新，FP32动量、Norm/bias不衰减；不参与稀疏预算相等的声明。

## 英语校准

共用训练侧reading/instruction/math来源池766例，不认证Base掌握。排除已知错误参考alpaca:43259和与完整MMLU文本重叠的RACE:train:high3601.txt:2。SSU静态128例；P每200更新32例，平方batch平均梯度后CPUFP32 EMA.9。各自输入、监督曝光和前后向成本单列，不称为同总计算。

## 测评与原始性

全部七模型统一Belebele EN/SW各900（已修正同源fewshot，MCF3shot ACC）、MMLU/GlobalMMLU各14042（57学科完整，MCF5shot，微平均为主、学科宏平均补充）、本地FLORES-200派生完整1012题/方向（0shot固定提示、语料级chrF++）、官方IFEval541（提示与指令strict）、GSM8K1319（lm_eval0.4.8 gsm8k5shot strict-match，flexible-extract另列）。不把格式提取失败称为内容错误；保留完整原始回答。

EN/SW同题四状态与Base到各模型对错转移单独提供，单次正确不等于掌握。所有已暴露benchmark如实称补充比较，不称全新独立测试。词面精确/16词重叠检查不能证明语义去污染。

主要结果仅完整预算终点；预检及中间模型不填主表，不做综合混合总分。输出主表CSV/Markdown、相对Base和FFT差值、来源配对区间、完整逐题和成本身份记录；区间不代表多seed稳定性。

## 执行

全账号最多四卡；无GPU小时或墙钟停止上限。checkpoint实体在/data1/lixiaoyang/sw_fullbudget_baselines_20261028，统一入口/data0/users/lixiaoyang/models/checkpoint/sw_fullbudget_baselines_20261028。每2000步及终点保存权重/FP32动量/调度/RNG/数据位置。全部真实方法预检、Base各任务smoke和CPU评分验收通过后才能开始正式训练；一次性systemd用户协调器与会话退出隔离，无开机自启或故障自动重启。普通故障由助手保留日志后按授权修复恢复。

独立研究分支：research/sw-fullbudget-baselines-20261028；实现准备目录outputs/sw_fullbudget_baselines_20261028，工作与逐步日志work/sw_fullbudget_baselines_20261028。代码发布前保留冻结身份与方法适配说明。
