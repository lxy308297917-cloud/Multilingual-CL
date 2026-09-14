# Igbo baseline 代码发布（2026-09-14）

这是基于 SSU 上游实现的 Qwen2.5-1.5B-Instruct／Igbo 实验代码快照。训练和正式评测仍在进行，本次发布不表示七种方法均已完成评测，也不表示完全复现 SSU 原论文。

## 入口

在已准备好冻结数据、模型、双环境和目录布局的实验服务器上：

```bash
cd /root/autodl-fs/ssu-project/ssu-main
# 仅查看冻结训练命令，不启动训练
bash scripts/baseline_train.sh full_fft --plan
bash scripts/baseline_train.sh ssu_en_freeze50 --plan
bash scripts/baseline_train.sh mofo15 --plan
# 已通过该源码版本的训练放行审计后，执行一个方法
bash scripts/baseline_train.sh full_fft --gpu 2
# 冻结评分入口：只更换完整模型路径
bash scripts/baseline_eval.sh /完整模型路径
# 推荐：同一评分协议，增加语言任务优先的阶段调度和显存分配设置
bash scripts/reproduce_baseline_eval.sh /完整模型路径
```

`baseline_train.sh` 是本次发布增加的薄入口，仅转发到现有 `train_cl.py` 和原样冻结的 `configs/ig_baseline_train_v1.json`。其他训练、评分源码按当前服务器文件逐字节保存。不要在正在运行的实验目录覆盖部署或同时手动重复启动后台已有任务。

## 七种方法

|方法参数|含义|实现入口|
|---|---|---|
|full_fft|全参数更新|training/src/main_bf16.py|
|ssu_en_freeze50|英语校准重要性选列，名义冻结50%|main_bf16.py + utils/model_utils.py|
|current_top20_all_layers|Igbo 激活与权重分数选择列|main_bf16.py + utils/model_utils.py|
|hft50|逐层随机冻结模块|main_bf16.py + utils/model_utils.py|
|lota90|100步校准选择后恢复Base，重新建立正式训练优化器|main_bf16.py + utils/model_utils.py|
|mofo15|逐张量按动量选取top15%，包含阈值并列项|training/src/run_mofo_bf16.py + utils/mofo.py|
|layers1_5_23_27_fft|仅训练零基编号1–5和23–27层|main_bf16.py 的整层选择|

共同训练预算：Igbo train10k、31575个512-token blocks；seed/data_seed42；15000 steps，batch2，梯度累积1，BF16，非重入梯度检查点；学习率5e-5，AdamW，cosine，5% warmup，weight decay0.01，clip1.0。只保存最终模型，不提供中间optimizer断点恢复；中断的方法需从Base重训。

名义比例不等于全模型有效更新比例：SSU、Top20和LoTA涉及embedding/head例外；LoTA额外100步校准单独计算成本。实际比例应以各次运行的参数审计为准。权重发生数值变化的比例与梯度允许更新的比例也不同。

## 冻结评测

机器配置：`configs/ig_baseline_eval_v3.json`；正式任务均只测seed42一次。不是三seed，不搜索最优解码方式。

- EN／IG Belebele：各900条完整测试，3-shot。
- EN／IG摘要：各固定500条，0-shot，逐句chrF++均值。
- 双向翻译：各固定500条，3-shot，逐句chrF++均值。
- MMLU／G-MMLU：57学科完整测试，5-shot，学科准确率宏平均；fewshot保持本项目validation口径。
- IFEval：541条；GSM8K：1319条，项目8-shot CoT，与SSU官方5-shot不同。
- PPL/NLL：独立诊断，不计入能力综合。

推荐封装先六项阅读/摘要/翻译，再MMLU/G-MMLU，最后通用任务；已完成任务经身份和文件哈希验证后跳过。封装需要已有 `protocol_lock.json`，目前用于已经冻结的服务器输出轨道。原入口支持 `bash scripts/baseline_eval.sh /完整模型路径 --check` 做只读计划校验；该选项不会创建协议锁。原入口首次正式运行会建立锁，但不提供同样的语言任务阶段屏障。新机器的初始化与目录迁移仍需另外核验，不能把参考锁直接冒充新环境验证。

EN3 = mean(EN阅读、EN摘要、IG→EN各项相对同协议Base变化)。Target3 = mean(IG阅读、IG摘要、EN→IG各项相对同协议Base变化)。PPL、MMLU、G-MMLU和通用任务不混入这两项综合。

摘要和翻译固定do_sample=true、5 beams、temperature0.8、top_k40、top_p0.9、max_new_tokens128；通用任务1 beam、top_p0.8、max_new_tokens1280。完整说明见 `BASELINE_EVAL_ZH.md`。

## 依赖和复现边界

本次包含所需LightEval源码快照 `../lighteval_latest/`，附上游许可证。不要把旧的顶层 `lighteval` gitlink 当成当前评测依赖。训练源码和所有evaluation Python文件也包含在快照内，因为冻结协议会对它们整体计算哈希。部分公共模块含其他实验方法，但本发布只承诺上述baseline路由。

冻结配置保留服务器绝对路径，不是全新机器下载后即可运行的安装包。需要准备：Base模型/tokenizer、同一Igbo训练blocks、英语校准数据、摘要/翻译固定子集、选择题数据、IFEval/GSM8K和固定双环境。运行还依赖服务器已有的评测输入身份清单和训练release；这次仅发布代码与冻结配置，不上传内部审计包、环境快照或数据清单。这意味着代码快照保留了校验逻辑，但全新机器还需要独立准备依赖并通过放行审计。LightEval源码必须位于协议指定路径并由评测环境使用。

不同方法在不同时间放行，源码身份可能与本发布快照不同。更换目录、代码或数据后必须重新核验并建立独立记录，不能删除校验来强行运行。Top20还需准备对应校准分数文件。此快照不提供新机器的一键数据准备或训练放行工具。

本次不包含checkpoint、optimizer状态、缓存、训练日志、完整逐样本预测或未完成的最终分数。后续完整结果另行发布。原来的 `training/scripts/`、`evaluation/scripts/` 保留为历史示例；当前baseline请使用本页入口。
