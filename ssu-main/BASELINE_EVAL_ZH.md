## 最新用户指定顺序

每个模型先完成六项阅读、摘要、翻译，再完成MMLU和G-MMLU，两项均完成后才进入通用任务。`reproduce_baseline_eval.sh /模型路径`自动执行阶段屏障；原始冻结baseline_eval.sh与v3评分代码未改动。已完成任务按原身份校验并跳过，调度不会重测。Base已完成八项语言任务，因此当前IFEval继续执行。PPL仅诊断、不进入综合；原尾部执行器的PPL可与尚未结束的GSM8K并行，不影响语言任务优先要求。

## 推荐的一条命令（显存恢复后）

```bash
bash /root/autodl-fs/ssu-project/ssu-main/scripts/reproduce_baseline_eval.sh /模型路径
```

此薄启动脚本固定 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，再调用原 `scripts/baseline_eval.sh`。原入口、v3配置、任务代码和结果身份不变；模型之间只修改模型路径。该分配器设置已用于完整Base G-MMLU成功重试，解决本次出现的显存分配失败；不改变样本数、提示词、seed或评分。不承诺任意硬件都不会OOM。封装身份记录见实验目录 `REPRODUCIBLE_RUNTIME_LAUNCHER.json`。当前正式结果仍为单seed42，不增加重复评测。

# 统一Igbo基线评测：固定seed42

```bash
bash /root/autodl-fs/ssu-project/ssu-main/scripts/baseline_eval.sh /完整模型路径
```

当前配置：configs/ig_baseline_eval_v3.json。正常命令只评测，不训练，不搜索超参数，不进行多seed或多种解码方式对照。同一任务固定seed42、固定样本与顺序、固定prompt/tokenizer和评分方法。GPU0、1处理互不重叠的任务，每项只执行一次。

|任务|范围|评分|
|---|---|---|
|Belebele EN、IG|各完整900条，3-shot|准确率|
|摘要EN、IG|各固定500条，0-shot|逐句chrF++均值|
|EN→IG、IG→EN翻译|各固定500条，3-shot|逐句chrF++均值|
|MMLU EN、G-MMLU IG|各57学科完整测试，5-shot；fewshot沿用冻结validation口径|学科准确率宏平均|
|IFEval|完整541条|prompt strict与instruction strict|
|GSM8K|完整1319条，固定8-shot CoT模板（与SSU官方gsm8k 5-shot不同）|strict与flexible答案匹配|
|PPL/NLL|现有EN/HA/IG/KY诊断集，各8192 token预算|单独诊断，不进入综合|

以上为11个调度任务，其中知识任务内部展开学科。

## 固定生成设置

|任务|seed|do_sample|num_beams|temperature|top_k|top_p|max_new_tokens|
|---|---:|---|---:|---:|---:|---:|---:|
|摘要、翻译|42|true|5|0.8|40|0.9|128|
|IFEval、GSM8K|42|true|1|0.8|40|0.8|1280|

两组repetition_penalty均为1.1。摘要和翻译使用early_stopping=true。选择题通过候选答案似然评分，不用生成采样来选答案。任务之间允许有不同但固定的设置；同一任务对所有模型使用完全相同的设置，不尝试多套参数后挑最好结果。当前不是纯greedy解码。

此前多seed计划已被用户2026-09-14的新要求取代。seed43/44只保留历史记录，不进入当前正式统计。已完成或正在完成的同条件seed42结果通过身份/哈希及逐任务设置核验后复用，REUSE_PROVENANCE.json记录出处，避免重复计算。

## 结果与恢复

输出：/root/autodl-fs/ssu-project/experiments/ig_baseline_eval_v3。每个任务保存身份、配置关联、实际生成参数、原始分数、逐样本记录及DONE哈希。再次执行命令会验证并跳过完整任务，继续缺失任务，不自动重新评测已完成任务。异常或不一致会停止报错。

相对变化需要同条件Base。第一次评测其他模型时，如果Base尚不完整，会先补齐Base；以后复用已完成的同协议Base，不是每换模型都重跑Base。

EN3 = EN Belebele、EN摘要、IG→EN各自相对Base变化的等权平均。
Target3 = IG Belebele、IG摘要、EN→IG各自相对Base变化的等权平均。
PPL/NLL、MMLU/G-MMLU、IFEval、GSM8K不混入这两项综合，分别报告。

持续结果表：../experiments/ig_baseline_suite_v1/FORMAL_RESULTS_PROGRESS_ZH.md。未完成项保持空白，不按零分处理。

## 2026-09-14对齐审查
用户允许合理的现有任务设置。继续v3，不迁移原始全量test。摘要EN/IG各500条已通过源数据重建审计：test+validation合并，Qwen tokenizer长度排序最短500条，ID、顺序、正文及参考摘要全部一致。翻译固定500条；FLORES+源读取返回401，官方抽样身份尚待核实，不宣称已完全验证。GSM8K为8-shot CoT，知识任务保持本协议固定fewshot分区，均与官方差异单列。

2026-09-14最新数据核验：翻译数据已对固定版本FLORES-200源数据核验：测试500对及dev示例997对的英语/Igbo文本与顺序全部一致；测试取样为seed42随机500条。未声称与不可访问的当前FLORES+版本逐字相同。 证据：official_data_audit/mt_bilingual_reference_audit.json。
