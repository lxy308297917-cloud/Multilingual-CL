# 原生SW阅读数据验收

官方 TyDiQA-GoldP secondary_task/train；版本 da78f23f9119363459acbaf46bf89426ff26c259。原始SW 2755条，初筛后2080条。

|阶段|条数|文档组|唯一输入token|唯一监督token|
|---|---:|---:|---:|---:|
|train|1928|863|478384|15622|
|dev|146|47|51091|1037|

排除计数：{"raw_sw": 2755, "invalid_answer_span": 478, "fixed_eval_overlap": 162, "documented_content_failure": 10, "duplicate_original_qa": 23, "conflicting_qa_groups": 2, "same_qa_in_aya": 0, "accepted_before_group_split_and_length": 2080}

C实际阅读采样：{"presentations": 5784, "unique_samples": 1928, "input_tokens": 1435152, "supervised_tokens": 46866, "max_repetitions": 3}

输入占比 9.3424%；占全部监督token 1.0118%。三组统一2048；不截断SFT。

全部保留阅读样本已校验完整模板和监督mask，只有答案及结束符受监督。格式化样本与完整mask见NATIVE_READING_FORMATTED_EXAMPLES.jsonl。

偏移检查与自动完整性验证不等于全量人工内容审核。已确认有内容缺陷的10条排除；没有将其余样本写成人工验证通过。

C额外TyDiQA监督不等于C全部阅读监督：Aya还含CoQA。C−B同时改变阅读、摘要与翻译，不能把全部变化归因于TyDiQA。
