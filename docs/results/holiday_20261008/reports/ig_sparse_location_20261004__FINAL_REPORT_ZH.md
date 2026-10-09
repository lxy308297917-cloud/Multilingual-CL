# Igbo 稀疏定位平行对照

四个模型按新建的同一固定评测配置执行；下表仅为单次 seed42 结果。历史密集 FFT 使用相同 MADLAD 数据和标准 AdamW，但与两个新稀疏组的参数冻结方式不同。

|指标|Base|密集 FFT|首尾层 15%|分散层 15%|首尾−分散|
|---|---:|---:|---:|---:|---:|
|belebele_ig|30.2222|30.8889|30.4444|29.8889|+0.5556|
|sum_ig|17.3512|23.1413|20.2464|16.2669|+3.9795|
|mt_en2ig|11.6024|18.3305|15.6731|13.3263|+2.3467|
|mt_ig2en|16.7040|15.2345|15.9981|12.2278|+3.7703|
|belebele_en|81.7778|78.4444|81.1111|81.6667|-0.5556|
|sum_en|21.9664|19.6773|21.3991|22.3923|-0.9932|
|ifeval_prompt_strict|39.9261|19.0388|35.3050|36.4140|-1.1091|
|ifeval_instruction_strict|49.6403|31.0552|46.0432|46.7626|-0.7194|
|gsm8k_strict|33.0553|18.8779|28.7339|33.4344|-4.7005|
|mmlu_en|61.2763|59.3773|60.3577|60.8840|-0.5263|
|gmmlu_ig|31.7747|35.1351|33.8702|31.7530|+2.1172|

两稀疏组完全相同的样本读取顺序 SHA256：`0625132e7fab83b0a5575e114d7d0e227e139f96c2a3facfdd1e2cfd57b660b1`。
PPL/NLL 仅留作诊断，不能替代上述下游任务。
这些组检验参数位置，不含动态英语保护；单 seed 差异不能证明稳定收益。

原始逐任务结果和样本输出：
- base: `/data0/users/lixiaoyang/logs/ig_sparse_location_20261004/evaluation/evaluation/6ea4a4edab8e22ce/primary`
- dense_fft: `/data0/users/lixiaoyang/logs/ig_sparse_location_20261004/evaluation/evaluation/2b513e9a80ac814a/primary`
- edge: `/data0/users/lixiaoyang/logs/ig_sparse_location_20261004/evaluation/evaluation/2311c2b4b69cecd0/primary`
- spread: `/data0/users/lixiaoyang/logs/ig_sparse_location_20261004/evaluation/evaluation/914a705b07fc59a9/primary`
