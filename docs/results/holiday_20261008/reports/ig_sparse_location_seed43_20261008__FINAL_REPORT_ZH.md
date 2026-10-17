# Igbo 稀疏定位第二训练顺序复测

四个模型按新建的同一固定评测配置执行；下表两条稀疏组为 seed43 训练顺序，评测仍固定 seed42。历史密集 FFT 使用 seed42 训练顺序；它只提供参考，不能作为严格同顺序训练对照。

|指标|Base|密集 FFT|首尾层 15%|分散层 15%|首尾−分散|
|---|---:|---:|---:|---:|---:|
|belebele_ig|30.2222|30.8889|30.5556|27.4444|+3.1111|
|sum_ig|17.3512|23.1413|19.6266|14.2021|+5.4245|
|mt_en2ig|11.6024|18.3305|16.2528|13.1598|+3.0930|
|mt_ig2en|16.7040|15.2345|15.2106|13.3018|+1.9087|
|belebele_en|81.7778|78.4444|81.0000|81.1111|-0.1111|
|sum_en|21.9664|19.6773|21.5816|22.0705|-0.4889|
|ifeval_prompt_strict|39.7412|19.0388|35.6747|36.7837|-1.1091|
|ifeval_instruction_strict|49.5204|31.0552|45.4436|47.7218|-2.2782|
|gsm8k_strict|33.0553|18.8779|24.7157|40.4094|-15.6937|
|mmlu_en|61.2763|59.3773|60.2641|60.3991|-0.1350|
|gmmlu_ig|31.7747|35.1351|33.4521|31.0574|+2.3948|

两稀疏组完全相同的样本读取顺序 SHA256：`bf5a3aaa85bd18293fe2e1491f43ac8ec2db1abc2500bf6c80bd5b5c217c1828`。
PPL/NLL 仅留作诊断，不能替代上述下游任务。
两条新训练组检验参数位置，不含动态英语保护；须结合 seed42 第一轮判断方向是否重复。

原始逐任务结果和样本输出：
- base: `/data0/users/lixiaoyang/logs/ig_sparse_location_seed43_20261008/evaluation/evaluation/6ea4a4edab8e22ce/primary`
- dense_fft: `/data0/users/lixiaoyang/logs/ig_sparse_location_seed43_20261008/evaluation/evaluation/2b513e9a80ac814a/primary`
- edge: `/data0/users/lixiaoyang/logs/ig_sparse_location_seed43_20261008/evaluation/evaluation/53907fcb871f44b3/primary`
- spread: `/data0/users/lixiaoyang/logs/ig_sparse_location_seed43_20261008/evaluation/evaluation/d2689ab6639bc09f/primary`
