# Igbo 首尾层与分散层：两种训练顺序

两轮的 Base、训练数据、矩阵名额、训练超参数及下游评测设置相同；训练顺序分别为 seed42 和 seed43。每轮首尾层与分散层共同使用该轮顺序。密集 FFT 是历史参考，不是同顺序稀疏对照。

|下游任务|seed42 首尾−分散|seed43 首尾−分散|方向重复|
|---|---:|---:|:---:|
|belebele_ig|+0.5556|+3.1111|是|
|sum_ig|+3.9795|+5.4245|是|
|mt_en2ig|+2.3467|+3.0930|是|
|mt_ig2en|+3.7703|+1.9087|是|
|belebele_en|-0.5556|-0.1111|是|
|sum_en|-0.9932|-0.4889|是|
|ifeval_prompt_strict|-1.1091|-1.1091|是|
|ifeval_instruction_strict|-0.7194|-2.2782|是|
|gsm8k_strict|-4.7005|-15.6937|是|
|mmlu_en|-0.5263|-0.1350|是|
|gmmlu_ig|+2.1172|+2.3948|是|

主要目标为 IG 摘要和双向翻译；两轮首尾层均领先的主要任务：sum_ig、mt_en2ig、mt_ig2en。
只有两种训练顺序，不能把方向重复当作统计显著性；英语阅读、IFEval、GSM8K 的代价须逐项检查。PPL/NLL 只作诊断。

|训练|GPU 小时|峰值显存 GiB|
|---|---:|---:|
|seed42 edge|1.081|6.58|
|seed42 spread|1.103|6.58|
|seed43 edge|1.090|6.58|
|seed43 spread|1.120|6.58|

seed42 样本顺序 SHA256：`0625132e7fab83b0a5575e114d7d0e227e139f96c2a3facfdd1e2cfd57b660b1`。
seed43 样本顺序 SHA256：`bf5a3aaa85bd18293fe2e1491f43ac8ec2db1abc2500bf6c80bd5b5c217c1828`。
正式原始结果目录详见两轮各自的 `REPORT_DONE.json`。
