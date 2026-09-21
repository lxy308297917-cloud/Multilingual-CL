# SW D2 同数据方法对照

仅列出固定 seed42 任务中通过 DONE、产物哈希和模型身份核验的成绩。D2 旧 FFT 是历史参考；同优化器密集 FFT 与两组机制对照为本轨道严格对照。
MMLU/G-MMLU 延期。PPL/NLL 仅作诊断，不进入 EN3/Target3 或能力结论。

待完成：{"edge_projected_rows": "训练未完成", "edge_rows_no_projection": "训练未完成", "fft_fp32_matched": "训练未完成"}

## 下游原始分数与相对 Base 变化

|方法|Benchmark|原始分数|相对 Base|相对历史 D2|
|---|---|---:|---:|---:|
|Base|SW Belebele|37.22|+0.00|-7.67|
|Base|SW XL-Sum|23.94|+0.00|-6.25|
|Base|EN→SW|17.44|+0.00|-18.90|
|Base|SW→EN|21.01|+0.00|-18.64|
|Base|EN Belebele|81.78|+0.00|+3.44|
|Base|EN XL-Sum|21.89|+0.00|+0.26|
|Base|IFEval prompt strict|42.88|+0.00|+22.74|
|Base|IFEval instruction strict|51.80|+0.00|+22.06|
|Base|GSM8K strict 5-shot|31.31|+0.00|+17.82|
|Base|GSM8K flexible 5-shot|54.74|+0.00|+17.59|
|D2_历史FFT|SW Belebele|44.89|+7.67|+0.00|
|D2_历史FFT|SW XL-Sum|30.19|+6.25|+0.00|
|D2_历史FFT|EN→SW|36.34|+18.90|+0.00|
|D2_历史FFT|SW→EN|39.64|+18.64|+0.00|
|D2_历史FFT|EN Belebele|78.33|-3.44|+0.00|
|D2_历史FFT|EN XL-Sum|21.63|-0.26|+0.00|
|D2_历史FFT|IFEval prompt strict|20.15|-22.74|+0.00|
|D2_历史FFT|IFEval instruction strict|29.74|-22.06|+0.00|
|D2_历史FFT|GSM8K strict 5-shot|13.50|-17.82|+0.00|
|D2_历史FFT|GSM8K flexible 5-shot|37.15|-17.59|+0.00|
|element_protected|SW Belebele|45.00|+7.78|+0.11|
|element_protected|SW XL-Sum|30.55|+6.61|+0.36|
|element_protected|EN→SW|36.92|+19.48|+0.58|
|element_protected|SW→EN|43.58|+22.57|+3.93|
|element_protected|EN Belebele|80.22|-1.56|+1.89|
|element_protected|EN XL-Sum|21.54|-0.35|-0.09|
|element_protected|IFEval prompt strict|25.69|-17.19|+5.55|
|element_protected|IFEval instruction strict|38.01|-13.79|+8.27|
|element_protected|GSM8K strict 5-shot|19.94|-11.37|+6.44|
|element_protected|GSM8K flexible 5-shot|41.47|-13.27|+4.32|
|element_unprotected|SW Belebele|43.33|+6.11|-1.56|
|element_unprotected|SW XL-Sum|30.56|+6.63|+0.37|
|element_unprotected|EN→SW|35.02|+17.58|-1.32|
|element_unprotected|SW→EN|39.29|+18.28|-0.36|
|element_unprotected|EN Belebele|76.33|-5.44|-2.00|
|element_unprotected|EN XL-Sum|21.93|+0.04|+0.30|
|element_unprotected|IFEval prompt strict|21.07|-21.81|+0.92|
|element_unprotected|IFEval instruction strict|30.34|-21.46|+0.60|
|element_unprotected|GSM8K strict 5-shot|13.42|-17.89|-0.08|
|element_unprotected|GSM8K flexible 5-shot|30.10|-24.64|-7.05|

## SW 阅读逐文章配对差值

|比较|差值 pp|95% CI pp|单侧配对检验 p|
|---|---:|---:|---:|
|element_protected − element_unprotected|+1.67|[-1.22, +4.44]|0.1435|
|element_protected − D2_历史FFT|+0.11|[-2.77, +2.99]|0.4956|
|element_unprotected − D2_历史FFT|-1.56|[-4.25, +1.12]|0.8852|
|D2_历史FFT − Base|+7.67|[+4.23, +11.12]|0.0001|
|element_protected − Base|+7.78|[+4.08, +11.54]|0.0001|
|element_unprotected − Base|+6.11|[+2.68, +9.56]|0.0005|

该区间按文章分组、配对重抽样10000次；只反映固定模型与固定测试集的抽样不确定性，不代表跨训练 seed 稳定性。

## EN3 与 Target3（相对 Base 百分比等权平均）

|方法|EN3|Target3|
|---|---:|---:|
|Base|+0.00%|+0.00%|
|D2_历史FFT|+27.78%|+51.70%|
|element_protected|+34.65%|+53.42%|
|element_unprotected|+26.86%|+48.30%|

## PPL/NLL 诊断

|方法|EN PPL|SW PPL|EN NLL|SW NLL|
|---|---:|---:|---:|---:|
|Base|14.280|69.588|2.659|4.243|
|D2_历史FFT|16.392|13.696|2.797|2.617|
|element_protected|14.506|14.301|2.675|2.660|
|element_unprotected|15.413|14.245|2.735|2.656|

## 训练成本与范围

最终权重只在预算终点比较。当前完成模型的训练成本、输入与监督 token 记录于 training_costs.json。
目标是判断 SW 阅读收益与 EN/通用保持之间的取舍；阅读 CI 为正也不能覆盖其他任务的下降。
