# SW 数据配方实验结果

当前未完成项：{"C": ["training_not_completed"], "B": ["training_not_completed"], "A": ["training_not_completed"]}

## 下游原始分数

|模型|指标|原始分数|相对Base绝对变化|
|---|---|---:|---:|
|Base|sum_sw|23.9353|+0.0000|
|Base|sum_en|21.8881|+0.0000|
|Base|mt_en2sw|17.4389|+0.0000|
|Base|mt_sw2en|21.0065|+0.0000|
|Base|belebele_sw|37.2222|+0.0000|
|Base|belebele_en|81.7778|+0.0000|
|Base|ifeval_prompt_strict|42.8835|+0.0000|
|Base|ifeval_instruction_strict|51.7986|+0.0000|
|Base|gsm8k_strict|31.3116|+0.0000|
|Base|gsm8k_flexible|54.7384|+0.0000|
|D|sum_sw|31.8634|+7.9281|
|D|sum_en|21.1426|-0.7455|
|D|mt_en2sw|19.3163|+1.8773|
|D|mt_sw2en|14.8719|-6.1346|
|D|belebele_sw|45.7778|+8.5556|
|D|belebele_en|79.2222|-2.5556|
|D|ifeval_prompt_strict|27.1719|-15.7116|
|D|ifeval_instruction_strict|38.3693|-13.4293|
|D|gsm8k_strict|0.7582|-30.5534|
|D|gsm8k_flexible|31.9939|-22.7445|

## 配对结论

{}

## EN3/Target3

{
  "Base": {
    "EN3": 0.0,
    "Target3": 0.0
  },
  "D": {
    "EN3": -11.91147082139851,
    "Target3": 22.291124618360936
  },
  "C": {
    "EN3": null,
    "Target3": null
  },
  "B": {
    "EN3": null,
    "Target3": null
  },
  "A": {
    "EN3": null,
    "Target3": null
  }
}

D仅历史参考。C摘要为同分布监督泛化，不可单独认定整体语言能力提高。PPL/NLL单列于CSV，未进入能力综合。

## 数据与训练状态

- C：未完成训练；数据预算已准备
{"length": 2048, "input_tokens": 15361733, "statistics": {"reading": {"target_input_tokens": 1435152, "actual_input_tokens": 1435152, "supervised_tokens": 46866, "presentations": 5784, "unique_samples": 1928, "mean_uses_selected": 3.0, "max_repetitions": 3}, "translation_en2sw": {"target_input_tokens": 1536000, "actual_input_tokens": 1535999, "supervised_tokens": 498197, "presentations": 16285, "unique_samples": 16285, "mean_uses_selected": 1.0, "max_repetitions": 1}, "translation_sw2en": {"target_input_tokens": 1536000, "actual_input_tokens": 1535988, "supervised_tokens": 288538, "presentations": 16278, "unique_samples": 16278, "mean_uses_selected": 1.0, "max_repetitions": 1}, "summary": {"target_input_tokens": 2304000, "actual_input_tokens": 2305678, "supervised_tokens": 170796, "presentations": 2844, "unique_samples": 2844, "mean_uses_selected": 1.0, "max_repetitions": 1}, "aya/Aya-Dataset": {"target_input_tokens": 17737, "actual_input_tokens": 17725, "supervised_tokens": 11383, "presentations": 113, "unique_samples": 113, "mean_uses_selected": 1.0, "max_repetitions": 1}, "aya/Dolly-v2 (T)": {"target_input_tokens": 1372620, "actual_input_tokens": 1372618, "supervised_tokens": 600576, "presentations": 4392, "unique_samples": 4392, "mean_uses_selected": 1.0, "max_repetitions": 1}, "aya/Flan-Coqa (T)": {"target_input_tokens": 2291193, "actual_input_tokens": 2291188, "supervised_tokens": 399016, "presentations": 1924, "unique_samples": 1924, "mean_uses_selected": 1.0, "max_repetitions": 1}, "aya/SODA-inst (T)": {"target_input_tokens": 2343510, "actual_input_tokens": 2343492, "supervised_tokens": 1330306, "presentations": 15039, "unique_samples": 15039, "mean_uses_selected": 1.0, "max_repetitions": 1}, "aya/Wiki-split-inst (T)": {"target_input_tokens": 2523788, "actual_input_tokens": 2523893, "supervised_tokens": 1286410, "presentations": 15033, "unique_samples": 15033, "mean_uses_selected": 1.0, "max_repetitions": 1}}}
- B：未完成训练；数据预算已准备
{"length": 2048, "input_tokens": 15361342, "statistics": {"aya/Aya-Dataset": {"target_input_tokens": 31869, "actual_input_tokens": 31859, "supervised_tokens": 21507, "presentations": 196, "unique_samples": 196, "mean_uses_selected": 1.0, "max_repetitions": 1}, "aya/Dolly-v2 (T)": {"target_input_tokens": 2466233, "actual_input_tokens": 2466228, "supervised_tokens": 1064498, "presentations": 7885, "unique_samples": 7885, "mean_uses_selected": 1.0, "max_repetitions": 1}, "aya/Flan-Coqa (T)": {"target_input_tokens": 4116664, "actual_input_tokens": 4117876, "supervised_tokens": 717292, "presentations": 3444, "unique_samples": 3444, "mean_uses_selected": 1.0, "max_repetitions": 1}, "aya/SODA-inst (T)": {"target_input_tokens": 4210663, "actual_input_tokens": 4210812, "supervised_tokens": 2388103, "presentations": 27066, "unique_samples": 27066, "mean_uses_selected": 1.0, "max_repetitions": 1}, "aya/Wiki-split-inst (T)": {"target_input_tokens": 4534571, "actual_input_tokens": 4534567, "supervised_tokens": 2313181, "presentations": 27011, "unique_samples": 27011, "mean_uses_selected": 1.0, "max_repetitions": 1}}}
- A：未完成训练；数据预算已准备
{"length": 2048, "input_tokens": 15360000, "statistics": {"cpt": {"target_input_tokens": 15360000, "actual_input_tokens": 15360000, "supervised_tokens": 15352500, "presentations": 7500, "unique_samples": 7500, "max_repetitions": 1}}}

## 三项判读

- C：阅读是否提升、保持是否达标、是否只有摘要收益均待完整评测。
- B：阅读是否提升、保持是否达标、是否只有摘要收益均待完整评测。
- A：阅读是否提升、保持是否达标、是否只有摘要收益均待完整评测。

## 输出诊断

{
  "Base/sum_sw": {
    "samples": 500,
    "mean_output_tokens": 115.792,
    "empty_output_fraction": 0.0,
    "generation_limit_fraction": 0.82,
    "input_truncated_fraction": 1.0,
    "format_note": "free generation has no unique required format; IFEval/GSM strict-vs-flexible errors reported separately"
  },
  "Base/sum_en": {
    "samples": 500,
    "mean_output_tokens": 106.766,
    "empty_output_fraction": 0.0,
    "generation_limit_fraction": 0.222,
    "input_truncated_fraction": 1.0,
    "format_note": "free generation has no unique required format; IFEval/GSM strict-vs-flexible errors reported separately"
  },
  "Base/mt_en2sw": {
    "samples": 500,
    "mean_output_tokens": 60.628,
    "empty_output_fraction": 0.0,
    "generation_limit_fraction": 0.078,
    "input_truncated_fraction": 1.0,
    "format_note": "free generation has no unique required format; IFEval/GSM strict-vs-flexible errors reported separately"
  },
  "Base/mt_sw2en": {
    "samples": 500,
    "mean_output_tokens": 26.256,
    "empty_output_fraction": 0.0,
    "generation_limit_fraction": 0.004,
    "input_truncated_fraction": 1.0,
    "format_note": "free generation has no unique required format; IFEval/GSM strict-vs-flexible errors reported separately"
  },
  "D/sum_sw": {
    "samples": 500,
    "mean_output_tokens": 117.114,
    "empty_output_fraction": 0.0,
    "generation_limit_fraction": 0.82,
    "input_truncated_fraction": 1.0,
    "format_note": "free generation has no unique required format; IFEval/GSM strict-vs-flexible errors reported separately"
  },
  "D/sum_en": {
    "samples": 500,
    "mean_output_tokens": 103.764,
    "empty_output_fraction": 0.0,
    "generation_limit_fraction": 0.326,
    "input_truncated_fraction": 1.0,
    "format_note": "free generation has no unique required format; IFEval/GSM strict-vs-flexible errors reported separately"
  },
  "D/mt_en2sw": {
    "samples": 500,
    "mean_output_tokens": 61.98,
    "empty_output_fraction": 0.0,
    "generation_limit_fraction": 0.07,
    "input_truncated_fraction": 1.0,
    "format_note": "free generation has no unique required format; IFEval/GSM strict-vs-flexible errors reported separately"
  },
  "D/mt_sw2en": {
    "samples": 500,
    "mean_output_tokens": 74.284,
    "empty_output_fraction": 0.0,
    "generation_limit_fraction": 0.106,
    "input_truncated_fraction": 1.0,
    "format_note": "free generation has no unique required format; IFEval/GSM strict-vs-flexible errors reported separately"
  }
}
