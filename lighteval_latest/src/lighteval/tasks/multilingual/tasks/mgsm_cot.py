# -*- coding: utf-8 -*-
"""
MGSM-CoT with extracted final-answer metric.

This task allows the model to generate a reasoning process and then extracts
the final numerical answer from the response. The score will appear directly
in the LightEval result table and saved results JSON.

Task names:
  mgsm_eng_cot
  mgsm_spa_cot
  mgsm_fra_cot
  mgsm_deu_cot
  mgsm_rus_cot
  mgsm_zho_cot
  mgsm_jpn_cot
  mgsm_swa_cot
  ...
"""

import re
import numpy as np
from langcodes import standardize_tag

from lighteval.metrics.utils.metric_utils import SampleLevelComputation, SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


def _normalize_num(x):
    x = str(x).strip()
    x = x.replace(",", "")
    x = x.replace("$", "")
    x = x.replace(" ", "")

    # 40.00 -> 40
    try:
        if re.fullmatch(r"-?\d+\.0+", x):
            x = str(int(float(x)))
    except Exception:
        pass

    return x


def _extract_answer(text):
    text = str(text)

    # 优先抽取明确 final answer 后面的数字
    patterns = [
        r"Final answer\s*[:：]\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"final answer\s*[:：]\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"Therefore,\s*the\s*final\s*answer\s*is\s*[:：]?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"the\s*final\s*answer\s*is\s*[:：]?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"Answer\s*[:：]\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"答案\s*[:：]?\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return _normalize_num(m.group(1))

    # 兜底：取最后一个数字
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return _normalize_num(nums[-1])

    return ""


def _get_final_text(model_response):
    # LightEval runtime object usually has final_text
    if hasattr(model_response, "final_text"):
        txt = model_response.final_text
        if isinstance(txt, (list, tuple)):
            return str(txt[0]) if txt else ""
        return str(txt)

    # fallback for dict-like object
    if isinstance(model_response, dict):
        txt = model_response.get("text", "")
        if hasattr(txt, "tolist"):
            txt = txt.tolist()
        if isinstance(txt, (list, tuple)):
            return str(txt[0]) if txt else ""
        return str(txt)

    return str(model_response)


def _get_gold(doc):
    gold_index = doc.gold_index

    if hasattr(gold_index, "tolist"):
        gold_index = gold_index.tolist()

    if isinstance(gold_index, (list, tuple)):
        gold_index = gold_index[0]

    gold_index = int(gold_index)
    return _normalize_num(doc.choices[gold_index])


class MGSMCotExtractMetric(SampleLevelComputation):
    def compute(self, doc, model_response, **kwargs):
        pred = _extract_answer(_get_final_text(model_response))
        gold = _get_gold(doc)
        return float(pred == gold)


mgsm_cot_extract_metric = SampleLevelMetric(
    metric_name="mgsm_cot_extract",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=MGSMCotExtractMetric(),
    corpus_level_fn=np.mean,
)


def _cot_instruction(language: Language) -> str:
    if language == Language.CHINESE:
        return "请一步一步推理。最后一行必须写：Final answer: <数字>"
    if language == Language.JAPANESE:
        return "ステップごとに考えてください。最後の行は必ず次の形式にしてください：Final answer: <number>"
    if language == Language.SPANISH:
        return "Resuelve paso a paso. La última línea debe ser: Final answer: <number>"
    if language == Language.FRENCH:
        return "Résolvez étape par étape. La dernière ligne doit être : Final answer: <number>"
    if language == Language.GERMAN:
        return "Löse die Aufgabe Schritt für Schritt. Die letzte Zeile muss lauten: Final answer: <number>"
    if language == Language.RUSSIAN:
        return "Решите задачу пошагово. Последняя строка должна быть: Final answer: <number>"
    if language == Language.SWAHILI:
        return "Tatua hatua kwa hatua. Mstari wa mwisho uwe: Final answer: <number>"
    if language == Language.BENGALI:
        return "ধাপে ধাপে সমাধান করুন। শেষ লাইনে অবশ্যই লিখুন: Final answer: <number>"
    if language == Language.TELUGU:
        return "దశలవారీగా పరిష్కరించండి. చివరి పంక్తి ఇలా ఉండాలి: Final answer: <number>"
    if language == Language.THAI:
        return "จงแก้ทีละขั้นตอน บรรทัดสุดท้ายต้องเป็น: Final answer: <number>"
    return "Solve the problem step by step. The last line must be exactly: Final answer: <number>"


def _make_line(language: Language, line):
    question = line["question"].strip()
    instruction = _cot_instruction(language)

    return {
        "question": f"{question}\n{instruction}",
        "choices": [str(line["answer_number"])],
    }


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mgsm_{language.value}_cot",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line, language=language: _make_line(language, line),
        ),
        hf_repo="juletxara/mgsm",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=512,
        metrics=[
            mgsm_cot_extract_metric,
        ],
        stop_sequence=None,
    )
    for language in [
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.RUSSIAN,
        Language.CHINESE,
        Language.JAPANESE,
        Language.THAI,
        Language.SWAHILI,
        Language.BENGALI,
        Language.TELUGU,
    ]
]
