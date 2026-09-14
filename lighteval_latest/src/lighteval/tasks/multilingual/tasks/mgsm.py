"""
name:
Mgsm

dataset:
juletxara/mgsm

abstract:
Mgsm multilingual benchmark.

languages:
bengali, chinese, english, french, german, japanese, russian, spanish, swahili,
telugu, thai

tags:
math, multilingual, reasoning

paper:
"""

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


def _direct_answer_instruction(language: Language) -> str:
    """
    Direct-answer instruction for MGSM.
    Goal: make instruct models output only the final number,
    so exact_match_*_full can work.
    """
    if language == Language.CHINESE:
        return "请只输出最终数字，不要解释。"
    if language == Language.JAPANESE:
        return "最終的な数値だけを出力してください。説明は不要です。"
    if language == Language.SPANISH:
        return "Responde solo con el número final, sin explicación."
    if language == Language.FRENCH:
        return "Répondez uniquement avec le nombre final, sans explication."
    if language == Language.GERMAN:
        return "Antworte nur mit der endgültigen Zahl, ohne Erklärung."
    if language == Language.RUSSIAN:
        return "Ответьте только итоговым числом, без объяснений."
    if language == Language.SWAHILI:
        return "Jibu kwa nambari ya mwisho tu, bila maelezo."
    if language == Language.BENGALI:
        return "শুধু চূড়ান্ত সংখ্যাটি লিখুন, কোনো ব্যাখ্যা নয়।"
    if language == Language.TELUGU:
        return "వివరణ లేకుండా తుది సంఖ్య మాత్రమే ఇవ్వండి."
    if language == Language.THAI:
        return "ตอบเฉพาะตัวเลขสุดท้ายเท่านั้น ไม่ต้องอธิบาย"
    return "Answer with only the final number, without explanation."


def _make_mgsm_line(language: Language, line):
    question = line["question"].strip()
    instruction = _direct_answer_instruction(language)

    return {
        "question": f"{question}\n{instruction}",
        "choices": [str(line["answer_number"])],
    }


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mgsm_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line, language=language: _make_mgsm_line(language, line),
        ),
        hf_repo="juletxara/mgsm",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=64,
        metrics=[
            MultilingualQuasiExactMatchMetric(language, "full"),
        ],
        stop_sequence=("\n",),
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
