"""Project-local LightEval tasks for multilingual continual pretraining.

This is the only custom-task registry that experiment launchers should pass to
LightEval.  It contains pinned MGSM/MGSM-CoT, SIB-200, XL-Sum, and FLORES MT
definitions so the installed LightEval checkout is never edited.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import os
import re

from langcodes import standardize_tag
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from transformers import AutoTokenizer
import lighteval.metrics.metrics_sample as _metrics_sample

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric, MultilingualQuasiExactMatchMetric
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.metrics.utils.metric_utils import SampleLevelComputation, SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.multilingual.tasks import (
    afri_mgsm as native_afri_mgsm, afri_mmlu as native_afri_mmlu,
    afri_xnli as native_afri_xnli, belebele as native_belebele,
    global_mmlu as native_global_mmlu, openai_mmlu as native_openai_mmlu,
    xcopa as native_xcopa, xcsqa as native_xcsqa, xnli as native_xnli,
)
from lighteval.tasks.tasks import (
    arc as native_arc,
    gsm8k as native_gsm8k,
    hellaswag as native_hellaswag,
    mmlu as native_mmlu,
    truthfulqa as native_truthfulqa,
)
from lighteval.tasks.tasks.ifeval import main as native_ifeval
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS, TranslationLiterals
from lighteval.utils.language import Language, manage_duplicate_language_codes
from taskmix_multilingual_tasks import CHRFPP_CORPUS, TASKS_TABLE as TASKMIX_TASKS


# LightEval 0.13.1.dev0 registers Amharic as a language but leaves all of its
# prompt literals unset. Native multilingual MCQ formatters therefore raise
# before producing the first document. Keep the compatibility fix here and
# never patch the installed LightEval checkout.
_AMHARIC_LITERALS = TRANSLATION_LITERALS[Language.AMHARIC]
for _name, _value in {
    "question_word": "ጥያቄ", "answer": "መልስ", "confirmation_word": "ትክክል",
    "yes": "አዎ", "no": "አይ", "also": "እንዲሁም", "cause_word": "ምክንያቱም",
    "effect_word": "ስለዚህ", "or_word": "ወይም", "and_word": "እና",
    "true": "እውነት", "false": "ሐሰት", "neither": "ሁለቱም አይደሉም",
    "full_stop": "።", "comma": "፣", "question_mark": "?",
    "exclamation_mark": "!", "colon": ":",
}.items():
    # Native formatter closures captured this object during module import.
    # Mutating it fixes existing tasks; replacing the dict entry does not.
    setattr(_AMHARIC_LITERALS, _name, _value)

# Yoruba is registered in the same LightEval revision with an empty literal
# object. Mutate that captured object as above so native Belebele/Afri tasks
# can build their prompts without changing the installed LightEval checkout.
_YORUBA_LITERALS = TRANSLATION_LITERALS[Language.YORUBA]
for _name, _value in {
    "question_word": "Ìbéèrè", "answer": "Ìdáhùn", "confirmation_word": "Ó tọ́",
    "yes": "Bẹ́ẹ̀ni", "no": "Rárá", "also": "Pẹ̀lúpẹ̀lú", "cause_word": "Nítorí",
    "effect_word": "Nítorí náà", "or_word": "Tàbí", "and_word": "Àti",
    "true": "Òótọ́", "false": "Irọ́", "neither": "Kò sí èyíkéyìí",
    "full_stop": ".", "comma": ",", "question_mark": "?",
    "exclamation_mark": "!", "colon": ":",
}.items():
    setattr(_YORUBA_LITERALS, _name, _value)


# LightEval 0.13.1.dev0 truncates output_tokens along the choice axis by the
# continuation-token length in its single-process Accelerate gather path.
# Recompute only the token-normalization denominator from the frozen tokenizer.
_ORIGINAL_NORMALIZE_LOG_PROBS = _metrics_sample.normalize_log_probs
_TOKEN_NORM_TOKENIZER = None

def _project_normalize_log_probs(normalization, choices_logprob, unconditioned_logprob, choices_text, choices_tokens):
    global _TOKEN_NORM_TOKENIZER
    if isinstance(normalization, LogProbTokenNorm):
        if _TOKEN_NORM_TOKENIZER is None:
            tokenizer_path = os.environ["MULTILINGUAL_CL_TOKENIZER"]
            _TOKEN_NORM_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        choices_tokens = [
            _TOKEN_NORM_TOKENIZER.encode(str(choice), add_special_tokens=False)
            for choice in choices_text
        ]
    return _ORIGINAL_NORMALIZE_LOG_PROBS(
        normalization, choices_logprob, unconditioned_logprob, choices_text, choices_tokens
    )

_metrics_sample.normalize_log_probs = _project_normalize_log_probs


# MGSM -----------------------------------------------------------------------

MGSM_LANGUAGES = [
    Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.GERMAN,
    Language.RUSSIAN, Language.CHINESE, Language.JAPANESE, Language.THAI,
    Language.SWAHILI, Language.BENGALI, Language.TELUGU,
]


def _mgsm_adapter(line):
    return {"question": line["question"], "choices": [str(line["answer_number"])]}


MGSM_OFFICIAL_TASKS = [
    LightevalTaskConfig(
        name=f"mgsm_official_{language.value}",
        prompt_function=get_qa_prompt_function(language, _mgsm_adapter),
        hf_repo="juletxara/mgsm",
        hf_subset=standardize_tag(language.value),
        hf_revision="b2f13d426afe3be8d69a7e739b36724db8b66bbc",
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=25,
        metrics=[MultilingualQuasiExactMatchMetric(language, "full")],
        stop_sequence=("\n",),
    )
    for language in MGSM_LANGUAGES
]

NUMBER = r"[-+−]?\d(?:[\d,\u202f ]*\d)?(?:\.\d+)?"
ANSWER_PATTERNS = (
    rf"final\s+answer\s*[:：=]?\s*({NUMBER})",
    rf"answer\s*[:：=]\s*({NUMBER})",
    rf"答案\s*[:：=]?\s*({NUMBER})",
    rf"答え\s*[:：=]?\s*({NUMBER})",
    rf"ответ\s*[:：=]?\s*({NUMBER})",
    rf"respuesta\s*[:：=]?\s*({NUMBER})",
)


def normalize_number(value: object) -> str:
    text = str(value).strip().replace("−", "-")
    text = text.replace(",", "").replace("\u202f", "").replace(" ", "")
    text = text.replace("$", "").replace("€", "").replace("£", "")
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if number == number.to_integral():
        return str(number.quantize(Decimal("1")))
    return format(number.normalize(), "f")


def extract_final_number(text: object) -> str:
    response = str(text)
    for pattern in ANSWER_PATTERNS:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return normalize_number(match.group(1))
    matches = re.findall(NUMBER, response)
    return normalize_number(matches[-1]) if matches else ""


def _response_text(response) -> str:
    value = getattr(response, "final_text", response)
    if isinstance(value, dict):
        value = value.get("text", "")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip()


class MGSMCotExtractMetric(SampleLevelComputation):
    def compute(self, doc, model_response, **kwargs):
        gold_index = doc.gold_index.tolist() if hasattr(doc.gold_index, "tolist") else doc.gold_index
        if isinstance(gold_index, (list, tuple)):
            gold_index = gold_index[0]
        gold = normalize_number(doc.choices[int(gold_index)])
        return float(extract_final_number(_response_text(model_response)) == gold)


MGSM_COT_METRIC = SampleLevelMetric(
    metric_name="mgsm_cot_extract",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=MGSMCotExtractMetric(),
    corpus_level_fn=np.mean,
)


def _mgsm_instruction(language: Language) -> str:
    prompts = {
        Language.CHINESE: "请逐步推理。最后一行必须写：Final answer: <数字>",
        Language.GERMAN: "Löse schrittweise. Die letzte Zeile muss lauten: Final answer: <number>",
        Language.SPANISH: "Resuelve paso a paso. La última línea debe ser: Final answer: <number>",
        Language.RUSSIAN: "Решите пошагово. Последняя строка: Final answer: <number>",
        Language.SWAHILI: "Tatua hatua kwa hatua. Mstari wa mwisho: Final answer: <number>",
    }
    return prompts.get(language, "Solve step by step. End with: Final answer: <number>")


def _mgsm_cot_adapter(language: Language, line):
    return {
        "question": f"{line['question'].strip()}\n{_mgsm_instruction(language)}",
        "choices": [str(line["answer_number"])],
    }


MGSM_COT_TASKS = [
    LightevalTaskConfig(
        name=f"mgsm_cot_{language.value}",
        prompt_function=get_qa_prompt_function(
            language, lambda line, language=language: _mgsm_cot_adapter(language, line)
        ),
        hf_repo="juletxara/mgsm",
        hf_subset=standardize_tag(language.value),
        hf_revision="b2f13d426afe3be8d69a7e739b36724db8b66bbc",
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=512,
        metrics=[MGSM_COT_METRIC],
        stop_sequence=None,
    )
    for language in MGSM_LANGUAGES
]


# SIB-200 --------------------------------------------------------------------

SIB_CHOICES = ["geography", "science/technology", "entertainment", "travel", "sports", "health", "politics"]
SIB_LANGUAGES = {
    "de": ("deu_Latn", Language.GERMAN, "Welches Thema behandelt der folgende Text?\n"),
    "es": ("spa_Latn", Language.SPANISH, "¿De qué tema trata el siguiente texto?\n"),
    "ru": ("rus_Cyrl", Language.RUSSIAN, "Какой теме посвящён следующий текст?\n"),
    "zh": ("zho_Hans", Language.CHINESE, "下面的文本属于哪个主题？\n"),
    "sw": ("swh_Latn", Language.SWAHILI, "Maandishi yafuatayo yanahusu mada gani?\n"),
    "am": ("amh_Ethi", Language.AMHARIC, "የሚከተለው ጽሑፍ ስለ የትኛው ርዕስ ነው?\n"),
    "yo": ("yor_Latn", Language.YORUBA, "Kókó wo ni ọ̀rọ̀ inú àpilẹ̀kọ yìí dá lé?\n"),
}


def _sib_adapter(instruction):
    return lambda line: {
        "question": instruction + line["text"],
        "choices": SIB_CHOICES,
        "gold_idx": SIB_CHOICES.index(line["category"]),
    }


SIB200_TASKS = []
for short, (subset, language, instruction) in SIB_LANGUAGES.items():
    for formulation in (MCFFormulation(), CFFormulation()):
        SIB200_TASKS.append(
            LightevalTaskConfig(
                name=f"sib200_new_{short}_{formulation.name.lower()}",
                prompt_function=get_mcq_prompt_function(language, _sib_adapter(instruction), formulation=formulation),
                hf_repo="Davlan/sib200",
                hf_subset=subset,
                hf_revision="38977a667f6fc264d5c26ec57a01e16db040b358",
                evaluation_splits=("test",),
                hf_avail_splits=["train", "validation", "test"],
                metrics=get_metrics_for_formulation(
                    formulation,
                    [
                        LogLikelihoodAccMetric(normalization=None),
                        LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                        LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                        LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
                    ],
                ),
                generation_size=-1,
            )
        )


# XL-Sum ---------------------------------------------------------------------

XLSUM_ROOT = os.environ.get("MULTILINGUAL_CL_XLSUM_ROOT", "/root/autodl-tmp/multilingual_cl_v2/eval_data/xlsum")
XLSUM_LANGUAGES = {
    "es": ("Resume el siguiente texto en español:", "Resumen:"),
    "ar": ("لخّص النص التالي بإيجاز باللغة العربية:", "الملخص:"),
    "ru": ("Кратко изложите следующий текст на русском языке:", "Краткое содержание:"),
    "zh": ("请用中文概括以下文本：", "摘要："),
    "sw": ("Fupisha maandishi yafuatayo kwa Kiswahili:", "Muhtasari:"),
    "bn": ("নিচের লেখাটি বাংলায় সংক্ষেপ করুন:", "সারাংশ:"),
    "te": ("క్రింది పాఠ్యాన్ని తెలుగులో సంక్షిప్తీకరించండి:", "సారాంశం:"),
    "am": ("የሚከተለውን ጽሑፍ በአማርኛ በአጭሩ አጠቃልል፦", "ማጠቃለያ፦"),
    "yo": ("Ṣe àkótán ọrọ̀ yìí ní èdè Yorùbá:", "Àkótán:"),
}


class ChrFPlusPlusMetric(SampleLevelComputation):
    def compute(self, doc, model_response, **kwargs):
        gold_index = doc.gold_index.tolist() if hasattr(doc.gold_index, "tolist") else doc.gold_index
        if isinstance(gold_index, (list, tuple)):
            gold_index = gold_index[0]
        gold = str(doc.choices[int(gold_index)])
        return float(sacrebleu.sentence_chrf(_response_text(model_response), [gold], word_order=2).score)


CHRFPP = SampleLevelMetric(
    metric_name="chrfpp_sample", higher_is_better=True, category=SamplingMethod.GENERATIVE,
    sample_level_fn=ChrFPlusPlusMetric(), corpus_level_fn=np.mean,
)


class UnicodeWordTokenizer:
    def tokenize(self, text):
        return re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE)


class UnicodeRougeMetric(SampleLevelComputation):
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.scorer = rouge_scorer.RougeScorer([metric_name], tokenizer=UnicodeWordTokenizer())

    def __str__(self) -> str:
        # LightEval hashes task configs through string representations. The
        # default RougeScorer repr contains a process-specific memory address,
        # so expose only the semantic configuration here.
        return f"UnicodeRougeMetric(metric_name={self.metric_name})"

    def compute(self, doc, model_response, **kwargs):
        gold_index = doc.gold_index.tolist() if hasattr(doc.gold_index, "tolist") else doc.gold_index
        if isinstance(gold_index, (list, tuple)):
            gold_index = gold_index[0]
        gold = str(doc.choices[int(gold_index)])
        score = self.scorer.score(gold, _response_text(model_response))[self.metric_name].fmeasure
        return float(score * 100.0)


XLSUM_UNICODE_ROUGE = [
    SampleLevelMetric(
        metric_name=f"{name}_unicode_pct", higher_is_better=True, category=SamplingMethod.GENERATIVE,
        sample_level_fn=UnicodeRougeMetric(name), corpus_level_fn=np.mean,
    )
    for name in ("rouge1", "rouge2", "rougeL")
]


def _xlsum_prompt(instruction: str, anchor: str):
    def prompt(line, task_name: str):
        return Doc(
            task_name=task_name,
            query=f"{instruction}\n{line['text'].strip()}\n{anchor}",
            choices=[str(line["summary"]).strip()], gold_index=0, instruction=instruction,
            specific={"source_id": str(line.get("id", ""))},
        )
    return prompt


XLSUM_TASKS = [
    LightevalTaskConfig(
        name=f"xlsum_cl_{short}", prompt_function=_xlsum_prompt(instruction, anchor),
        hf_repo=os.path.join(XLSUM_ROOT, short), hf_subset=None,
        evaluation_splits=("test",), hf_avail_splits=["test"], metrics=[CHRFPP, *XLSUM_UNICODE_ROUGE],
        generation_size=256, stop_sequence=["\n\n"],
    )
    for short, (instruction, anchor) in XLSUM_LANGUAGES.items()
]


# FLORES-200 MT --------------------------------------------------------------

FLORES_ROOT = os.environ.get("MULTILINGUAL_CL_FLORES_ROOT", "/root/autodl-tmp/multilingual_cl_v2/eval_data/flores200")
FLORES_LANGUAGES = (
    "deu_Latn", "spa_Latn", "rus_Cyrl", "arb_Arab", "zho_Hans", "swh_Latn",
    "ben_Beng", "tel_Telu", "amh_Ethi", "yor_Latn",
)


def _flores_adapter(line):
    return {"source_text": line["source_text"], "target_text": line["target_text"]}


def _flores_task(source: str, target: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"flores_cl:{source}-{target}",
        prompt_function=get_translation_prompt_function(
            source_language=Language(manage_duplicate_language_codes(source.split("_")[0])),
            target_language=Language(manage_duplicate_language_codes(target.split("_")[0])),
            adapter=_flores_adapter, formulation=CFFormulation(),
        ),
        hf_repo=os.path.join(FLORES_ROOT, f"{source}-{target}"), hf_subset=None,
        hf_avail_splits=["validation", "test"], evaluation_splits=["test"],
        few_shots_split="validation", generation_size=300, metrics=[CHRFPP_CORPUS, CHRFPP],
        stop_sequence=["\n"], version=0,
    )


FLORES_TASKS = [
    _flores_task(source, target)
    for language in FLORES_LANGUAGES
    for source, target in ((language, "eng_Latn"), ("eng_Latn", language))
]



# Pinned clones of native multilingual tasks.  Formal launchers use this local
# registry so upstream task code cannot silently follow a moving dataset HEAD.
def _pinned(tasks, prefixes, revision):
    selected = []
    for task in tasks:
        if any(task.name.startswith(prefix) for prefix in prefixes):
            clone = deepcopy(task)
            clone.hf_revision = revision
            selected.append(clone)
    return selected


def _pinned_renamed(tasks, prefixes, revision, rename):
    selected = _pinned(tasks, prefixes, revision)
    for clone in selected:
        clone.name = rename(clone.name)
    return selected


def _module_lists(module):
    result = []
    seen = set()
    for value in vars(module).values():
        if not isinstance(value, list):
            continue
        for item in value:
            name = getattr(item, "name", None)
            if name and name not in seen:
                seen.add(name)
                result.append(item)
    return result


def _global_mmlu_cf(tasks, revision):
    """Create answer-text CF variants missing from upstream Global-MMLU."""
    languages = {
        "swa": Language.SWAHILI,
        "amh": Language.AMHARIC,
        "yor": Language.YORUBA,
    }
    selected = []
    for task in tasks:
        match = re.match(r"global_mmlu_all_(swa|amh|yor)_mcf:", task.name)
        if not match:
            continue
        clone = deepcopy(task)
        clone.name = task.name.replace("_mcf:", "_cf:", 1)
        clone.hf_revision = revision
        clone.prompt_function = get_mcq_prompt_function(
            languages[match.group(1)],
            lambda line: {
                "question": line["question"],
                "choices": [line["option_a"], line["option_b"], line["option_c"], line["option_d"]],
                "gold_idx": "ABCD".index(line["answer"]),
            },
            formulation=CFFormulation(),
        )
        clone.metrics = [
            LogLikelihoodAccMetric(normalization=None),
            LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
            LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
        ]
        selected.append(clone)
    return selected


PINNED_ENGLISH_TASKS = (
    _pinned_renamed(native_ifeval.TASKS_TABLE, ("ifeval",), "966cd89545d6b6acfd7638bc708b98261ca58e84", lambda _: "cl_en_ifeval")
    + _pinned_renamed(native_gsm8k.TASKS_TABLE, ("gsm8k",), "740312add88f781978c0658806c59bc2815b9866", lambda _: "cl_en_gsm8k")
    + _pinned_renamed(native_mmlu.TASKS_TABLE, ("mmlu:",), "31d46ab06e6934bb0d95f6918668716d1db6f921", lambda name: name.replace("mmlu:", "cl_en_mmlu:", 1))
    + _pinned_renamed(native_hellaswag.TASKS_TABLE, ("hellaswag",), "218ec52e09a7e7462a5400043bb9a69a41d06b76", lambda _: "cl_en_hellaswag")
    + _pinned_renamed(native_arc.TASKS_TABLE, ("arc:challenge",), "210d026faf9955653af8916fad021475a3f00453", lambda _: "cl_en_arc:challenge")
    + _pinned_renamed(native_truthfulqa.TASKS_TABLE, ("truthfulqa:mc",), "741b8276f2d1982aa3d5b832d3ee81ed3b896490", lambda _: "cl_en_truthfulqa:mc")
)


PINNED_NATIVE_TASKS = (
    _pinned(native_global_mmlu.TASKS_TABLE, ("global_mmlu_all_swa_mcf:", "global_mmlu_all_amh_mcf:", "global_mmlu_all_yor_mcf:"), "0e619dbeb34206cd48705a1a0ea7fb21cae09993")
    + _global_mmlu_cf(native_global_mmlu.TASKS_TABLE, "0e619dbeb34206cd48705a1a0ea7fb21cae09993")
    + _pinned(_module_lists(native_afri_mmlu), (
        "afri_mmlu_swa_mcf:", "afri_mmlu_amh_mcf:", "afri_mmlu_yor_mcf:",
        "afri_mmlu_swa_cf:", "afri_mmlu_amh_cf:", "afri_mmlu_yor_cf:",
    ), "16938ceaa58d24be90e501abbbb12cdacf5810fa")
    + _pinned(native_afri_mgsm.TASKS_TABLE, ("afri_mgsm_swa", "afri_mgsm_amh", "afri_mgsm_yor"), "8e4268d2b94941f18f63f694cf48e4ae26fbec65")
    + _pinned(native_afri_xnli.TASKS_TABLE, (
        "afri_xnli_swa_mcf", "afri_xnli_amh_mcf", "afri_xnli_yor_mcf",
        "afri_xnli_swa_cf", "afri_xnli_amh_cf", "afri_xnli_yor_cf",
    ), "e3ca06b30f3e7af2a86f6c8609ea76fee326bc56")
    + _pinned(native_belebele.TASKS_TABLE, (
        "belebele_swh_Latn_mcf", "belebele_amh_Ethi_mcf", "belebele_yor_Latn_mcf",
        "belebele_swh_Latn_cf", "belebele_amh_Ethi_cf", "belebele_yor_Latn_cf",
    ), "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2")
    + _pinned(native_xcsqa.TASKS_TABLE, ("xcsqa_swa_mcf", "xcsqa_swa_cf"), "f691f311a11f84773a899e9bfc7da2bae51b0b02")
    + _pinned(native_xnli.TASKS_TABLE, ("xnli_swa_mcf", "xnli_swa_cf"), "b8dd5d7af51114dbda02c0e3f6133f332186418e")
    + _pinned(native_xcopa.TASKS_TABLE, ("xcopa_swa_mcf", "xcopa_swa_cf"), "042f78955ba48e6404616762fa6e05e839c3907a")
    + _pinned(native_openai_mmlu.TASKS_TABLE, (
        "openai_mmlu_swa_mcf:", "openai_mmlu_yor_mcf:",
        "openai_mmlu_swa_cf:", "openai_mmlu_yor_cf:",
    ), "038c7808122969ead7456361af05cb8f47d247f8")
)

# The pinned SW/AM/YOR test splits contain no high_school_mathematics rows.
# Removing only these zero-document configs preserves every official example
# and prevents LightEval from aborting the entire grouped benchmark.
PINNED_NATIVE_TASKS = [
    task for task in PINNED_NATIVE_TASKS
    if not (task.name.startswith("afri_mmlu_") and task.name.endswith(":high_school_mathematics"))
]


def _uses_unimplemented_pmi(metric):
    sample_fn = getattr(metric, "sample_level_fn", None)
    return isinstance(getattr(sample_fn, "logprob_normalization", None), LogProbPMINorm)


# This LightEval checkout declares PMI metrics but never schedules or stores the
# required unconditioned log-probability requests. Do not silently substitute
# conditioned scores: omit PMI here and compute it only in a dedicated paired
# conditioned/unconditioned evaluator.
for _task in PINNED_ENGLISH_TASKS + PINNED_NATIVE_TASKS + SIB200_TASKS:
    _task.metrics = [metric for metric in _task.metrics if not _uses_unimplemented_pmi(metric)]


TASKS_TABLE = PINNED_ENGLISH_TASKS + PINNED_NATIVE_TASKS + MGSM_OFFICIAL_TASKS + MGSM_COT_TASKS + SIB200_TASKS + XLSUM_TASKS + FLORES_TASKS + TASKMIX_TASKS

# Isolated baseline protocol uses this same public registry. Other families are unchanged.
if os.environ.get('BASELINE_EVAL_TASK_SOURCE'):
    import importlib.util as _baseline_import
    from pathlib import Path as _BaselinePath
    _source = os.environ['BASELINE_EVAL_TASK_SOURCE']
    if _source not in {'sum.py', 'mt.py', 'belebele.py', 'mmlu_local.py', 'gmmlu.py'}:
        raise ValueError('Unknown frozen baseline task source')
    _spec = _baseline_import.spec_from_file_location('baseline_selected_task', _BaselinePath(__file__).parent / 'src' / _source)
    _module = _baseline_import.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    TASKS_TABLE = _module.TASKS_TABLE
