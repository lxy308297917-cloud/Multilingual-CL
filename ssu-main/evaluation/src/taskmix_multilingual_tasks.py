"""Pinned local LightEval tasks for the multilingual task-mixture protocols."""

from __future__ import annotations

import os

import numpy as np
import sacrebleu

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.metrics.metrics_corpus import CorpusLevelComputation
from lighteval.metrics.sample_preparator import GenerativePreparator
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.metrics.utils.metric_utils import CorpusLevelMetric, SampleLevelComputation, SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.templates.nli import get_nli_prompt_function
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.tasks.templates.utils.formulation import MCFFormulation
from lighteval.utils.language import Language


LANGUAGES = {
    "ru": (Language.RUSSIAN, "rus_Cyrl"),
    "ar": (Language.ARABIC, "arb_Arab"),
    "sw": (Language.SWAHILI, "swh_Latn"),
    "bn": (Language.BENGALI, "ben_Beng"),
    "te": (Language.TELUGU, "tel_Telu"),
}
XNLI_ROOT = os.environ.get("MULTILINGUAL_CL_XNLI_ROOT", "/root/autodl-tmp/multilingual_cl_v3_task/eval_data/xnli")
MASSIVE_ROOT = os.environ.get("MULTILINGUAL_CL_MASSIVE_ROOT", "/root/autodl-tmp/multilingual_cl_v3_task/eval_data/massive")
TYDIQA_ROOT = os.environ.get("MULTILINGUAL_CL_TYDIQA_ROOT", "/root/autodl-tmp/multilingual_cl_v3_task/eval_data/tydiqa")


def _response_text(response) -> str:
    value = getattr(response, "final_text", response)
    if isinstance(value, dict):
        value = value.get("text", "")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip()


class FixedCorpusChrFPlusPlus(CorpusLevelComputation):
    """Standard corpus chrF++ with references transposed to reference streams."""

    def compute_corpus(self, items):
        predictions = [_response_text(item.preds) for item in items]
        references = [str(item.golds[0]) for item in items]
        return float(sacrebleu.corpus_chrf(predictions, [references], word_order=2).score)


CHRFPP_CORPUS = CorpusLevelMetric(
    metric_name="chrfpp_corpus", higher_is_better=True, category=SamplingMethod.GENERATIVE,
    sample_level_fn=GenerativePreparator(), corpus_level_fn=FixedCorpusChrFPlusPlus(),
)


class ExactLabelMetric(SampleLevelComputation):
    def compute(self, doc, model_response, **kwargs):
        gold_index = doc.gold_index.tolist() if hasattr(doc.gold_index, "tolist") else doc.gold_index
        if isinstance(gold_index, (list, tuple)):
            gold_index = gold_index[0]
        gold = str(doc.choices[int(gold_index)]).strip().casefold()
        predicted = _response_text(model_response).splitlines()[0].strip().strip("`'\" .,:;").casefold()
        return float(predicted == gold)


INTENT_ACCURACY = SampleLevelMetric(
    metric_name="intent_accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=ExactLabelMetric(),
    corpus_level_fn=np.mean,
)


def _massive_prompt(language_code: str):
    instructions = {
        "ru": "Определите намерение высказывания. Выведите только одну метку из списка",
        "ar": "حدّد نية العبارة. أخرج تسمية واحدة فقط من القائمة",
        "sw": "Tambua nia ya kauli. Toa lebo moja tu kutoka kwenye orodha",
        "bn": "বক্তব্যটির উদ্দেশ্য নির্ধারণ করুন। তালিকা থেকে শুধু একটি লেবেল লিখুন",
        "te": "వాక్యం యొక్క ఉద్దేశాన్ని గుర్తించండి. జాబితా నుండి ఒక లేబుల్ మాత్రమే ఇవ్వండి",
    }

    def prompt(line, task_name: str):
        labels = sorted(set(line.get("all_intents", [])))
        if not labels:
            labels = [
                "activate_my_card", "age_limit", "apple_pay_or_google_pay", "beneficiary_not_allowed",
                "cash_withdrawal", "cash_withdrawal_card", "cash_withdrawal_charge",
            ]
        query = f"{instructions[language_code]}: {', '.join(labels)}\nТекст/Text: {line['utt']}"
        return Doc(task_name=task_name, query=query, choices=[str(line["intent"])], gold_index=0,
                   specific={"source_id": str(line.get("id", ""))})
    return prompt


def _tydi_adapter(line):
    return {
        "context": str(line["context"]),
        "question": str(line["question"]),
        "choices": [str(x) for x in line["answers"]["text"]],
    }


def _xnli_adapter(line):
    label = int(line["label"])
    if label not in (0, 1, 2):
        return None
    return {"premise": line["premise"], "hypothesis": line["hypothesis"], "gold_idx": label}


TASKS_TABLE = []
for short, (language, _) in LANGUAGES.items():
    formulation = MCFFormulation()
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"xnli3_cl_{short}",
            prompt_function=get_nli_prompt_function(
                language, _xnli_adapter,
                relations=["entailment", "neutral", "contradiction"], formulation=formulation,
            ),
            hf_repo=os.path.join(XNLI_ROOT, short), hf_subset=None,
            hf_avail_splits=["validation", "test"], evaluation_splits=["test"],
            metrics=get_metrics_for_formulation(
                formulation,
                [LogLikelihoodAccMetric(normalization=LogProbTokenNorm()), LogLikelihoodAccMetric(normalization=LogProbCharNorm())],
            ),
            generation_size=-1,
        )
    )
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"massive_intent_cl_{short}", prompt_function=_massive_prompt(short),
            hf_repo=os.path.join(MASSIVE_ROOT, short), hf_subset=None,
            hf_avail_splits=["validation", "test"], evaluation_splits=["test"],
            metrics=[INTENT_ACCURACY], generation_size=32, stop_sequence=["\n"],
        )
    )
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"tydiqa_cl_{short}", prompt_function=get_qa_prompt_function(language, _tydi_adapter),
            hf_repo=os.path.join(TYDIQA_ROOT, short), hf_subset=None,
            hf_avail_splits=["test"], evaluation_splits=["test"],
            metrics=[MultilingualQuasiExactMatchMetric(language, "prefix"), MultilingualQuasiF1ScoreMetric(language)],
            generation_size=400, stop_sequence=["\n"],
        )
    )
