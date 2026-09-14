"""LightEval 0.10-compatible SIB-200 tasks for the five CL languages.

Dataset/prompt design follows the official LightEval SIB-200 task and the
Davlan/sib200 release.  Evaluation uses the held-out test split (204 examples
per language); no test row is used for training, replay, or method selection.
"""

from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import MCFFormulation
from lighteval.utils.language import Language


TASKS_TABLE = []
CHOICES = [
    "geography",
    "science/technology",
    "entertainment",
    "travel",
    "sports",
    "health",
    "politics",
]

LANGUAGES = {
    "ig": (
        "ibo_Latn",
        Language.IGBO,
        "Kedu isiokwu ederede na-esonụ na-ekwu maka ya?\n",
    ),
    "ha": (
        "hau_Latn",
        Language.HAUSA,
        "Wane batu ne rubutun da ke ƙasa yake magana a kai?\n",
    ),
    "ky": (
        "kir_Cyrl",
        Language.KYRGYZ,
        "Төмөнкү текст кайсы тема жөнүндө?\n",
    ),
    "ne": (
        "npi_Deva",
        Language.NEPALI,
        "तलको पाठ कुन विषयको बारेमा हो?\n",
    ),
    "am": (
        "amh_Ethi",
        Language.AMHARIC,
        "የሚከተለው ጽሑፍ ስለ ምን ርዕስ ነው?\n",
    ),
}


def _adapter(instruction):
    return lambda line: {
        "question": instruction + line["text"],
        "choices": CHOICES,
        "gold_idx": CHOICES.index(line["category"]),
    }


for short_code, (flores_code, language, instruction) in LANGUAGES.items():
    formulation = MCFFormulation()
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"sib200:{short_code}",
            prompt_function=get_mcq_prompt_function(
                language,
                _adapter(instruction),
                formulation=formulation,
            ),
            suite=("custom",),
            hf_repo="Davlan/sib200",
            hf_subset=flores_code,
            evaluation_splits=("test",),
            hf_avail_splits=["train", "validation", "test"],
            metric=get_metrics_for_formulation(
                formulation,
                [
                    loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                    loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                ],
            ),
            generation_size=-1,
            trust_dataset=True,
        )
    )
