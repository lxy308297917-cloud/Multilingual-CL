# 名字叫做：gmmlu.py

# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Callable
from pathlib import Path
from typing_extensions import NotRequired, TypedDict

from functools import partial
from langcodes import standardize_tag

from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.utils.adapter_utils import create_adapter_from_dict
from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.utils.formulation import MCFFormulation, Formulation, build_answers, build_choices
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language
from lighteval.utils.utils import as_list
from lighteval.tasks.templates.utils.formatting_utils import capitalize, fix_ending_punct


LOCAL_GMMLU_ROOT = Path("/data/HwHiAiUser/cl_workspace/data/gmmlu")
language_to_code = {
    "ne": "npi",
    "am": "amh",
    "ha": "hau",
    "ig": "ibo",
    "ky": "kir",
    "da": "dan",
    "is": "isl",
    "no": "nob",
    "fil": "fil",
    "ro": "ron",
    "id": "ind",
    "bn": "ben",
    "el": "ell",
    "he": "heb",
    "ko": "kor",
    "lt": "lit",
    "ms": "zsm",
    "uk": "ukr",
}

language_to_local_dir = {
    "ne": "ne",
    "am": "am",
    "ha": "ha",
    "ig": "ig",
    "ky": "ky",
    "da": "da",
    "is": "is",
    "no": "no",
    "fil": "fil",
    "ro": "ro",
    "id": "id",
    "bn": "bn",
    "el": "el",
    "he": "he",
    "ko": "ko",
    "lt": "lt",
    "ms": "ms",
    "uk": "uk",
}


language_to_class = {
    "ne": Language.NEPALI,
    "am": Language.AMHARIC,
    "ha": Language.HAUSA,
    "ig": Language.IGBO,
    "ky": Language.KYRGYZ,
    "da": Language.DANISH ,
    "is": Language.ICELANDIC,
    "no": Language.NORWEGIAN_BOKMAL,
    "fil": Language.TAGALOG,
    "ro": Language.ROMANIAN,
    "id": Language.INDONESIAN,
    "bn": Language.BENGALI,
    "el": Language.GREEK,
    "he": Language.HEBREW,
    "ko": Language.KOREAN,
    "lt": Language.LITHUANIAN,
    "ms": Language.MALAY,
    "uk": Language.UKRAINIAN,
}

TASKS_TABLE = []
MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def lang_code_to_instruction(lang_code: str, subject: str) -> str:
    if lang_code == "am":
        return f"ከታች ስለ {subject.replace('_', ' ')} የቀረቡ ባለብዙ ምርጫ ጥያቄዎች (ከመልሶች ጋር) ናቸው።\n\n"
    elif lang_code == "ne":
        return f"तल {subject.replace('_', ' ')} सम्बन्धी बहु-विकल्प प्रश्नहरू (उत्तर सहित) दिइएका छन्।\n\n"
    elif lang_code == "ha":
        return f"Waɗannan tambayoyi masu zaɓi da yawa (tare da amsoshi) game da {subject.replace('_', ' ')} ne.\n\n"
    elif lang_code == "ig":
        return f"Nke a bụ ajụjụ ọnụ nhọrọ ọtụtụ (na azịza) gbasara {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "ky":
        return f"Бул {subject.replace('_', ' ')} боюнча бир нече тандоо суроолору (жооптор менен) төмөндө келтирилген.\n\n"
    elif lang_code == "en":
        return f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "da":
        return f"Nedenfor er multiple choice-spørgsmål (med svar) om {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "is":
        return f"Hér fyrir neðan eru fjölvalsspurningar (með svörum) um {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "no":
        return f"Nedenfor er flervalgsspørsmål (med svar) om {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "fil":
        return f"Ang mga sumusunod ay mga tanong na maramihang pagpipilian (na may sagot) tungkol sa {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "ro":
        return f"Următoarele sunt întrebări cu variante multiple de răspuns (cu răspunsuri) despre {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "id":
        return f"Berikut adalah pertanyaan pilihan ganda (beserta jawaban) tentang {subject.replace('_', ' ')}.\n\n"  
    elif lang_code == "bn":
        return f"নিচে {subject.replace('_', ' ')} সম্পর্কে বহু নির্বাচনী প্রশ্ন (উত্তরসহ) দেওয়া হলো।\n\n"
    elif lang_code == "el":
        return f"Ακολουθούν ερωτήσεις πολλαπλής επιλογής (με απαντήσεις) σχετικά με το {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "he":
        return f"להלן שאלות רב-ברירה (עם תשובות) בנושא {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "ko":
        return f"다음은 {subject.replace('_', ' ')}에 관한 객관식 문제(정답 포함)입니다.\n\n"
    elif lang_code == "lt":
        return f"Toliau pateikiami kelių pasirinkimų klausimai (su atsakymais) apie {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "ms":
        return f"Berikut ialah soalan aneka pilihan (berserta jawapan) tentang {subject.replace('_', ' ')}.\n\n"
    elif lang_code == "uk":
        return f"Нижче наведено запитання з кількома варіантами відповіді (з відповідями) щодо {subject.replace('_', ' ')}.\n\n"  
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


MULTI_CHOICE_QA_QUERY = (
    "{instruction}{context}{question_word}{colon}{sentence_space}{question}\n{options}{answer_word}{colon}"
)


# Defined for type hinting only
class MCQInput(TypedDict):
    """
    Input for the multiple choice task.
    Args:
        question: The question to be answered (e.g. What is the capital of France?)
        choices: Possible choices for the question (e.g. [Paris, London, Berlin, Rome])
        gold_idx: The index of the correct choice
        context (optional): The context of the question (e.g. Capital of France starts with P)
        instruction (optional): The instruction of the task (e.g. Answer the following question)
    """

    question: str
    choices: list[str]
    gold_idx: list[int] | int
    context: NotRequired[str]
    instruction: NotRequired[str]


class MCQDictAdapter(TypedDict):
    """
    Adapter for mapping from the dataset row into the MCQInput format.
    Args:
        question: Column name in the row that contains the question to be answered (e.g. What is the capital of France?)
        choices: Column name in the row that contains the possible choices for the question (e.g. [Paris, London, Berlin, Rome])
        gold_idx: Column name in the row that contains the index of the correct choice
        context (optional): Column name in the row that contains the context of the question (e.g. Capital of France starts with P)
        instruction (optional): Column name in the row that contains the instruction of the task (e.g. Answer the following question)
    """

    question: str
    choices: str
    gold_idx: str
    context: NotRequired[str]
    instruction: NotRequired[str]



def get_mcq_prompt_function(
    language: Language,
    adapter: Callable[[dict], MCQInput | None] | MCQDictAdapter,
    formulation: Formulation = MCFFormulation(),
):
    """
    Create a templated prompt function for a Multiple Choice Question (MCQ) task.
    Example tasks:
    - ARC
    - TruthfulQA

    Format:
    *CF*
    Question: xxx
    Answer: | Answer

    *Hybrid*
    Question: xxx
    A. Answer
    B. Answer
    C. Answer
    D. Answer
    Answer: | Answer

    *MCF*
    Question: xxx
    A. Answer
    B. Answer
    C. Answer
    D. Answer
    Answer: | A/B/C/D

    Args:
        language (Language): The language of the MCQ task.
        adapter (Callable[[dict], MCQInput] | MCQDictAdapter): A function or dictionary to adapt the input data to the required MCQInput format.
            Must map data from the dataset row to the MCQInput format.
        formulation (Formulation, optional): The formulation to use for the task. Defaults to MCFFormulation().

    Returns:
        Callable: A function that generates MCQ prompts based on the given parameters.
    """

    adapter_fn = create_adapter_from_dict(adapter)

    def prompt_fn(line, task_name: str):
        mcq_input = adapter_fn(line)
        if mcq_input is None:
            return None

        translation_literals = TRANSLATION_LITERALS[language_to_class[language]]

        instruction = lang_code_to_instruction(language, line["subject"])
        context_val = mcq_input.get("context")
        context = f"{capitalize(fix_ending_punct(context_val, translation_literals))}\n" if context_val else ""
        question = capitalize(fix_ending_punct(mcq_input["question"], translation_literals))
        answers = [capitalize(fix_ending_punct(str(answer), translation_literals)) for answer in mcq_input["choices"]]
        options = build_choices(answers, formulation, translation_literals)
        options = f"{options}\n" if options else ""
        answers = build_answers(answers, formulation, translation_literals)
        answer_word = capitalize(translation_literals.answer)
        question_word = capitalize(translation_literals.question_word)

        query = MULTI_CHOICE_QA_QUERY.format(
            instruction=instruction,
            question=question,
            context=context,
            question_word=question_word,
            answer_word=answer_word,
            colon=translation_literals.colon,
            sentence_space=translation_literals.sentence_space,
            options=options,
        )

        return Doc(
            task_name=task_name,
            query=query,
            gold_index=as_list(mcq_input["gold_idx"]),
            choices=answers,
            instruction=instruction,
            unconditioned_query=f"{answer_word}{translation_literals.colon}",
        )

    return prompt_fn

global_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"gmmlu_{language_to_code[language]}_mcf:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": [line["option_a"], line["option_b"], line["option_c"], line["option_d"]],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
                "subject": subset,
            },
            formulation=MCFFormulation(),
        ),
        suite=("lighteval",),
        hf_repo=str(LOCAL_GMMLU_ROOT / language_to_local_dir[language]),
        hf_subset=None,
        evaluation_splits=("test",),
        few_shots_split="dev",
        hf_filter=partial(
            lambda subset, sensitivity_label, x: x["subject"].lower() == subset
            and (
                sensitivity_label == "ALL" or sensitivity_label in x["cultural_sensitivity_label"].replace("-", "UNK")
            )
            and all(x[f"option_{opt}"] is not None and x[f"option_{opt}"].strip() for opt in "abcd"),
            subset,
            "ALL",
        ),
        metric=get_metrics_for_formulation(
            MCFFormulation(),
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in MMLU_SUBSETS
    for language in [
            "ne",
            "am",
            "ha",
            "ig",
            "ky",
            "da",
            "is",
            "no",
            "fil",
            "ro",
            "id",
            "bn",
            "el",
            "he",
            "ko",
            "lt",
            "ms",
            "uk",
        ]
]
TASKS_TABLE.extend(global_mmlu_tasks)
