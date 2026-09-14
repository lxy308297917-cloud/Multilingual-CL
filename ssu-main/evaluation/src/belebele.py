import os
# 名字叫做：belebele.py
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

from langcodes import Language as LangCodeLanguage

from pathlib import Path
from datasets import load_from_disk

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language, iso_639_3_ind_to_iso_639_3_macro
from lighteval.tasks.templates.utils.translation_literals import (
    TRANSLATION_LITERALS,
    TranslationLiterals,
)


def resolve_language(tag: str) -> Language:
    code = LangCodeLanguage.get(tag).to_alpha3()
    if code in iso_639_3_ind_to_iso_639_3_macro:
        return iso_639_3_ind_to_iso_639_3_macro[code]
    return Language(code)

TRANSLATION_LITERALS[Language.IGBO] = TranslationLiterals(
    language=Language.IGBO,
    question_word="ajụjụ",
    answer="azịza",
    confirmation_word="ọ dị mma",
    yes="ee",
    no="mba",
    also="kwa",
    cause_word="n’ihi na",
    effect_word="ya mere",
    or_word="ma ọ bụ",
    and_word="na",
    true="ezigbo",
    false="ụgha",
    neither="ọ dịghị nke a",
)


TASKS_TABLE = []

LOCAL_BELEBELE_ROOT = Path((os.path.join(os.environ["BASELINE_EVAL_DATA_ROOT"], "belebele") if os.environ.get("BASELINE_EVAL_DATA_ROOT") else "/root/autodl-tmp/eval_datasets_local/belebele"))

# Belebele: A large-scale reading comprehension dataset covering 122 languages.
# https://arxiv.org/abs/2308.16884
belebele_tasks = [
    LightevalTaskConfig(
        name=f"belebele_{language}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            resolve_language(language),
            lambda line: {
                "question": line["question"],
                "context": line["flores_passage"],
                "choices": [line[f"mc_answer{i}"] for i in range(1, 5)],
                "gold_idx": int(line["correct_answer_num"]) - 1,
            },
            formulation=formulation,
        ),
        hf_repo=str(LOCAL_BELEBELE_ROOT / language),
        hf_subset=None,
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
    for language in [
        "acm_Arab",
        "arz_Arab",
        "ceb_Latn",
        "fin_Latn",
        "hin_Deva",
        "ita_Latn",
        "khm_Khmr",
        "lvs_Latn",
        "npi_Deva",
        "pol_Latn",
        "slv_Latn",
        "swe_Latn",
        # "tso_Latn",
        # "xho_Latn",
        "afr_Latn",
        "asm_Beng",
        "ces_Latn",
        "fra_Latn",
        "hin_Latn",
        "jav_Latn",
        # "kin_Latn",
        "mal_Mlym",
        "npi_Latn",
        "por_Latn",
        # "sna_Latn",
        "swh_Latn",
        "tur_Latn",
        "yor_Latn",
        "als_Latn",
        "azj_Latn",
        "ckb_Arab",
        # "fuv_Latn",
        "hrv_Latn",
        "jpn_Jpan",
        "kir_Cyrl",
        "mar_Deva",
        # "nso_Latn",
        "snd_Arab",
        "tam_Taml",
        "ukr_Cyrl",
        "zho_Hans",
        "amh_Ethi",
        # "bam_Latn",
        "dan_Latn",
        # "gaz_Latn",
        "hun_Latn",
        # "kac_Latn",
        "kor_Hang",
        "mkd_Cyrl",
        # "nya_Latn",
        "ron_Latn",
        "som_Latn",
        "tel_Telu",
        "urd_Arab",
        "zho_Hant",
        "apc_Arab",
        "ben_Beng",
        "deu_Latn",
        # "grn_Latn",
        "hye_Armn",
        "kan_Knda",
        "lao_Laoo",
        "mlt_Latn",
        "ory_Orya",
        "rus_Cyrl",
        # "sot_Latn",
        "tgk_Cyrl",
        "urd_Latn",
        "zsm_Latn",
        "arb_Arab",
        "ben_Latn",
        "ell_Grek",
        "guj_Gujr",
        "ibo_Latn",
        "kat_Geor",
        # "lin_Latn",
        # "mri_Latn",
        "pan_Guru",
        # "shn_Mymr",
        "spa_Latn",
        "tgl_Latn",
        "uzn_Latn",
        # "zul_Latn",
        "arb_Latn",
        # "bod_Tibt",
        "eng_Latn",
        # "hat_Latn",
        # "ilo_Latn",
        "kaz_Cyrl",
        "lit_Latn",
        "mya_Mymr",
        "pbt_Arab",
        "sin_Latn",
        "srp_Cyrl",
        "tha_Thai",
        "vie_Latn",
        "ars_Arab",
        "bul_Cyrl",
        "est_Latn",
        "hau_Latn",
        "ind_Latn",
        # "kea_Latn",
        # "lug_Latn",
        "nld_Latn",
        "pes_Arab",
        "sin_Sinh",
        # "ssw_Latn",
        # "tir_Ethi",
        "war_Latn",
        "ary_Arab",
        "cat_Latn",
        "eus_Latn",
        "heb_Hebr",
        "isl_Latn",
        # "khk_Cyrl",
        # "luo_Latn",
        "nob_Latn",
        "plt_Latn",
        "slk_Latn",
        # "sun_Latn",
        # "tsn_Latn",
        # "wol_Latn",
    ]
]

TASKS_TABLE.extend(belebele_tasks)