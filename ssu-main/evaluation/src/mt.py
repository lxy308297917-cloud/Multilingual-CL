from aenum import extend_enum
import numpy as np

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

TASKS_TABLE = []

# CUSTOM METRIC IF NEEDED
class SampleLevelTranslationMetric:
    def __init__(self, metric_type: str):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
        """
        import sacrebleu
        self.metric_type = metric_type
        if metric_type == "chrf":
            self.metric = sacrebleu.sentence_chrf
        elif metric_type == "chrf++":
            self.metric = sacrebleu.sentence_chrf
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {metric_type}")

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        assert len(golds) == 1 and len(predictions) == 1
        if self.metric_type == "chrf++":
            return float(self.metric(predictions.pop(), golds, word_order=2).score)
        else:
            return float(self.metric(predictions.pop(), golds).score)

chrf_sample = SampleLevelMetric(
    metric_name="chrfpp_sample",
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    sample_level_fn=SampleLevelTranslationMetric("chrf++").compute, # how to compute score for one sample
    corpus_level_fn=np.mean, # aggregation
    higher_is_better=True,
)
extend_enum(Metrics, "chrfpp_sample", chrf_sample)


def lang_code_to_2en_instruction(lang_code: str) -> str:
    if lang_code == "am":
        return "አማርኛን ወደ እንግሊዝኛ ተርጉም:\n"
    elif lang_code == "ne":
        return "नेपालीलाई अङ्ग्रेजीमा अनुवाद गर्नुहोस्:\n"
    elif lang_code == "ha":
        return "Fassara Hausa zuwa Turanci:\n"
    elif lang_code == "ig":
        return "Sụgharịa Igbo gaa na Bekee:\n"
    elif lang_code == "ky":
        return "Кыргызчадан англисчеге которуу:\n"
    elif lang_code == "da":
        return "Oversæt dansk til engelsk:\n"
    elif lang_code == "is":
        return "Þýddu íslensku yfir á ensku:\n"
    elif lang_code == "no":
        return "Oversett norsk til engelsk:\n"
    elif lang_code == "fil":
        return "Isalin ang Filipino sa Ingles:\n"
    elif lang_code == "ro":
        return "Tradu româna în engleză:\n"
    elif lang_code == "id":
        return "Terjemahkan bahasa Indonesia ke bahasa Inggris:\n"
    elif lang_code == "bn":
        return "বাংলা থেকে ইংরেজিতে অনুবাদ করুন:\n"
    elif lang_code == "el":
        return "Μεταφράστε τα ελληνικά στα αγγλικά:\n"
    elif lang_code == "he":
        return "תרגם מעברית לאנגלית:\n"
    elif lang_code == "ko":
        return "한국어를 영어로 번역하세요:\n"
    elif lang_code == "lt":
        return "Išverskite iš lietuvių kalbos į anglų kalbą:\n"
    elif lang_code == "ms":
        return "Terjemahkan bahasa Melayu ke bahasa Inggeris:\n"
    elif lang_code == "uk":
        return "Перекладіть з української на англійську:\n"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")

def lang_code_to_2tgt_instruction(lang_code: str) -> str:
    if lang_code == "am":
        return "እንግሊዝኛን ወደ አማርኛ ተርጉም:\n"
    elif lang_code == "ne":
        return "अङ्ग्रेजीलाई नेपालीमा अनुवाद गर्नुहोस्:\n"
    elif lang_code == "ha":
        return "Fassara Turanci zuwa Hausa:\n"
    elif lang_code == "ig":
        return "Sụgharịa Bekee gaa n'Igbo:\n"
    elif lang_code == "ky":
        return "Англисчеден кыргызчага которуу:\n"
    elif lang_code == "da":
        return "Oversæt engelsk til dansk:\n"
    elif lang_code == "is":
        return "Þýddu ensku yfir á íslensku:\n"
    elif lang_code == "no":
        return "Oversett engelsk til norsk:\n"
    elif lang_code == "fil":
        return "Isalin ang Ingles sa Filipino:\n"
    elif lang_code == "ro":
        return "Tradu engleza în română:\n"
    elif lang_code == "id":
        return "Terjemahkan bahasa Inggris ke bahasa Indonesia:\n"
    elif lang_code == "bn":
        return "ইংরেজি থেকে বাংলায় অনুবাদ করুন:\n"
    elif lang_code == "el":
        return "Μεταφράστε τα αγγλικά στα ελληνικά:\n"
    elif lang_code == "he":
        return "תרגם מאנגלית לעברית:\n"
    elif lang_code == "ko":
        return "영어를 한국어로 번역하세요:\n"
    elif lang_code == "lt":
        return "Išverskite iš anglų kalbos į lietuvių kalbą:\n"
    elif lang_code == "ms":
        return "Terjemahkan bahasa Inggeris ke bahasa Melayu:\n"
    elif lang_code == "uk":
        return "Перекладіть з англійської на українську:\n"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


def buffer_fn_2en(
    language: str, 
    instruction: str,
):
    def prompt_fn(line, task_name: str):
        return Doc(
            task_name=task_name,
            query=f"{instruction}{line[language]} =",
            gold_index=0,
            choices=[line["en"]],
            instruction=instruction,
        )
    return prompt_fn


def buffer_fn_2tgt(
    language: str,
    instruction: str,
):
    def prompt_fn(line, task_name: str):
        return Doc(
            task_name=task_name,
            query=f"{instruction}{line['en']} =",
            gold_index=0,
            choices=[line[language]],
            instruction=instruction,
        )
    return prompt_fn


for language in [
    "am", "ne", "ha", "ig", "ky",
    "da", "is", "no",
    "fil", "ro", "id",
    "bn", "el", "he", "ko", "lt", "ms", "uk",
]:
    task = LightevalTaskConfig(
        name=f"mt:{language}2en",
        prompt_function=buffer_fn_2en(
            language=language,
            instruction=lang_code_to_2en_instruction(language),
        ),
        suite=("custom",),
        hf_repo="/data/HwHiAiUser/cl_workspace/data/mt_flores_ssu_18",
        hf_subset="default",
        evaluation_splits=("test",),
        hf_avail_splits=["validation", "test"],
        metric=[chrf_sample],
        generation_size=128,
        stop_sequence=["\n"],
        trust_dataset=True,
    )
    TASKS_TABLE.append(task)

    task = LightevalTaskConfig(
        name=f"mt:en2{language}",
        prompt_function=buffer_fn_2tgt(
            language=language,
            instruction=lang_code_to_2tgt_instruction(language),
        ),
        suite=("custom",),
        hf_repo="/data/HwHiAiUser/cl_workspace/data/mt_flores_ssu_18",
        # hf_repo="/home/HwHiAiUser/cl_workspace/data/mt_flores_ssu_shift3",
        hf_subset="default",
        evaluation_splits=("test",),
        hf_avail_splits=["validation", "test"],
        metric=[chrf_sample],
        generation_size=128,
        stop_sequence=["\n"],
        trust_dataset=True,
    )
    TASKS_TABLE.append(task)
