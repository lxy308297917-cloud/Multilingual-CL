import numpy as np
import os
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


TASKS_TABLE = []
LOCAL_SUM_ROOT = "/root/autodl-tmp/eval_datasets_local/sum_ssu"

# CUSTOM METRIC IF NEEDED
class SampleLevelTranslationMetric(SampleLevelComputation):
    def __init__(self, metric_type: str):
        import sacrebleu

        self.metric_type = metric_type
        if metric_type not in {"chrf", "chrf++"}:
            raise ValueError(f"Unknown corpus level translation metric type: {metric_type}")
        self.metric = sacrebleu.sentence_chrf

    def compute(self, doc, model_response, **kwargs) -> float:
        golds = doc.get_golds()
        predictions = model_response.final_text
        if not golds or not predictions:
            raise ValueError("chrF++ requires at least one gold and prediction")
        word_order = 2 if self.metric_type == "chrf++" else 0
        return float(np.mean([
            self.metric(prediction, golds, word_order=word_order).score
            for prediction in predictions
        ]))

chrf_sample = SampleLevelMetric(
    metric_name="chrfpp_sample",
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=SampleLevelTranslationMetric("chrf++"),
    corpus_level_fn=np.mean, # aggregation
    higher_is_better=True,
)


def lang_code_to_instruction(lang_code: str) -> str:
    """Converts a language code to an instruction to summarize the text in that language.

    Args:
        lang_code: The language code

    Returns:
        The instruction in the specified language.

    Raises:
        ValueError: If the language code is unknown.
    """
    if lang_code == "en":
        return "Summarize the following text in English:"
    elif lang_code == "am":
        return "የታችኛው ጽሁፍን በአማርኛ አጭር በማድረግ አሳትረኝ።:"
    elif lang_code == "ne":
        return "तलको पाठलाई नेपालीमा संक्षेपमा लेख्नुहोस्:"
    elif lang_code == "ha":
        return "Taƙaita rubutu mai zuwa cikin Hausa:"
    elif lang_code == "ig":
        return "Chịkọta edemede a n'Igbo:"
    elif lang_code == "ky":
        return "Төмөнкү текстти кыргызча кыскача жазыңыз:"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


def lang_code_to_anchor(lang_code: str) -> str:
    """Converts a language code to an anchor to summarize the text in that language.

    Args:
        lang_code: The language code

    Returns:
        The anchor in the specified language.

    Raises:
        ValueError: If the language code is unknown.
    """
    if lang_code == "en":
        return "Summary:"
    elif lang_code == "am":
        return "አጭር መግለጫ:"
    elif lang_code == "ne":
        return "सारांश:"
    elif lang_code == "ha":
        return "Taƙaitawa:"
    elif lang_code == "ig":
        return "Nchịkọta:"
    elif lang_code == "ky":
        return "Кыскача:"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


def buffer_fn(
    instruction: str,
    anchor: str,
):
    def prompt_fn(line, task_name: str):
        summary = line["summary"]
        text = line["text"]
        return Doc(
            task_name=task_name,
            query=f"{instruction} {text} {anchor}",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text}
        )
    return prompt_fn


for language in [
    "en",
    "am",
    "ne",
    "ha",
    "ig",
    "ky",
]:
    task = LightevalTaskConfig(
        name=f"sum:{language}",
        prompt_function=buffer_fn(
            instruction=lang_code_to_instruction(language),
            anchor=lang_code_to_anchor(language),
        ),
        hf_repo=os.path.join(LOCAL_SUM_ROOT, language),
        hf_subset=None,
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        metrics=[chrf_sample],
        generation_size=128,
        stop_sequence=["\n"],
    )
    TASKS_TABLE.append(task)
