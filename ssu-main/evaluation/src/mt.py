import numpy as np

from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

TASKS_TABLE = []

# ===== 本地 MT 数据集调试打印 =====
import os
import json

LOCAL_MT_DATA_DIR = (os.path.join(os.environ["BASELINE_EVAL_DATA_ROOT"], "mt_flores_ssu_9langs") if os.environ.get("BASELINE_EVAL_DATA_ROOT") else "/root/autodl-tmp/eval_datasets_local/mt_flores_ssu_9langs")

print("========================================")
print("[MT.py] 正在加载自定义机器翻译任务")
print(f"[MT.py] 本地 MT 数据目录：{LOCAL_MT_DATA_DIR}")
print(f"[MT.py] 目录是否存在：{os.path.exists(LOCAL_MT_DATA_DIR)}")
print(f"[MT.py] 是否为目录：{os.path.isdir(LOCAL_MT_DATA_DIR)}")

if os.path.isdir(LOCAL_MT_DATA_DIR):
    print("[MT.py] 本地 MT 数据目录下的文件：")
    for fn in sorted(os.listdir(LOCAL_MT_DATA_DIR)):
        fp = os.path.join(LOCAL_MT_DATA_DIR, fn)
        if os.path.isfile(fp):
            print(f"[MT.py]   文件名：{fn} | 大小：{os.path.getsize(fp)} bytes")

    test_file = os.path.join(LOCAL_MT_DATA_DIR, "test.jsonl")
    if os.path.isfile(test_file):
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                line = f.readline().strip()
            obj = json.loads(line)
            print(f"[MT.py] test.jsonl 第一行字段：{list(obj.keys())}")
            preview = {k: str(v)[:80] for k, v in obj.items()}
            print(f"[MT.py] test.jsonl 第一行内容预览：{preview}")
        except Exception as e:
            print(f"[MT.py] 读取 test.jsonl 第一行失败：{type(e).__name__}: {e}")
    else:
        print("[MT.py] 警告：没有找到 test.jsonl")
else:
    print("[MT.py] 警告：本地 MT 数据目录不存在！")

print("========================================")
# =======================================

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
        hf_repo=(os.path.join(os.environ["BASELINE_EVAL_DATA_ROOT"], "mt_flores_ssu_9langs") if os.environ.get("BASELINE_EVAL_DATA_ROOT") else "/root/autodl-tmp/eval_datasets_local/mt_flores_ssu_9langs"),
        hf_subset="default",
        evaluation_splits=("test",),
        hf_avail_splits=["validation", "test"],
        metrics=[chrf_sample],
        generation_size=128,
        stop_sequence=["\n"],
    )
    TASKS_TABLE.append(task)

    task = LightevalTaskConfig(
        name=f"mt:en2{language}",
        prompt_function=buffer_fn_2tgt(
            language=language,
            instruction=lang_code_to_2tgt_instruction(language),
        ),
        hf_repo=(os.path.join(os.environ["BASELINE_EVAL_DATA_ROOT"], "mt_flores_ssu_9langs") if os.environ.get("BASELINE_EVAL_DATA_ROOT") else "/root/autodl-tmp/eval_datasets_local/mt_flores_ssu_9langs"),
        # hf_repo="/home/HwHiAiUser/cl_workspace/data/mt_flores_ssu_shift3",
        hf_subset="default",
        evaluation_splits=("test",),
        hf_avail_splits=["validation", "test"],
        metrics=[chrf_sample],
        generation_size=128,
        stop_sequence=["\n"],
    )
    TASKS_TABLE.append(task)
