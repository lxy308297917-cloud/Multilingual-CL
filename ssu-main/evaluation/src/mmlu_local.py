"""Offline English MMLU tasks equivalent to lighteval's leaderboard suite."""

from pathlib import Path
from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


LOCAL_MMLU = Path("/root/.cache/huggingface/hub/datasets--lighteval--mmlu/snapshots/31d46ab06e6934bb0d95f6918668716d1db6f921")


def mmlu_harness(line, task_name: str | None = None):
    topic = line["subject"]
    instruction = (
        "The following are multiple choice questions (with answers) about "
        f"{topic.replace('_', ' ')}.\n\n"
    )
    query = instruction + line["question"] + "\n"
    query += "".join(
        f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, line["choices"])
    )
    query += "Answer:"
    answer = line["answer"]
    gold_index = ascii_uppercase.index(answer) if isinstance(answer, str) else answer
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_index,
        instruction=instruction,
    )

SUBJECTS = (
    "abstract_algebra anatomy astronomy business_ethics clinical_knowledge "
    "college_biology college_chemistry college_computer_science college_mathematics "
    "college_medicine college_physics computer_security conceptual_physics econometrics "
    "electrical_engineering elementary_mathematics formal_logic global_facts "
    "high_school_biology high_school_chemistry high_school_computer_science "
    "high_school_european_history high_school_geography "
    "high_school_government_and_politics high_school_macroeconomics "
    "high_school_mathematics high_school_microeconomics high_school_physics "
    "high_school_psychology high_school_statistics high_school_us_history "
    "high_school_world_history human_aging human_sexuality international_law "
    "jurisprudence logical_fallacies machine_learning management marketing "
    "medical_genetics miscellaneous moral_disputes moral_scenarios nutrition "
    "philosophy prehistory professional_accounting professional_law "
    "professional_medicine professional_psychology public_relations security_studies "
    "sociology us_foreign_policy virology world_religions"
).split()

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mmlu_local:{subject}",
        prompt_function=mmlu_harness,
        hf_repo=str(LOCAL_MMLU),
        hf_subset=subject,
        hf_avail_splits=("test", "validation"),
        evaluation_splits=("test",),
        few_shots_split="validation",
        few_shots_select="sequential",
        generation_size=1,
        metrics=[Metrics.loglikelihood_acc],
        stop_sequence=["\n"],
        version=0,
    )
    for subject in SUBJECTS
]
