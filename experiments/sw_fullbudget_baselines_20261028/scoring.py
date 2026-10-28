"""Official strict instruction checks and corpus-level translation statistics."""
from settings import *
import importlib.util
import sys
import random
import numpy as np

_OFFICIAL = None

def instruction(row, text):
    global _OFFICIAL
    if _OFFICIAL is None:
        script = H/'reproductions/sw_d2_20260921/eval_project/ssu-main/evaluation/src/ifeval.py'
        spec = importlib.util.spec_from_file_location('direction_screen_official_ifeval', script)
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _, _, registry = mod.load_official_ifeval_modules(script.parent/'utils')
        _OFFICIAL = mod, registry
    mod, registry = _OFFICIAL
    py, np_state = random.getstate(), np.random.get_state()
    seed = int.from_bytes(hashlib.sha256(('strict:'+str(row['key'])).encode()).digest()[:4], 'big')
    try:
        random.seed(seed); np.random.seed(seed)
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        value = mod.test_instruction_following_strict(mod.InputExample(
            key=row['key'], instruction_id_list=row['instruction_id_list'],
            prompt=row['prompt'], kwargs=row['kwargs']), text, registry)
        return dict(prompt_strict=bool(value.follow_all_instructions),
                    instruction_pass=list(map(bool, value.follow_instruction_list)),
                    checker_seed=seed)
    finally:
        random.setstate(py); np.random.set_state(np_state)

def translation(text, reference):
    from sacrebleu.metrics import CHRF
    metric = CHRF(char_order=6, word_order=2, beta=2)
    stats = metric._compute_segment_statistics(metric._preprocess_segment(text),
                     metric._extract_reference_info([metric._preprocess_segment(reference)]))
    return dict(chrfpp_stats=list(stats), sentence_chrfpp=metric._compute_score_from_stats(stats).score)
