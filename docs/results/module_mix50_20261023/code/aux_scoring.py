"""Development-only metrics with explicit extraction and source-cluster identity."""
import collections,hashlib,importlib.util,random,re,sys,unicodedata
from pathlib import Path
import numpy as np
from mastery import extract_answer,numeric_value
SCORER=Path('/srv/multilingual/data/sw-d2-eval-inputs/project/ssu-main/evaluation/src/ifeval.py')
_IF=None

def normalized_tokens(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    return ''.join(' ' if unicodedata.category(c).startswith(('P','S')) else c for c in text).split()
def token_scores(pred,reference):
    a,b=normalized_tokens(pred),normalized_tokens(reference)
    common=sum((collections.Counter(a)&collections.Counter(b)).values())
    return {'token_f1':float(2*common/(len(a)+len(b))) if a or b else 1.,'exact_match':float(a==b)}
def instruction_score(row,text):
    global _IF
    py,nps=random.getstate(),np.random.get_state();seed=int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:4],'big')
    try:
        if _IF is None:
            spec=importlib.util.spec_from_file_location('paired_frozen_ifeval',SCORER);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            _,_,registry=module.load_official_ifeval_modules(SCORER.parent/'utils');_IF=module,registry
        module,registry=_IF
        from langdetect import DetectorFactory
        DetectorFactory.seed=0;random.seed(seed);np.random.seed(seed)
        example=module.InputExample(key=seed,instruction_id_list=row['instruction_id_list'],prompt=row['prompt'],kwargs=row['kwargs'])
        value=module.test_instruction_following_strict(example,text,registry)
        return {'prompt_strict':float(value.follow_all_instructions),'instruction_pass':list(map(bool,value.follow_instruction_list)),'scoring_seed':seed}
    finally:random.setstate(py);np.random.set_state(nps)
def score_record(partition,row,text):
    if partition=='en_math':
        value,parser=extract_answer(text);gold=numeric_value(row['reference'])
        if gold is None:raise ValueError('Unsupported frozen numeric gold')
        correct=value is not None and value==gold
        return {'numeric_equivalence':float(correct),'boxed_equivalence':float(correct and parser=='boxed'),'parsed':str(value) if value is not None else None,'parser':parser}
    if partition=='en_reading':
        strict=text.strip().upper();m=re.search(r'(?:answer|option)\s*(?:is\s*)?[:：]?\s*\(?([A-D])\)?(?:[.\s]|$)',text,re.I)
        explicit=(m[1].upper() if m else strict if strict in 'ABCD' and len(strict)==1 else None)
        return {'strict_letter':float(strict==row['reference']),'explicit_letter':float(explicit==row['reference']),'parsed_letter':explicit}
    if partition=='sw_reading_kenswquad':return token_scores(text,row['reference'])
    if partition=='en_instruction':return instruction_score(row,text)
    if partition in ('translation','sw_summary'):
        from sacrebleu.metrics import CHRF
        metric=CHRF(char_order=6,word_order=2,beta=2)
        stats=metric._compute_segment_statistics(metric._preprocess_segment(text),metric._extract_reference_info([metric._preprocess_segment(row['reference'])]))
        return {'chrfpp_sentence':metric._compute_score_from_stats(stats).score,'chrfpp_stats':stats}
    raise ValueError(partition)

def summarize(records):
    groups={}
    for r in records:
        label=r['task'];groups.setdefault(label,[]).append(r)
    result={}
    for task,rows in groups.items():
        s=rows[0]['scores'];n=len(rows)
        if 'chrfpp_stats' in s:
            from sacrebleu.metrics import CHRF
            metric=CHRF(char_order=6,word_order=2,beta=2);stats=np.array([r['scores']['chrfpp_stats'] for r in rows]).sum(axis=0).tolist()
            score={'corpus_chrfpp':metric._compute_score_from_stats(stats).score,'metric_signature':'chrF2++|nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|beta:2'}
        else:
            keys={'en_math':['numeric_equivalence','boxed_equivalence'],'en_reading':['strict_letter','explicit_letter'],'en_instruction':['prompt_strict'],'sw_reading_kenswquad':['token_f1','exact_match']}[task]
            score={k:100*sum(r['scores'][k] for r in rows)/n for k in keys}
            if task=='en_instruction':score['instruction_strict']=100*sum(sum(r['scores']['instruction_pass']) for r in rows)/sum(len(r['scores']['instruction_pass']) for r in rows)
        result[task]={**score,'n':n,'capped':sum(r['capped'] for r in rows),'generated_tokens':sum(r['generated_tokens'] for r in rows)}
    return result
