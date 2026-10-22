"""Freeze reviewed training-side pairs and independently checked full references."""
import ast,hashlib,json,re,unicodedata
from fractions import Fraction
from pathlib import Path
from mastery import extract_answer,numeric_value
ROOT=Path(__file__).resolve().parent
H=Path('/data0/users/lixiaoyang')
SOURCE=H/'Documents/Codex/2026-10-20/multilingual-continual-learning-github-https-github/work/gsm_bilingual_20261020/FROZEN_PROBE128.json'
SYSTEM='Solve the problem. Reason briefly in English and put the final numeric answer in \\boxed{...}.'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def arithmetic(text):
    def walk(n):
        if isinstance(n,ast.Expression):return walk(n.body)
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return Fraction(str(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):return walk(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1)
        if isinstance(n,ast.BinOp):
            a,b=walk(n.left),walk(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Div):return a/b
        raise ValueError('Unapproved arithmetic AST')
    return walk(ast.parse(text,mode='eval'))
def main():
    from datasets import Dataset
    audit=json.loads((ROOT.parents[1]/'docs/research/paired_capability_sparse/SOURCE_AUDIT.json').read_text())
    if sha(SOURCE)!=audit['files'][str(SOURCE)]['sha256']:raise RuntimeError('Reviewed pair source changed')
    review_path=ROOT/'data/calibration_review.json';review=json.loads(review_path.read_text())
    source_rows={x['id']:x for x in json.loads(SOURCE.read_text())}
    train_path=H/'data/sw-d2-eval-inputs/hf_cache/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow'
    train=Dataset.from_file(str(train_path));test=Dataset.from_file(str(train_path.with_name('gsm8k-test.arrow')))
    blocked={norm(x['question']) for x in test};rows=[]
    for qid,v in sorted(review['verified'].items(),key=lambda x:hashlib.sha256(('paired-cal-v1:'+x[0]).encode()).hexdigest()):
        old=source_rows[qid]
        if old['split']!='construction' or old['audit']['decision']!='provisionally_accept':raise RuntimeError('Invalid original split/review')
        if norm(train[old['source_index']]['question'])!=norm(old['question_en']) or norm(old['question_en']) in blocked:raise RuntimeError('Source identity/test overlap')
        gold=numeric_value(old['gold']);result=arithmetic(v['expression'])
        ref=v['reference'].replace('\\\\boxed','\\boxed')
        if result!=gold or extract_answer(ref)[0]!=gold:raise RuntimeError('Independent derivation/reference mismatch: '+qid)
        source_gold=train[old['source_index']]['answer'].rsplit('####',1)[-1].strip()
        if numeric_value(source_gold)!=gold or gold.denominator!=1:raise RuntimeError('Source gold mismatch')
        wrong=str(int(gold)+1)
        if numeric_value(wrong)==gold:raise RuntimeError('Equivalent distractor')
        rows.append({'id':qid,'group_id':qid,'split':'calibration','source_index':old['source_index'],
          'question_en':old['question_en'],'question_sw':old['question_sw'],'gold':str(int(gold)),
          'reference_answer':ref,'reference_output_language':'en','negative_answer':wrong,
          'negative_verification':'unique integer numeric gold +1; exact rational inequivalence checked',
          'derivation':v['expression'],'derivation_exact':str(result),'original_source_solution_en':old['worked_answer_en'],
          'sw_review':old['audit'],'native_swahili_gold':False})
    if set(review['excluded'])&set(review['verified']) or not 8<=len(rows)<=32:raise RuntimeError('Invalid calibration cardinality')
    target=ROOT/'data/calibration_pairs.v1.jsonl';text=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)
    if target.exists():
        if target.read_text()!=text:raise RuntimeError('Refuse to overwrite frozen calibration pairs')
    else:target.write_text(text)
    manifest={'status':'frozen_source_and_references; mastery/overlap_with_final_dev_pending','source':str(SOURCE),'source_sha256':sha(SOURCE),
      'review_sha256':sha(review_path),'path':str(target),'sha256':sha(target),'rows':len(rows),'ids':[x['id'] for x in rows],
      'system':SYSTEM,'mastery_rule':'1greedy+3samples; greedy correct AND >=2/3 sampled correct; capped/unparsed fail',
      'generation':{'sample_seeds':[17,18,19],'temperature':.7,'top_p':1.,'top_k':0,'max_new_tokens':512},
      'selection':'all reviewed construction pairs minus explicit interpretation ambiguities; independent of new model scores',
      'excluded':review['excluded'],'protected_language_first_stage':'en',
      'reference_policy':'independently rederived concise full English solution with final numeric box; same output-language policy for both input languages',
      'calibration_max_new_tokens_base':len(rows)*2*4*512,
      'limitations':['SW semantics reuse previous assistant review, not independent native gold','Calibration only; no formal MGSM/PolyMath test question','No default margin advantage; full-reference CE and answer-margin scored separately before efficacy training'],
      'pair_mechanism_F_status':'stopped; no distinct paired efficacy claim from D'}
    path=ROOT/'data/calibration_manifest.v1.json';content=json.dumps(manifest,indent=2,ensure_ascii=False)+'\n'
    if path.exists() and path.read_text()!=content:raise RuntimeError('Refuse to overwrite immutable manifest')
    if not path.exists():path.write_text(content)
    print(json.dumps({'rows':len(rows),'sha256':manifest['sha256'],'excluded':list(review['excluded']),'base_generation_token_cap':manifest['calibration_max_new_tokens_base']},indent=2))
if __name__=='__main__':main()
