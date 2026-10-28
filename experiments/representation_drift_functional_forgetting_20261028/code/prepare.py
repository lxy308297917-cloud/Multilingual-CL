"""Freeze score-blind diagnostic samples, prompts, groups and analysis choices."""
from pathlib import Path
import json,hashlib,unicodedata,re,sys,time
ROOT=Path(__file__).resolve().parents[3];H=Path('/data0/users/lixiaoyang')
REPORT=ROOT/'docs/results/representation_drift_functional_forgetting_20261028'
WORK=ROOT/'work/representation_drift_functional_forgetting_20261028'
SOURCES=ROOT/'work/sw_fullbudget_baselines_20261028/evaluation_inputs'
BASE=H/'models/Qwen2.5-1.5B-Instruct'
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
def read(name):return [json.loads(x) for x in (SOURCES/(name+'.jsonl')).open()]
def split(group):return 'discovery' if int(digest('drift-group-20261028:'+group)[:8],16)%2==0 else 'confirmation'
def choose(rows,n):
 return sorted(rows,key=lambda r:digest('drift-sample-42:'+r['id']))[:n]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert not (REPORT/'PROTOCOL_FROZEN.json').exists()
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
examples=[]
for task in ('belebele','knowledge'):
 en,sw=read(task+'_en'),read(task+'_sw');assert len(en)==len(sw)
 if task=='belebele':
  key=lambda r:r['link']+'#'+str(r['question_number'])
 else:key=lambda r:r['id']
 sidx={key(r):r for r in sw};assert len(sidx)==len(sw)
 candidates=[]
 for row in en:
  rid=key(row);assert rid in sidx
  # Source document or subject kept wholly on one side, not individual questions.
  group=task+':'+(row['link'] if task=='belebele' else row['subject'])
  candidates.append(dict(id=task+':'+rid,group=group,phase=split(group),row=row,sw=sidx[rid]))
 for phase in ('discovery','confirmation'):
  eligible=[r for r in candidates if r['phase']==phase];assert len(eligible)>=128
  for item in choose(eligible,128):
   for lang in ('en','sw'):
    r=item['row'] if lang=='en' else item['sw']
    if task=='belebele':
     content=r['flores_passage']+'\n\n'+r['question']+'\n'+''.join(f'{a}. {r["mc_answer"+str(i+1)]}\n' for i,a in enumerate('ABCD'))
     gold=int(r['correct_answer_num'])-1
    else:
     content=r['question']+'\n'+''.join(f'{a}. {v}\n' for a,v in zip('ABCD',r['choices']));gold='ABCD'.index(r['answer'])
    examples.append(dict(id=item['id']+':'+lang,pair_id=item['id'],group=item['group'],phase=phase,
     task=task,language=lang,content=content,gold_index=gold,source_row_id=key(r)))
math=[]
for index,r in enumerate(read('gsm8k')):
 group='gsm8k:'+digest(re.sub(r'\d+(?:[.,]\d+)*','<n>',norm(r['question'])))
 math.append(dict(id='gsm8k:'+str(index),pair_id=None,group=group,phase=split(group),task='gsm8k',language='en',
  content=r['question'],reference_answer=r['answer'],source_row_id=index))
for phase in ('discovery','confirmation'):
 eligible=[r for r in math if r['phase']==phase];assert len(eligible)>=64
 examples+=choose(eligible,64)
label_ids=[tok.encode(' '+a,add_special_tokens=False) for a in 'ABCD'];assert all(len(x)==1 for x in label_ids)
for r in examples:
 instruction=('Solve the problem. End with a line of the form #### followed by the final numerical answer.'
  if r['task']=='gsm8k' else 'Answer the multiple-choice question. Choose exactly one of A, B, C, D.')
 text=instruction+'\n\n'+r['content']
 rendered=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True)
 if r['task']!='gsm8k':rendered+='Answer:'
 enc=tok(rendered,add_special_tokens=False,return_offsets_mapping=True)
 first=rendered.index(r['content']);last=first+len(r['content'])
 indices=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>a and a>=first and b<=last]
 assert indices and len(enc['input_ids'])<=4096
 r.update(input_ids=enc['input_ids'],attention_mask=[1]*len(enc['input_ids']),content_indices=indices,
  last_content_position=indices[-1],decision_position=len(enc['input_ids'])-1,
  prompt_sha256=digest(rendered),input_sha256=digest(json.dumps(enc['input_ids'])))
private=WORK/'INPUTS_PRIVATE.jsonl';private.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples))
assert len(examples)==1152
assert not {r['group'] for r in examples if r['phase']=='discovery'} & {r['group'] for r in examples if r['phase']=='confirmation'}
protocol=dict(version=1,frozen_unix=time.time(),models=['Base','Dense','Element','Protected'],
 adapted_model_requirement=dict(steps=42465,main_input_tokens=15360935,main_supervised_tokens=5984371,
  training_config_sha256=sha(ROOT/'outputs/sw_fullbudget_baselines_20261028/TRAINING_CONFIG_FROZEN.json')),
 supplementary_models_not_queued=['SSU','HFT15','MoFO'],no_training=True,
 paper=dict(title='Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models',
  pdf_sha256=sha(H/'.codex/attachments/66f2365c-a2a4-4edc-a000-7a0d6533f99d/Aggarwal 等 - 2026 - Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language.pdf'),
  read_sections=['4.4','5.4','Table 5','Figures 1-3','Future Work'],
  formula_caveat='Paper X is layer x feature after sample averaging, giving one feature covariance across layers; no unique per-layer curve follows from this formula alone. Do not invent its curve implementation.'),
 input_manifest=dict(path=str(private),sha256=sha(private),n=1152,
  tasks={'belebele_en':256,'belebele_sw':256,'knowledge_en':256,'knowledge_sw':256,'gsm8k_en':128},
  split='128/source-task-language per half for MC; 64 math per half; documents/subjects/normalized numeric-skeleton groups disjoint',
  all_materials_exposed='Existing benchmark pools; development mechanism diagnostic, not untouched final confirmation',
  no_train_or_protection_use=True,exact_near_duplicate_limit='Source grouping and normalized numeric skeletons; no claim of exhaustive semantic deduplication'),
 prompt_protocol=dict(fewshot=0,chat_template_sha256=sha(BASE/'tokenizer_config.json'),
  MCF='full-sequence BF16 logits, four single-token space-prefixed labels, assistant prefill Answer:',
  label_token_ids=[x[0] for x in label_ids],math='greedy 0-shot, max_new_tokens=1024; explicit #### numerical answer',
  benchmark_caveat='Diagnostic prompts differ from formal Belebele 3-shot/MMLU 5-shot/GSM8K 5-shot. State transitions refer to THIS frozen protocol; never merge into formal benchmark table.'),
 extraction=dict(dtype='bfloat16',reduction_dtype='float32',analysis_dtype='float64',batch_size=1,
  attention='sdpa',use_cache=False,max_input_tokens=4096,truncation=False,layers=list(range(28)),
  boundary='raw residual stream immediately after each decoder block, before final model norm',
  primary='decision_position = last effective input token, including assistant prefill',
  sensitivities=['last_content_position','mean over content_indices; exclude chat/instruction template'],
  last_input_equals_decision_not_independent=True,storage='Only 3 pooled vectors per layer per sample; no full-token caches',
  tensor_shape='N=1152 x L=28 x P=3 x D=1536, float32 on CPU'),
 metrics=dict(cosine_drift='1-dot(b,a)/(norm(b)*norm(a)), epsilon=1e-12',
  normalized_l2='norm(a-b)/max(norm(b),1e-12)',activation_norm_ratio='norm(a)/max(norm(b),1e-12)',
  layer_covariance='For each fixed task/language/phase set, X is N x D; C=(X-mean(X)).T@(X-mean(X))/(N-1); report ||C_adapt-C_base||F/max(||C_base||F,1e-12), float64, via exact sample Gram identities',
  paper_style='Average representations over fixed samples to get L x D; center across L and compute covariance with denominator L-1; report separate global scalar, not a guessed layerwise reproduction'),
 analysis=dict(primary='EN Base-correct: forgotten vs retained, mean decision-position cosine drift across all 28 blocks',
  controls=['task one-hot','log input length','Base gold-label probability and margin for MC','Base activation norm','adapted/Base activation norm ratio'],
  math_analysis='Separate task; no token likelihood treated as probability of full reasoning correctness. Exclude unparsed outputs from F/R, report counts and strict-format rate separately.',
  model='logistic regression C=1, standardize on discovery only; fixed group-disjoint confirmation; no regularization scan',
  comparisons=['controls only','controls plus drift','controls plus activation norm change'],
  parameter_norm_limit='Global update norms are constant within each model, only 3 adapted models; descriptive cross-model comparison, no independent-sample regression claim',
  bootstrap_replicates=2000,bootstrap_seed=42,unit='source group; same sampled groups for paired comparisons',
  multiplicity='Primary mean-all-layer analysis predeclared; all 28 layer curves reported, layerwise p-values exploratory with BH-FDR',
  directional='Residual change projected onto fixed Base correct-vs-highest-incorrect unembedding contrast; exploratory, only MC, layer coordinate caveat'),
 patching_gate=dict(minimum_per_class_per_split=20,
  primary_MC_only=True,required_models=2,of_models=['Dense','Element','Protected'],
  confirmation_delta_AUROC_at_least=.05,paired_bootstrap95_lower_bound_gt=0,
  contradiction_rule='No other adequately powered model has delta AUROC CI wholly below zero',
  inadequate_samples='Evidence insufficient; no data expansion or automatic patching',
  no_stable_association='Stop patching, report negative/uncertain evidence'),
 patching_if_gate_passes=dict(models='Only models passing individual gate; no supplemental models',
  candidate_block=13,noncandidate_control_block=0,position='decision_position',alpha=1.,
  conditions=['unmodified','same-input Base activation','Base activation at fixed noncandidate block0','seed42 random direction with per-example matched L2 displacement at block13'],
  inputs='All fixed confirmation MC samples, both EN and SW; masks/tokens identical; use_cache=False',
  retained_and_acquired_side_effects=True,report='English recovered/newly lost, SW acquired lost, paired source intervals; no claim of unique knowledge storage',
  max_new_forward_passes=3*512*4,math_generation_not_patched=True),
 tsne=dict(layers=[0,13,27],position='decision_position',subset='first32 hash-ranked confirmation examples per task/language among MC, same IDs across four models',
  joint_fit=True,standardization='joint feature mean/std, then joint PCA50',perplexity=30,learning_rate=200,
  init='pca',iterations=1000,seeds=[42,43],role='auxiliary only; report instability, never choose best seed or layer'),
 execution=dict(max_account_GPUs=4,dedicated_uuid='GPU-1a3a5c22-63f7-8edb-d112-4458bb5baa58',
  no_time_stop_threshold=True,formal_endpoint_only=True,no_checkpoint_selection=True,
  no_training_or_hyperparameter_scan=True))
write(REPORT/'PROTOCOL_FROZEN.json',protocol)
write(REPORT/'SOURCE_IDENTITIES.json',dict(protocol_sha256=sha(REPORT/'PROTOCOL_FROZEN.json'),base_weights_sha256=sha(BASE/'model.safetensors'),
 input_sha256=sha(private),source_files={str(p):sha(p) for p in sorted(SOURCES.glob('*.jsonl'))},
 model_endpoints_pending=['Dense','Element','Protected']))
print('Frozen',len(examples),'inputs; max tokens',max(len(r['input_ids']) for r in examples))
