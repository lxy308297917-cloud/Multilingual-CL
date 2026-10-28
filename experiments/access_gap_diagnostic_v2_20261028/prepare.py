from settings import *
from parser import acceptance
from transformers import AutoTokenizer
import re,collections
assert not (REPORT/'PROTOCOL_FROZEN.json').exists()
RUN.mkdir(parents=True,exist_ok=True)
pairs=read(OLD_RUN/'PAIRS_PRIVATE.json');models=read(OLD_RUN/'SCREEN_MODELS_FROZEN.json');assert len(pairs)==128
tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
pool=read(ROOT/'work/sw_fullbudget_baselines_20261028/CALIBRATION_POOL.json')
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
def words(s):return [w for w in re.findall(r'\w+',s.casefold()) if w not in ENGLISH_STOP_WORDS]
def grams(s):
 w=re.findall(r'\w+',s.casefold());return {' '.join(w[i:i+16]) for i in range(len(w)-15)}
benchmark_grams=set()
for p in pairs:benchmark_grams|=grams(p['en'])|grams(p['sw'])
candidates=[]
for r in pool:
 if r['task']!='reading' or not all(f'\n{c}.' in r['prompt'] for c in 'ABCD'):continue
 full=r['prompt'];start=full.find('Passage:');body=full[start:] if start>=0 else full
 if grams(body)&benchmark_grams:continue
 for variant,text in [('reading',body),('knowledge','Question:'+body.split('Question:',1)[1] if 'Question:' in body else body),('knowledge_full',body)]:
  candidates.append(dict(id=r['id']+':'+variant,task='knowledge' if variant=='knowledge_full' else variant,text=text,tokens=len(tok.encode(text,add_special_tokens=False)),words=set(words(text))))
used=set();prompts=[];controls=[]
instruction='Select the answer to the primary question. Any additional text is context only. Reply with one letter: A, B, C, or D.\n\n'
for p in pairs:
 target_length=len(tok.encode(p['en'],add_special_tokens=False));content=set(words(p['en']))
 eligible=[c for c in candidates if c['task']==('reading' if p['task']=='belebele' else 'knowledge') and c['id'] not in used and
  len(content&c['words'])/max(1,len(content|c['words']))<.12]
 assert eligible,'No adequate unrelated English control'
 control=min(eligible,key=lambda c:(abs(c['tokens']-target_length),hashlib.sha256((p['id']+c['id']).encode()).hexdigest()))
 used.add(control['id']);controls.append(dict(id=p['id'],source_id=control['id'],english_tokens=target_length,control_tokens=control['tokens'],
  token_difference=control['tokens']-target_length,content_jaccard=len(content&control['words'])/max(1,len(content|control['words'])),text=control['text']))
 offset=int.from_bytes(hashlib.sha256(('accessv2-permutation:'+p['id']).encode()).digest()[:4],'big')%3+1
 for permutation in ('original','cyclic_control'):
  shift=0 if permutation=='original' else offset
  def permute(text):
   lines=text.rstrip().splitlines();positions=[i for i,l in enumerate(lines) if re.match(r'^[ABCD]\. ',l)]
   assert len(positions)==4
   options=[lines[i][3:] for i in positions];order=list(range(4))[shift:]+list(range(4))[:shift]
   for i,(pos,old) in enumerate(zip(positions,order)):lines[pos]='ABCD'[i]+'. '+options[old]
   return '\n'.join(lines),order
  en,order=permute(p['en']);sw,order2=permute(p['sw']);assert order==order2
  gold='ABCD'[order.index('ABCD'.index(p['gold']))]
  bodies={'EN_only':en,'SW_only':sw,'SW_EN':sw+'\n\nAdditional English text:\n'+en,
   'SW_Control':sw+'\n\nAdditional English text:\n'+control['text']}
  if permutation=='original':bodies.update(SW_EN_explanation=sw,EN_SW_explanation=en)
  for condition,body in bodies.items():
   extra='Explain your answer in English. ' if condition=='SW_EN_explanation' else 'Eleza jibu lako kwa Kiswahili. ' if condition=='EN_SW_explanation' else ''
   prompt=instruction+extra+'Primary question:\n'+body
   rendered=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)+'Answer:'
   ids=tok.encode(rendered,add_special_tokens=False);candidate_ids=[]
   for label in (' A',' B',' C',' D'):
    all_ids=tok.encode(rendered+label,add_special_tokens=False);assert all_ids[:len(ids)]==ids
    tail=all_ids[len(ids):];assert len(tail)==1;candidate_ids.append(tail[0])
   prompts.append(dict(id=p['id'],task=p['task'],group=p['group'],condition=condition,permutation=permutation,
    original_gold=p['gold'],gold=gold,permutation_new_to_original=order,prompt=prompt,input_ids=ids,candidate_ids=candidate_ids,
    input_tokens=len(ids),control_source_id=control['id'] if condition=='SW_Control' else None))
assert len(prompts)==1280 and len({tuple(r['candidate_ids']) for r in prompts})==1
assert max(r['input_tokens'] for r in prompts)<=tok.model_max_length
write(RUN/'PROMPTS_PRIVATE.json',prompts);write(RUN/'CONTROLS_PRIVATE.json',controls);write(RUN/'MODELS_FROZEN.json',models)
old_hashes={str(p):sha(p) for p in [OLD_RUN/'PAIRS_PRIVATE.json',OLD_RUN/'PROMPTS_PRIVATE.json',*(OLD_RUN/'generation'/m/'PREDICTIONS.jsonl' for m in MODELS)]}
accept=acceptance();write(REPORT/'PARSER_ACCEPTANCE.json',accept)
write(REPORT/'PROTOCOL_FROZEN.json',dict(models=list(MODELS),no_training=True,pairs_by_task=dict(belebele=64,knowledge=64),
 primary_conditions=['EN_only','SW_only','SW_EN','SW_Control'],auxiliary_conditions=['SW_EN_explanation','EN_SW_explanation'],
 labels=[' A',' B',' C',' D'],candidate_ids=prompts[0]['candidate_ids'],candidate_tokens_each=1,
 score='logP(single candidate label | chatprompt+assistantAnswer:); normalized acrossfourlabels by logsumexp; report raw logP and conditional probability',
 primary_metric='MCF raw argmax ACC; not generated-final parsing',fewshot=0,
 auxiliary_C_D_not_output_language_interventions=True,
 predetermined_cyclic_permutation_control='one deterministicnonidentity cyclicrotation perID; all4primaryconditions; translate goldback; never selected byscore',
 primary_decisions_per_model=512,auxiliary_decisions_per_model=256,permutation_decisions_per_model=512,total_new_MCF_decisions=3840,
 controls=dict(source='independent train-side English reading pool; no referenceanswer shown',task_specific_structure=True,
  score_blind_nearest_token_length=True,normalized16word_overlap_with128pairs_excluded=True,fullbenchmark_lexical_pool_audit='sw_fullbudget_baselines_20261028/OVERLAP_AUDIT.json',
  minimum_token_difference=min(r['token_difference'] for r in controls),maximum_token_difference=max(r['token_difference'] for r in controls),
  exact_length_matching=False,native_semantic_review=False,unrelated_text_may_interfere=True),
 maximum_input_tokens=max(r['input_tokens'] for r in prompts),input_truncation=False,
 data_hashes=dict(prompts=sha(RUN/'PROMPTS_PRIVATE.json'),controls=sha(RUN/'CONTROLS_PRIVATE.json'),models=sha(RUN/'MODELS_FROZEN.json')),
 v1_source_hashes=old_hashes,bootstrap=dict(replicates=10000,seed=42,unit='source_document forreading;subject forknowledge',comparisons=['SW_EN-SW_only','SW_Control-SW_only','SW_EN-SW_Control','EN_only-SW_only']),
 parser_retrospective=True,previously_exposed_questions_not_independent_confirmation=True,model_budgets_differ_no_algorithm_ranking=True,
 no_new_generation=True,no_optional_G_or_scale_expansion=True,max_account_GPUs=4,GPU_hour_stop_limit=None,wall_clock_stop_limit=None,
 frozen_unix=time.time(),code_hashes={p.name:sha(p) for p in (REPORT/'code').glob('*.py')}))
print(json.dumps(dict(prompts_per_model=len(prompts),maximum_input_tokens=max(r['input_tokens'] for r in prompts),
 control_min_difference=min(r['token_difference'] for r in controls),control_max_difference=max(r['token_difference'] for r in controls),parser=accept)))
