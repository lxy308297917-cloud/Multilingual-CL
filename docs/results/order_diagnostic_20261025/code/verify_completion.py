"""Read-only requirement audit after both midpoint evaluations; never generates answers."""
from settings import *
import math
assert (RUN/'SERIES_DONE.json').exists() and (RUN/'FINALIZATION_DONE.json').exists()
assert not (RUN/'SERIES_FAILED.json').exists() and not (RUN/'FINALIZATION_FAILED.json').exists()
source=read(REPORT/'SOURCE_FROZEN.json');assert all(sha(REPORT/'code'/n)==h for n,h in source['code'].items())
for name,digest in read(PRIOR/'SOURCE_FROZEN.json')['code'].items():assert sha(PRIOR/'code'/name)==digest
p=read(REPORT/'PROTOCOL_FROZEN.json');assert p['new_responses_ceiling']==512 and p['GPU_context_hours_ceiling']==2 and p['no_training']
meta=read(RUN/'CONFIRMATION_FROZEN.json');assert sha(meta['data_path'])==p['data_sha256'];rr=rows(meta['data_path']);ids={r['id'] for r in rr};assert len(rr)==len(ids)==256
models=read(RUN/'MODELS_FROZEN.json');total=0;generated=0;prompt=0;checks={};prior_scorers=read(PRUN/'CONFIRMATION_FROZEN.json')['scorer_hashes']
# Load exact frozen scorer functions in CPU context; no model or CUDA initialization.
sys.path.insert(0,str(PRIOR/'code'));sys.path.append(str(ROOT/'outputs/Multilingual-CL-alignment-recovery-v3/experiments/paired_capability_sparse_v1'))
from reading_scoring import sw_score,reading_score,behavior
from aux_scoring import score_record
byid={r['id']:r for r in rr}
for seed in ('42','43'):
 name=f'LR077-s{seed}-mid';folder=RUN/'evaluation/confirmation'/name;cfg=read(folder/'CONFIG.json');done=read(folder/'DONE.json');cached=PRUN/'evaluation/confirmation'/f'LR077-s{seed}';ecfg=read(cached/'CONFIG.json')
 assert done['rows']==done['scored_rows']==256 and done['data_sha256']==meta['data_sha256'];assert sha(folder/'PREDICTIONS.jsonl')==done['predictions_sha256']
 assert sha(Path(models[name]['model'])/'model.safetensors')==models[name]['sha256']==done['model_sha256']==cfg['model_sha256']
 assert cfg['caps']==ecfg['caps']==meta['caps'] and cfg['batch']==ecfg['batch']==4
 assert cfg['scorer_hashes']==ecfg['scorer_hashes']==prior_scorers and cfg['code_sha256']==ecfg['code_sha256']==sha(PRIOR/'code/evaluate.py')
 # Model weights/config identity changes at step2146; generation defaults must still be identical.
 assert cfg['generation_config']==ecfg['generation_config']
 pred=rows(folder/'PREDICTIONS.jsonl');assert len(pred)==len({v['id'] for v in pred})==256 and {v['id'] for v in pred}==ids
 expected=[r['id'] for task in ('sw_reading','en_reading','translation','en_instruction') for r in rr if r['study_partition']==task];assert [v['id'] for v in pred]==expected
 for v in pred:
  row=byid[v['id']];assert v['model']==name and v['group_id']==row['group_id'] and v['score_eligible']
  assert v['study_partition']==row['study_partition'] and v['generated_tokens']==len(v['token_ids'])
  assert v['generated_tokens']<=meta['caps'][v['study_partition']]
  if v['study_partition']=='sw_reading':s=sw_score(v['text'],row['reference'])
  elif v['study_partition']=='en_reading':s=reading_score(v['text'],row['reference'],row['options'])
  else:s=score_record(v['study_partition'],row,v['text'])
  assert s==v['scores'];assert behavior(v['text'])==v['behavior']
 total+=len(pred);generated+=sum(v['generated_tokens'] for v in pred);prompt+=sum(v['prompt_tokens'] for v in pred)
 assert done['counts']=={'prompt_tokens':sum(v['prompt_tokens'] for v in pred),'generated_tokens':sum(v['generated_tokens'] for v in pred)}
 checks[name]={'rows':256,'data_and_current_model_hash_verified':True,'all256_scores_CPU_recomputed_exact':True,'caps_batch_generation_config_and_scorers_match_cached_endpoint':True,'row_order_and_ids_verified':True,'DONE_sha256':sha(folder/'DONE.json'),'prediction_sha256':sha(folder/'PREDICTIONS.jsonl'),'peak_allocated_bytes':done['peak_allocated_bytes'],'physical_peak_mib_observed':done['physical_peak_mib_observed']}
seconds,active=usage();assert total==512 and seconds<=7200 and active==0
result=read(REPORT/'RESULTS.json');assert not result['missing'] and len(result['scores'])==4 and len(result['paired_comparisons'])==4
assert all((REPORT/n).exists() for n in ['DATA_COMPOSITION.png','MIDPOINT_COMPARISON.png','README_ZH.md','SW_OUTPUT_AUDIT.json','DATA_ORDER_AUDIT.json','CASE_REVIEW_ZH.json'])
write(REPORT/'COMPLETION_VERIFIED.json',{'passed':True,'verified_unix':time.time(),'new_answers':total,'new_prompt_tokens':prompt,'new_generated_tokens':generated,'GPU_context_hours':seconds/3600,'GPU_context_hours_ceiling':2,'new_training_updates':0,'current_sources_match_freezes':True,'both2146_models_verified_against_cached4293_protocol':checks,'data_stream_and_four_checkpoint_fullhashes_verified_in_CPU_audit':read(RUN/'AUDIT_DONE.json')['passed'],'no_checkpoint_selection_or_formal_test':True,'all_required_artifacts_present':True,'needs_scientific_readthrough_and_figure_review':True})
print('Completion requirement audit passed; scientific interpretation still requires review')
