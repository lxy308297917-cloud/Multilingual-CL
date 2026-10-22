"""Final bounded cost and completeness receipt, with limits preserved."""
import time,collections
from settings import *
assert all((RUN/'generation'/m/'DONE.json').exists() for m in MODELS)
assert all((RUN/'mechanism'/m/'DONE.json').exists() for m in ['Base','A','B','C','D','B-mid','C-mid','D-mid'])
generation={m:json.loads((RUN/'generation'/m/'DONE.json').read_text()) for m in MODELS}
mechanism={m:json.loads((RUN/'mechanism'/m/'DONE.json').read_text()) for m in ['Base','A','B','C','D','B-mid','C-mid','D-mid']}
teacher=json.loads((REPORT/'TEACHER_ACTUAL_BATCH_IDENTITY.json').read_text());assert teacher['all_original_shape_bitwise_equal']
assert all(v['rows']==192 for v in generation.values())
worker_runs=[]
for folder in [RUN/'mechanism_worker_receipts',RUN]:
    for startfile in folder.glob('WORKER_GPU-*.json'):
        donefile=startfile.with_name(startfile.name.replace('WORKER_','WORKER_DONE_'));assert donefile.exists(),str(donefile)
        s=json.loads(startfile.read_text());e=json.loads(donefile.read_text());worker_runs.append({'gpu_uuid':s['gpu_uuid'],'pid':s['pid'],'started_unix':s['started_unix'],'finished_unix':e['finished_unix'],'wall_gpu_hours':(e['finished_unix']-s['started_unix'])/3600})
wallhours=sum(x['wall_gpu_hours'] for x in worker_runs);assert wallhours<8
plan=json.loads((RUN/'PROTOCOL_FROZEN.json').read_text());assert max(x['started_unix'] for x in worker_runs)<plan['stop_new_jobs_unix'];assert max(x['finished_unix'] for x in worker_runs)<plan['hard_finish_unix']
sampler=rows(RUN/'physical_gpu_samples.jsonl');maxima={}
for item in sampler:
    for gpu in item.get('gpus',[]):maxima[gpu['uuid']]=max(maxima.get(gpu['uuid'],0),gpu['memory_used_mib'])
write(REPORT/'COMPLETION_AND_COSTS.json',{'completed_unix':time.time(),'new_scientific_training':0,'new_generation_rows':sum(v['rows'] for v in generation.values()),'same192_per_model':True,'all_five_models_complete':True,'source_role_overlap':0,'acceptance_cases':290,'gradient_probes':21,'gradient_model_states':7,'representation_models':5,'raw_artifact_verified_original_training_endpoints':7,'raw_original_development_endpoints_rehashed':9,'models':generation,'mechanism_costs':mechanism,'teacher_batch_identity':teacher,'worker_accounted_gpu_hours':wallhours,'generation_gpu_hours':sum(v['gpu_hours'] for v in generation.values()),'mechanism_and_representation_gpu_hours':sum(v.get('gpu_hours',v['wall_seconds']/3600) for v in mechanism.values())+teacher['gpu_hours'],'workers':worker_runs,'physical_generation_peak_mib_from_batches':max(v['physical_peak_mib_observed'] for v in generation.values()),'physical_sampler_observed_peak_mib':maxima,'physical_sampler_limits':'Sampler covers latter evaluation/identity block; per-model batch physical checks cover full individual evaluation. Earlier mechanism physical peaks were not sampled; torch allocator peaks are not physical peaks.','ceiling_gpu_hours':8,'all_jobs_started_before1630':True,'all_jobs_finished_before1715':True,'scorer_source_unchanged_since_acceptance':sha(pathlib.Path(__file__).parent/'scoring_v2.py')==json.loads((RUN/'ACCEPTANCE_FROZEN.json').read_text())['scorer_sha256'],'no_new_hyperparameter_selection':True,'original_v1_POSTHOC_preserved':True,'training_expansion_authorized':False})
receipt=json.loads((REPORT/'COMPLETION_AND_COSTS.json').read_text())
receipt.update(bounded_execution_and_reporting_complete=True,all_scientific_acceptance_conditions_met=False,validated_natural_math_automatic_scoring_obtained=False,independent_human_validation_obtained=False,math_validity='FAILED; original frozen numbers preserved, no ability ranking',assistant_final_answer_reviewed_occurrences=320,assistant_unique_response_reviews=281,future_uncompleted_targets_not_executed=['Reliable natural-response mathematical automatic scoring on unused independent validation','Independent human confirmation of source labels and assistant judgments'],no_more_training_or_generations_released=True)
write(REPORT/'COMPLETION_AND_COSTS.json',receipt)
print('Complete new audit; worker GPU hours',wallhours)
