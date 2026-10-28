from settings import *
assert read(REPORT/'OVERLAP_AUDIT.json')['passed']
data=read(REPORT/'DATA_FROZEN.json')
budget=read(REPORT/'COMMON_BUDGET_20261028.json')
config=dict(model=str(BASE),model_safetensors_sha256=sha(BASE/'model.safetensors'),
 steps=data['steps'],main_input_tokens=data['main_input_tokens'],main_supervised_tokens=data['main_supervised_tokens'],
 learning_rate=5e-5,warmup_steps=2124,
 scheduler='full cosine covering all42465updates; initialzeroLR, endpointnextLRzero',effective_batch=2,micro_batch=1,gradient_accumulation=2,
 loss='mean of per-example shifted supervised-token mean CE',max_length=2048,
 Adam_betas=[.9,.999],Adam_eps=1e-8,weight_decay=.01,clip_norm=1.,weights='BF16',Adam_moments='FP32',
 seed=42,methods=list(METHODS),main_data_sha256=sha(REPORT/'DATA_FROZEN.json'),evaluation_sha256=sha(REPORT/'EVALUATION_FROZEN.json'),
 overlap_sha256=sha(REPORT/'OVERLAP_AUDIT.json'),
 candidate_scope='196Attention/MLP2D matrices,1310195712elements; Dense trains all unique model parameters including tiedembedding/head,norm,bias',
 mask_fraction=.15,common_budget_sha256=sha(REPORT/'COMMON_BUDGET_20261028.json'),common_selected_elements=budget['common_selected_elements'],common_actual_fraction=budget['common_actual_fraction'],element_mask_refresh=20,protected_calibration_refresh=200,protected_calibration_batch=32,protected_sensitivity_EMA=.9,
 protected_calibration='square of batch-mean EnglishCEgradient; CPUFP32 EMA; not certified mastery',
 SSU='static lowest Englishimportance wholecolumns;exactcommon196608000elements via proportionalfloorceil columnallocation;128examples same filteredpool',
 HFT15='historical static seeded whole-matrix closest-subset adaptation; not originalHFT50%',
 MoFO='official per-matrix magnitude threshold everyupdate;FP32moments; tiescanexceedfloor15%;globalcandidateweightdecay before sparseAdamadd',
 Element='exactcommon196608000elements;proportionalper-matrix integerquota;topmomentum;refresh20;maskeddecay',
 Protected='Element+normalized English squared-gradient sensitivity score;other settingsidentical',
 max_account_GPUs=4,GPU_hour_stop_limit=None,wall_time_stop_limit=None,
 checkpoints='every2000updates andendpoint;completeoptimizer/scheduler/RNG/data position; no scorebasedselection',
 short_preflight_only='all6methods real longestbatch+save/resume parity; never count in officialresults',
 protocol_revision='v2_userconfirmed_exact_common_budget',frozen_unix=time.time())
assert not (REPORT/'TRAINING_CONFIG_FROZEN.json').exists()
write(REPORT/'TRAINING_CONFIG_FROZEN.json',config)
print('Common full-budget configuration frozen')
