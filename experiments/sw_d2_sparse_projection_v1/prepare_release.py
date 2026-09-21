"""Freeze five SW D2 methods, fail closed when storage is below suite reserve."""
import hashlib,json,shutil,datetime
from pathlib import Path
ROOT=Path(__file__).parent;PROJ=ROOT.parent.parent/'ssu-main';TMP=Path('/root/autodl-tmp/sw_d2_sparse_projection_v1')
MAN=Path('/root/autodl-tmp/d2_bilingual_v1/sw/manifest.json');CAL=Path('/root/autodl-tmp/sw_update_utility_v1/calibration_candidates.jsonl')
BASIS=TMP/'edge_bases_rank32.safetensors';SEL=ROOT.parent/'d2_bilingual_v1/D2_METHOD_TRACK_SELECTION.json'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads(MAN.read_text());selection=json.loads(SEL.read_text())
assert selection['selected_language']=='SW' and selection['selected_recipe']=='D2'
assert manifest['budget_nonpadding_input_tokens']==15360372 and manifest['max_length']==2048 and manifest['sample_count']==44292
for kind in ('train_pool','dev_pool','indices'):assert sha(manifest[kind])==manifest[kind+'_sha256']
assert sha(BASIS)==json.loads((ROOT/'BASIS_BUILD.json').read_text())['basis_sha256']
audit=json.loads((ROOT/'D2_CALIBRATION_OVERLAP_AUDIT.json').read_text());assert audit['calibration_sha256']==sha(CAL)
assert not audit['overlapping_calibration_ids']
methods=['element_protected','element_unprotected','edge_projected_rows','edge_rows_no_projection','fft_fp32_matched']
override_path=ROOT/'USER_STAGED_STORAGE_OVERRIDE.json'
first_pair=['element_protected','element_unprotected']
user_override=json.loads(override_path.read_text()) if override_path.exists() else None
if user_override:assert user_override['scope']=='first pair only: element_protected and element_unprotected'
full_release=shutil.disk_usage(TMP).free>=105*2**30
configs={};cfg_dir=ROOT/'configs';cfg_dir.mkdir(exist_ok=True)
for method in methods:
 cfg={'experiment_family':'sw_d2_sparse_projection_v1','protocol_id':'sw_d2_sparse_projection_seed42_v1',
      'status':('frozen_staged_release' if user_override and method in first_pair else 'frozen_full_release' if full_release else 'frozen_pending_storage_release'),'recipe':'D2','recipe_selection':str(SEL),
      'manifest':str(MAN),'manifest_sha256':sha(MAN),'checkpoint_root':str(TMP/'checkpoints'),
      'output_dir':str(TMP/'checkpoints'/method),'base_model':'/root/models/Qwen2.5-1.5B-Instruct',
      'seed':42,'input_token_budget':15360000,'max_length':2048,'learning_rate':5e-5,'weight_decay':.01,
      'training_python':'/root/miniconda3/envs/cl/bin/python','training_environment':{'python':'3.10.20','torch':'2.5.1+cu121','transformers':'4.57.6','datasets':'4.8.5','accelerate':'1.7.0'},
      'minimum_start_free_bytes':20*2**30,'calibration_file':str(CAL),'calibration_sha256':sha(CAL),
      'basis_file':str(BASIS),'basis_sha256':sha(BASIS),'method':method}
 path=cfg_dir/(method+'.json');path.write_text(json.dumps(cfg,indent=2)+'\n');configs[str(path.resolve())]=sha(path)
code_paths=[ROOT/x for x in ['method_optimizers.py','activation_bases.py','train_method.py','evaluate_method.py']]
code_paths +=[PROJ/'scripts'/x for x in ['train_cl.py','evaluate_cl.py']]
old=ROOT.parent/'sw_update_utility_v1/prototype'
code_paths +=[old/x for x in ['safe_state.py','safe_resume.py','source_gradients.py','raw_gradient_capture.py','dense_optimizer.py','block_ops.py']]
base=Path('/root/models/Qwen2.5-1.5B-Instruct')
base_paths=[base/x for x in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']]
evidence_paths=[MAN,CAL,BASIS,SEL,ROOT/'BASIS_BUILD.json',ROOT/'D2_CALIBRATION_OVERLAP_AUDIT.json']
evidence_paths +=[Path(manifest[k]) for k in ('train_pool','dev_pool','indices')]
if user_override:evidence_paths.append(override_path)
required=105*2**30;stage_required=(105 if full_release or not user_override else 65)*2**30;free=shutil.disk_usage(TMP).free
release_configs={p:h for p,h in configs.items() if full_release or not user_override or Path(p).stem in first_pair}
record={'status':'released' if free>=stage_required else 'blocked_storage','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'protocol_id':'sw_d2_sparse_projection_seed42_v1','methods':methods if full_release or not user_override else first_pair,'future_methods':[] if full_release or not user_override else methods[2:],'configs':release_configs,
        'code':{str(p):sha(p) for p in code_paths},'base_files':{str(p):sha(p) for p in base_paths},
        'evidence':{str(p):sha(p) for p in evidence_paths},'free_bytes':free,
        'suite_required_free_bytes':required,'current_stage_required_free_bytes':stage_required,'deficit_bytes':max(0,stage_required-free),'full_suite_deficit_bytes':max(0,required-free),'user_staged_override':str(override_path) if user_override else None,
        'reason':'Full five-model release at >=105 GiB free, retaining first-pair config identity.' if full_release else 'Explicit user-authorized first-pair launch at >=65 GiB free; full five-model suite still requires 105 GiB. No historical deletion or link workaround.' if user_override else 'Five final weights and FP32 recovery checkpoints, protected-model sensitivity state, serialized checkpoint rotation, and >=10 GiB margin; no historical deletion or link workaround.'}
(ROOT/'TRAINING_RELEASE.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'free_gib':free/2**30,'required_gib':stage_required/2**30,'deficit_gib':record['deficit_bytes']/2**30,'full_suite_deficit_gib':record['full_suite_deficit_bytes']/2**30,'configs':len(release_configs)}))
