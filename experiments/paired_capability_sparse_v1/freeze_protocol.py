"""Freeze data, budget, comparisons and downstream protocol before any new score."""
import hashlib,json,random
from pathlib import Path
from prepare_calibration import ROOT,H,sha

def main():
    destination=ROOT/'comparison_protocol.v1.json'
    if destination.exists():raise FileExistsError('Do not overwrite preregistration')
    dev=ROOT/'data/dev_v4';audit=json.loads((dev/'INDEPENDENT_AUDIT.json').read_text());manifest=json.loads((dev/'manifest.json').read_text())
    if audit['status']!='passed' or not all(audit['gates'].values()):raise RuntimeError('Development identity/isolation audit failed')
    cal_manifest=ROOT/'data/calibration_manifest.v1.json';cal_meta=json.loads(cal_manifest.read_text());cal_path=Path(cal_meta['path']);cal=[json.loads(s) for s in cal_path.read_text().splitlines()]
    if len(cal)!=29 or sha(cal_path)!=cal_meta['sha256']:raise RuntimeError('Calibration identity changed')
    boundary=ROOT/'data/TOKEN_BOUNDARY_AUDIT.json'
    if json.loads(boundary.read_text())['status']!='passed':raise RuntimeError('Teacher-forcing boundary audit not passed')
    frozen={str(p):sha(p) for p in [cal_manifest,cal_path,ROOT/'data/calibration_review.json',boundary,dev/'manifest.json',dev/'INDEPENDENT_AUDIT.json']}
    for entry in manifest['partitions'].values():frozen[entry['path']]=entry['sha256']
    for p,h in manifest['sources'].items():frozen[p]=h
    for p,h in manifest['formal_exclusion_sha256'].items():frozen[p]=h
    extra=H/'data/sw-d2-eval-inputs/data/eval_datasets_local/belebele/eng_Latn/test.jsonl';frozen[str(extra)]=audit['formal_english_reading_sha256']
    scorer=Path(manifest['scorer']);frozen[str(scorer)]=manifest['scorer_sha256']
    for p in (scorer.parent/'utils').glob('instructions*.py'):frozen[str(p)]=sha(p)
    recipe=H/'reproductions/sw_recipe_fft_20261010/data/aya80_mt20';rm=json.loads((recipe/'manifest.json').read_text());idx=json.loads((recipe/'indices.json').read_text());prefix=idx[:4096];wanted=set(prefix);found={}
    if sha(rm['source_file'])!=rm['source_sha256']:raise RuntimeError('Main source identity')
    with open(rm['source_file']) as stream:
        for i,line in enumerate(stream):
            if i in wanted:found[i]=json.loads(line)
            if len(found)==len(wanted):break
    counts={'input':0,'supervised':0};buckets={}
    for i in prefix:
        r=found[i];n=len(r['input_ids']);s=sum(v!=-100 for v in r['labels'][1:]);counts['input']+=n;counts['supervised']+=s;buckets.setdefault(r['bucket'],{'input':0,'supervised':0});buckets[r['bucket']]['input']+=n;buckets[r['bucket']]['supervised']+=s
    frozen[str(recipe/'manifest.json')]=sha(recipe/'manifest.json');frozen[str(recipe/'indices.json')]=sha(recipe/'indices.json');frozen[rm['source_file']]=rm['source_sha256']
    base=H/'models/Qwen2.5-1.5B-Instruct';source_audit=json.loads((ROOT.parents[1]/'docs/research/paired_capability_sparse/SOURCE_AUDIT.json').read_text())
    for p in list(base.glob('*.safetensors'))+[base/'config.json',base/'tokenizer.json',base/'tokenizer_config.json',base/'generation_config.json']:
        if str(p) in source_audit['files']:h=source_audit['files'][str(p)]['sha256']
        else:h=sha(p)
        frozen[str(p)]=h
    perm=list(range(len(cal)));random.Random(42).shuffle(perm)
    protocol={
      'status':'preregistered_before_new_mastery_or_development_scores; training_requires_separate_code_release',
      'date':'2026-10-21','scope':'single3090 Qwen2.5-1.5B-Instruct SW Aya80+MT20; actual-update sparse constraints; no OPD,merging,SVD,dynamic module budgets or multi-language sequence',
      'frozen_files':frozen,
      'training':{'steps':2048,'formal_schedule_total_steps':42465,'warmup_steps':2124,'learning_rate_peak':5e-5,'weight_decay':.01,'beta1':.9,'beta2':.999,'adam_epsilon':1e-8,'main_effective_batch':2,'main_microbatch':1,'max_length':2048,'main_loss':'per-example assistantCE then sample mean; original source IDs/labels/collator; no packing','seed':42,'fraction_per_matrix':.15,'mask_interval':20,'guard_gradient_refresh_steps':1,'mastery_refresh_steps':512,'calibration_pairs_per_update':2,'calibration_batch_per_language':2,'calibration_window':'cyclic fixed two-pair window(2*step+j)mod29; same main and calibration text exposure in all groups','reference_loss':'reviewed full English solution CE with EOS; identical output-language request for EN/SW input','replay_alpha':.1,'R_gradient':'main +.1*equal mean of EN/SW calibration CE, mixed in native BF16 before global normclip1; FP32 Adam moments','A_D_E_F_S_gradient':'main target only, normclip1; FP32 moments; calibration gradients independent','projection':'one global jointQP on mask support; epsilon0; FP64 sufficient statistics','rounding':'reject whole weight write if BF16 actual residual exceeds1e-12; retain one moment advance','main_prefix_indices_sha256':hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),'main_prefix_tokens':counts,'main_prefix_bucket_tokens':buckets,'resumable_checkpoint_steps':[1024,2048]},
      'comparisons':{'mandatory':['D','A','R'],'fixed_order':['D','A','R'],'D':'English mastered-reference update constraint','A':'original element sparsity, no protection','R':'matched exposure replay/auxiliary CE','conditional':['E','F','S'],'C_strong_baseline':'Required before any affirmative long-training novelty claim; old HFT/SSU endpoints are historical and not same-recipe prefix controls'},
      'mastery':{'sample_budget':'onegreedy+samples17,18,19 per source-language at each fixed snapshot','greedy_correct_and_sample_successes_required':2,'max_new_tokens':512,'temperature':.7,'top_p':1.,'top_k':0,'cache_batch_size':4,'Base_floor':'immutable','new_mastery':'two consecutive successful refreshes then latch; cannot revoke on one error','minimum_Base_EN_known':8,'language_diagnostic':'langdetect seed0, logged but not correctness gate'},
      'pairing':{'definition':'F doubles own-known reference weight when same-source partner is also known; E independent weights; S same F with fixed partner bijection','shuffled_partner_permutation':perm,'adequacy_joint_known':8,'adequacy_EN_only':8,'fixed_fork_step':1024,'Base_degeneracy_is_not_permanent_rejection':True,'gate':'Use only D1024 mastery, before any dev score. Require8joint+8ENonly. If BaseSWknown0, D/E/F/S paths are identical before1024 by acquisition hysteresis; reuse complete D1024 checkpoint only after verifying this. If no valid shared prefix, coldE/F/S require the same preregistered full2048 and remaining budget; never relabelD checkpoint. No search over fork points.','budget_gate':'Optional E/F/S only as a complete matched block if projected remaining physicalGPU cost incl all required dev fits total32; shared-prefix work counted once. Otherwise report this contribution unresolved, no long-training claim.'},
      'development':{'manifest':str(dev/'manifest.json'),'counts':{k:v['rows'] for k,v in manifest['partitions'].items()},'endpoints':['Base','A2048','D2048','R2048'],'optional_endpoints':['E2048','F2048','S2048'],'checkpoint_search':False,'generation':'greedy batch4, fixed caps from manifest; no generation hyperparameter search','max_new_tokens':manifest['generation_max_new_tokens'],'scoring':{'en_math':'canonical numeric equivalence, strict boxed secondary; no incidental last-number primary','en_reading':'option-letter strict primary and explicit-answer extraction secondary','en_instruction':'frozen IF-multi/IFEval-compatible strict prompt and instruction checks; fixed per-source scoring seed andlangdetect0; not formalIFEval','sw_reading_kenswquad':'Unicode normalized tokenF1 primary, exact match secondary; not Belebele','sw_summary':'sacrebleu chrF++ char6/word2 beta2','translation':'same chrF++; both directions; bootstrap clusters by original FLORES documentURL'},'bootstrap':{'resamples':10000,'seed':42,'paired_by':'shared source/question/document, no mixed-unit average'},'no_new_formal_test':True},
      'decision':{'first_stage':'D must improve a preregistered English retention primary vsA andR with paired95percentCI>0; SWreadingF1 lower bound -2points, summary/bidirectionalMT-1chrF++','retention_primaries':['en_math','en_instruction'],'pair_contribution':'F must exceed E and S and matched replay without SW noninferiority failure; pairing must change actual updates, not metadata only','multiple_comparisons':'report all predeclared task deltas and CIs; no selection of best checkpoint/proxy; no long-train approval from one small dev slice','confirmatory_requirement':'unused confirmation slice and strong baseline required before affirmative long-training upgrade','negative_or_inequivalent':'stop expansion; do not claim pairing superiority from D or weights alone'},
      'budget':{'total_gpu_hours_including_failed_jobs':32,'base_mastery_gpu_hours':.75,'per_training_gpu_hours':7.5,'per_development_model_gpu_hours':1.,'technical_ledger_already_spent_included':True,'no_infinite_retries_or_extra_checkpoints':True},
      'stop':{'BF16_rejections_first32_nonzero_lr':'>50percent stops spec','nonfinite_or_competing_compute':'exit and preserve logs','data_or_code_identity_change':'refuse launch','physical_memory_ceiling':'one24GB3090; no silentprecisionchange','time':'do not exceed frozen allocation; record partial work and no efficacy claim for incomplete matching'},
      'limits':['No new method originality claim for update projection itself (UP-MOPD precedent)','Full-reference CE and answer margin are capability proxies, not reasoning guarantees','Calibration SW semantic review is assistant-based, not independent nativegold','All FLORES dev EN/SW rows match published original; paragraphs grouped; known examples and formal source documents excluded','The main frozen NLLB training stream contains observed noise and was not rewritten','2048 is a formal warmup prefix, not a whole trained method or evidence of long-term forgetting']}
    destination.write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'protocol_sha256':sha(destination),'calibration_pairs':len(cal),'main_prefix_tokens':counts,'dev_samples':sum(protocol['development']['counts'].values()),'data_frozen_files':len(frozen),'budget':protocol['budget']},indent=2))
if __name__=='__main__':main()
