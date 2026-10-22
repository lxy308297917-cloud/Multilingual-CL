"""Describe fixed-token HF/native/dense probability differences; not math efficacy."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def difference(a,b):
    x=np.asarray([r['target_logprob'] for r in a]);y=np.asarray([r['target_logprob'] for r in b]);delta=y-x
    return {'prefixes':len(a),'mean_signed_second_minus_first_logprob':float(delta.mean()),'mean_absolute_logprob_difference':float(np.abs(delta).mean()),
        'max_absolute_logprob_difference':float(np.abs(delta).max()),'RMS_logprob_difference':float(np.sqrt(np.mean(delta**2))),
        'argmax_agreement_count':sum(x['argmax_token_id']==y['argmax_token_id'] for x,y in zip(a,b))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();h=a.root;code=Path(__file__).parent
    freeze=code/'engine_probability_probe_frozen_20261022.PUBLIC.json';f=read(freeze)
    state=read(h/'gpu-queue/queues/copsd-engine-probability-probe-v1/state.json');assert state['phase']=='completed' and state['job']['exit_code']==0
    job=state['job'];assert len(job['gpu_uuids'])==1 and job['config']['max_user_gpus']==4 and job['script_sha256']==f['code_sha256']['run_engine_probability_probe.sh']
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    inputs=read(h/f['input_relative']);assert sha(h/f['input_relative'])==f['input_sha256']
    results={};sources={};details={};seconds={}
    for stage in ['hf','native','dense']:
        path=h/f['output_relative']/(stage+'_DONE.PRIVATE.json');r=read(path)
        assert r['protocol']==f['protocol'] and r['input_sha256']==f['input_sha256'] and r['stage']==stage
        sources[stage]=sha(path);details[stage]=r['details'];seconds[stage]=r['elapsed_seconds_including_load']
        for condition,rr in r['records'].items():
            assert [x['key'] for x in rr]==[x['key'] for x in inputs] and len(rr)==32
            assert all(np.isfinite(x['target_logprob']) and x['target_logprob']<=1e-5 and np.isfinite(x['argmax_logprob']) for x in rr)
            results[stage+'_'+condition]=rr
    expected={'hf_Base','hf_LoRA','native_Base','native_LoRA','dense_Dense'};assert set(results)==expected
    groups={}
    for lang in ['all','sw','en']:
        indices=[i for i,x in enumerate(inputs) if lang=='all' or x['language']==lang]
        rr={k:[v[i] for i in indices] for k,v in results.items()}
        comparisons={name:difference(rr[first],rr[second]) for name,first,second in [
            ('HF_Base_vs_native_Base','hf_Base','native_Base'),('HF_LoRA_vs_native_LoRA','hf_LoRA','native_LoRA'),
            ('HF_LoRA_vs_dense_vLLM','hf_LoRA','dense_Dense'),('native_LoRA_vs_dense_vLLM','native_LoRA','dense_Dense')]}
        delta_hf=np.array([x['target_logprob']-y['target_logprob'] for x,y in zip(rr['hf_LoRA'],rr['hf_Base'])])
        delta_native=np.array([x['target_logprob']-y['target_logprob'] for x,y in zip(rr['native_LoRA'],rr['native_Base'])])
        error=delta_native-delta_hf
        comparisons['adapter_minus_Base_effect_comparison']={'prefixes':len(indices),'HF_mean_absolute_effect':float(np.abs(delta_hf).mean()),'native_mean_absolute_effect':float(np.abs(delta_native).mean()),
            'mean_absolute_effect_difference':float(np.abs(error).mean()),'max_absolute_effect_difference':float(np.abs(error).max()),
            'Pearson_effect_correlation':float(np.corrcoef(delta_hf,delta_native)[0,1]) if min(delta_hf.std(),delta_native.std())>1e-12 else None,
            'effect_sign_agreement_count':int(np.sum(np.sign(delta_hf)==np.sign(delta_native)))}
        groups[lang]=comparisons
    hours=(datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()/3600
    dummy=sum(r.get('dummy_generated_tokens',0) for rows in results.values() for r in rows)
    out={'observed_at':datetime.datetime.now().isoformat(),'protocol':f['protocol'],'frozen_sha256':sha(freeze),'parent_completed_exit0':True,'all32_prefix_keys_and160_probability_records_verified':True,
        'probability_comparison':groups,'actual_engine_details':details,'stage_seconds_including_load':seconds,'raw_stage_sha256':sources,
        'cost':{'unique_parent_GPU_hours_counted_once':hours,'dummy_generated_tokens':dummy,'training_updates':0,'new_math_benchmark_responses':0,'fixed_prefix_probability_records':160},
        'analysis_source_sha256':sha(Path(__file__)),'scope':f['scope'],
        'limitations':'32 correlated prefixes over4 shared originalquestions,1 SW20 checkpoint andshort fixed contexts. BF16 dense export is an existing CPUPEFT export, not a direct capture of active training-engine weights. HFSDPA andvLLM native/dense kernels/dtypes may differ. This cannot certify fullgeneration equivalence, all languages/checkpoints, teacher-loss correctness or math-effect causality. No arbitrary numerical pass/fail or benchmark regrading.'}
    with a.output.open('x') as z:json.dump(out,z,ensure_ascii=False,indent=2,allow_nan=False);z.write('\n')
    print(json.dumps({'all':groups['all'],'cost':out['cost']}))


if __name__=='__main__':main()
