"""One manually launched serial mandatory block, with no retries/restarts.

Optional pairing runs require a separate complete-block budget release. The
coordinator records the fixed D1024 strata before producing any dev score.
"""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from calibration import ROOT,write,sha
from budget import ledger,remaining_seconds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--base-cache',required=True);args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();completed=[]
    try:
        protocol=json.loads((ROOT/'comparison_protocol.v1.json').read_text());base=Path(args.base_cache)
        if not (base.parent/'DONE.json').is_file() or (base.parent/'FAILED.json').exists():raise RuntimeError('Base mastery stage not accepted')
        if remaining_seconds()<3*protocol['budget']['per_training_gpu_hours']*3600+4*protocol['budget']['per_development_model_gpu_hours']*3600:raise RuntimeError('Cannot reserve mandatory block under frozen total budget')
        write(out/'CONFIG.json',{'protocol_sha256':sha(ROOT/'comparison_protocol.v1.json'),'methods':['D','A','R'],'base_cache':str(base),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'code_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'manual_one_shot':True,'optional_pairing_requires_separate_release':True,'started_unix':time.time(),'ledger_at_start':ledger()})
        def run(label,command):
            remaining_seconds();write(out/'PROGRESS.json',{'stage':label,'completed':completed,'wall_seconds':time.monotonic()-start,'ledger':ledger()})
            with (out/(label+'.log')).open('x') as log:
                subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy(),cwd=ROOT)
            completed.append(label)
        for method in protocol['comparisons']['fixed_order']:
            folder=out/method
            run('train_'+method,[str(ROOT/'train.py'),'--method',method,'--output',str(folder),'--base-cache',str(base)])
            if method=='D':
                cache=json.loads((folder/'cache_001024/MASTERY.json').read_text());rows=[json.loads(s) for s in (ROOT/'data/calibration_pairs.v1.jsonl').read_text().splitlines()]
                joint=sum(cache['records'][r['id']+':en']['known'] and cache['records'][r['id']+':sw']['known'] for r in rows)
                enonly=sum(cache['records'][r['id']+':en']['known'] and not cache['records'][r['id']+':sw']['known'] for r in rows)
                initial=json.loads((base/'MASTERY.json').read_text());base_sw=sum(v['base'] for k,v in initial['records'].items() if k.endswith(':sw'))
                write(out/'PAIRING_GATE_AT1024.json',{'fixed_checkpoint':str(folder/'checkpoint-001024.pt'),'cache_sha256':sha(folder/'cache_001024/MASTERY.json'),'joint_known':joint,'EN_only_known':enonly,'Base_SW_known':base_sw,'adequate':joint>=8 and enonly>=8,'shared_prefix_possible':base_sw==0,'no_development_score_used':True,'not_permission_to_run_without_complete_block_budget':True})
        from train import BASE
        endpoint_paths={}
        for label,model in [('Base',BASE)]+[(m+'2048',out/m/'final') for m in ['A','D','R']]:
            folder=out/('dev_'+label);run('dev_'+label,[str(ROOT/'evaluate_dev.py'),'--model',str(model),'--label',label,'--output',str(folder)]);endpoint_paths[label]=str(folder)
        # No automatic optional run or result-dependent hyperparameter change.
        # Keep mandatory report separate from later complete pairing report.
        command=[str(ROOT/'compare_dev.py'),'--output',str(out/'MANDATORY_DECISION.json')]
        for label,folder in endpoint_paths.items():command+=['--endpoint',label+'='+folder]
        run('compare_mandatory',command)
        write(out/'DONE.json',{'completed':completed,'protocol_sha256':sha(ROOT/'comparison_protocol.v1.json'),'wall_seconds':time.monotonic()-start,'ledger':ledger(),'mandatory_report':str(out/'MANDATORY_DECISION.json'),'pairing_status':'requires fixed-strata and remaining-budget audit; no pairing superiority from D','long_training_released':False})
    except BaseException as error:
        write(out/'FAILED.json',{'error':str(error),'traceback':traceback.format_exc(),'completed':completed,'wall_seconds':time.monotonic()-start,'ledger':ledger(),'no_retry_or_restart':True});raise
if __name__=='__main__':main()
