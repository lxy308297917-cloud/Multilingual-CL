"""Read-only current release report. Historical plan is not launch authority."""
import argparse,json,shutil
from pathlib import Path
from calibration import ROOT,sha
from budget import ledger

def report():
    protocol_path=ROOT/'comparison_protocol.v1.json';protocol=json.loads(protocol_path.read_text());docs=ROOT.parents[1]/'docs/research/paired_capability_sparse';storage=Path('/data0/users/lixiaoyang/models/checkpoint/paired_capability_sparse_20261021/comparison')
    changed=[p for p,h in protocol['frozen_files'].items() if not Path(p).is_file() or sha(p)!=h]
    audit=json.loads((ROOT/'data/dev_v4/INDEPENDENT_AUDIT.json').read_text());free=shutil.disk_usage(storage).free
    releases={}
    for name in ('TRAINING_RELEASE','DEVELOPMENT_RELEASE','PAIRING_RELEASE'):
        path=ROOT/(name+'.json');releases[name]=json.loads(path.read_text()) if path.exists() else None
    return {'protocol_sha256':sha(protocol_path),'changed_frozen_inputs':changed,'gates':{'frozen_inputs_unchanged':not changed,'dev_v4_isolation':audit['status']=='passed' and all(audit['gates'].values()),'storage_for_seven_two_checkpoint_runs':free>300*1024**3,'real_model_technical_pass':(docs/'TECHNICAL_DONE.json').exists(),'Base_mastery_pass':(storage/'base_mastery_v1/DONE.json').exists()},'releases':releases,'budget':protocol['budget'],'ledger':ledger(),'full_loop_throughput_gate':'first32 nonzeroLR steps, before completing comparison; require physical24GB and frozen per-training time','historical_plan':'plan.technical-history.json is archival and superseded','no_gpu_launch_or_mutation':True}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output');args=ap.parse_args();r=report();text=json.dumps(r,ensure_ascii=False,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    else:print(text)
