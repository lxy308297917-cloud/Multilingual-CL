"""Release the one fixed TE seed42 checkpoint20 for evaluation after a completed parent."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import torch


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();code=Path(__file__).parent
    train_path=code/'te4b_resume5_frozen_20261024.PUBLIC.json';f=read(train_path)
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    state=read(a.root/'gpu-queue/queues/copsd-te4b-resume5-v1/state.json')
    assert state['phase']=='completed' and state['job']['exit_code']==0
    assert state['job']['script_sha256']==f['code_sha256']['run_te4b_resume5.sh']
    folder=a.root/f['checkpoint_relative'];proof=read(folder/'SERVER_PILOT20_UPDATES_DONE.json');parent=read(folder/'SERVER_PARENT_DONE.json')
    assert proof['completed_updates']==20 and proof['micro_steps']==480 and len(proof['syncs'])==16 and proof['resumed_from_global_step']==5 and proof['logical_total_exposures']==640
    assert proof['base_unchanged_bitwise'] and parent['two_role_job_completed']
    assert proof['cost']['loss_calls']==proof['cost']['teacher_forward_calls']==proof['cost']['student_forward_calls']==480
    assert len(read(folder/'EXPOSURE_PROMPTS.PRIVATE.json'))==480
    for marker in ['RESUME_OPTIMIZER_MODEL_LOADED.json','RESUME_RNG_LOAD_CALLED.json','RESUME_DATA_ORDER_MATCH.json']:assert (folder/marker).is_file()
    assert proof['retained_checkpoint5_cost']['loss_calls']==160
    cp=folder/'checkpoint-20';ts=read(cp/'trainer_state.json');assert ts['global_step']==20 and ts['max_steps']==100
    args=torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False)
    assert args.seed==args.data_seed==42 and args.max_steps==100 and args.learning_rate==5e-6
    assert args.per_device_train_batch_size==1 and args.gradient_accumulation_steps==32
    assert torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)['last_epoch']==20
    assert {int(v['step']) for v in torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)['state'].values()}=={20}
    assert (cp/'rng_state.pth').stat().st_size>0
    eval_path=code/'te4b_resume5_eval_frozen_20261024.PUBLIC.json';e=read(eval_path)
    assert e['training_freeze_sha256']==sha(train_path)
    assert e['cases']['TE20R']['adapter_relative']==str(cp.relative_to(a.root))
    release={'released_at':datetime.now().isoformat(),'global_step':20,'training_seed':42,'data_seed':42,'resumed_from_global_step':5,'new_loss_calls':480,'logical_total_exposures':640,
        'adapter_sha256':sha(cp/'adapter_model.safetensors'),'training_freeze_sha256':sha(train_path),
        'evaluation_freeze_sha256':sha(eval_path),'training_proof_sha256':sha(folder/'SERVER_PILOT20_UPDATES_DONE.json'),
        'optimizer_scheduler_step20_verified':True,'RNG_file_present':True,
        'scope':'One fixedcheckpoint20 release; optimizer/model/RNG/data-order restore markers verified; serverRNG bitwisecontinuation not claimed. No mathematical result or candidate selection.'}
    output=a.root/e['checkpoint_release_relative']
    with output.open('x') as stream:stream.write(json.dumps(release,indent=2)+'\n')
    print(json.dumps(release))


if __name__=='__main__':main()
