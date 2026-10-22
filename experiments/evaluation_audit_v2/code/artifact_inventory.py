"""Separate observed raw artifacts, fresh runtime probes and static identities."""
from settings import *
import time
inventory={'checked_unix':time.time(),'public_snapshot':'67a16e0eae725d90b7217394fbd6116e4bbe4201','dense_training':{},'sparse_training':{},'original_development':{}}
for name in 'ABCD':
    folder=TRAIN/'comparison_v3'/name;done=json.loads((folder/'TRAINING_DONE.json').read_text());final=pathlib.Path(done['final_model']);identity=json.loads((RUN/'mechanism'/name/'IDENTITY.json').read_text());receipt=json.loads((final.parent/'RECOVERABLE.json').read_text());assert receipt['files']['model/model.safetensors']==identity['model_sha256']
    assert (final/'model.safetensors').is_file() and (final.parent/'state.pt').is_file() and (folder/'STEPS.jsonl').is_file()
    inventory['dense_training'][name]={'steps':done['steps'],'model':str(final),'model_bytes':(final/'model.safetensors').stat().st_size,'actual_model_sha256_read_this_task':identity['model_sha256'],'optimizer_state':str(final.parent/'state.pt'),'optimizer_state_exists':True,'optimizer_state_sha256_read_and_loaded_this_task':name in 'BCD','raw_step_log_sha256':sha(folder/'STEPS.jsonl'),'scope':'Existing model loaded for fresh true-Qwen diagnostics/evaluation; original training not rerun'}
sparse=H/'models/checkpoint/paired_capability_sparse_20261021/comparison/matched_v1'
for name in ['A','D','R']:
    folder=sparse/name;done=json.loads((folder/'TRAINING_DONE.json').read_text());final=folder/'final/model.safetensors';assert final.is_file() and (folder/'STEPS.jsonl').is_file()
    inventory['sparse_training'][name]={'training_done':str(folder/'TRAINING_DONE.json'),'model':str(final),'model_exists':True,'model_bytes':final.stat().st_size,'steps_log_exists':True,'scope':'Today metadata/filesystem/raw-output verification only; original sparse weight tensors not reloaded or retrained today'}
for track,folder,names in [('dense',TRAIN/'development',['Base']+list('ABCD')),('sparse',sparse,['dev_Base','dev_A2048','dev_D2048','dev_R2048'])]:
    for name in names:
        out=folder/name;done=json.loads((out/'DONE.json').read_text());pp=out/'PREDICTIONS.jsonl';h=sha(pp);assert h==done['predictions_sha256'];n=sum(1 for _ in pp.open());assert n==1024
        inventory['original_development'][track+'/'+name]={'predictions_path':str(pp),'actual_sha256_read':h,'rows':n,'done_path':str(out/'DONE.json'),'scope':'Raw output identity rechecked, original scoring retained; not new generation or rerun of original training'}
inventory['today_execution_scope']=['Scorer acceptance on36 new source groups and290 natural/controlled cases','Five existing Dense models loaded for192 fresh responses each','Actual fixed64 training-pair forwards and cached1000 independent-pair diagnostics','Seven gradient states,21 real fixed-batch probes; saved-state AdamW candidates on independent copies only','Original cache shapes replayed for8 teacher vectors; no new scientific training']
write(REPORT/'ORIGINAL_ARTIFACT_INVENTORY.json',inventory);print('Verified original7training endpoints files and9 raw1024-item development outputs; no retraining')
