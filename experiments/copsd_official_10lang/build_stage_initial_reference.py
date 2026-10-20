"""Map a completed parent stage's final sync to a fresh stage's step0.

This constructs a private expectation file only. It never loads a model,
optimizer or GPU, and cannot prove that the next runtime actually loaded it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def build(parent_checkpoint, output, order_id, next_stage, model_revision, engineering_only=False):
    parent_checkpoint=Path(parent_checkpoint);output=Path(output)
    assert not output.exists()
    assert re.fullmatch(r'[A-Za-z0-9_-]+',order_id)
    assert next_stage in [2,3] and re.fullmatch(r'[0-9a-f]{40}',model_revision)
    state=json.loads((parent_checkpoint/'trainer_state.json').read_text())
    step=state['global_step'];assert isinstance(step,int) and step>0
    proof_path=parent_checkpoint.parent/'COLLOCATE_TRAIN_UPDATES_DONE.json'
    proof=json.loads(proof_path.read_text())
    assert proof['global_step']==step and proof['base_unchanged_bitwise']
    assert json.loads((parent_checkpoint.parent/'RUNTIME_SHUTDOWN_DONE.json').read_text())['ordered_pool_release_completed']
    rows=[x for x in proof['syncs'] if x['global_step']==step]
    assert len(rows)==1 and proof['syncs'][-1]==rows[0]
    final=rows[0]
    for field in ['adapter_hash','engine_hash']:
        assert re.fullmatch(r'[0-9a-f]{64}',final[field])
    if not engineering_only:
        context=proof['sequential_context']
        assert context['order_id']==order_id and context['stage_index']==next_stage-1
        assert context['model_revision']==model_revision
    weights=parent_checkpoint/'adapter_model.safetensors'
    assert weights.stat().st_size>0
    value={'scope':'Expected initial policy reference only; requires actual adapter loading, fresh optimizer/scheduler and runtime sync equality.',
        'engineering_only':engineering_only,'order_id':order_id,'next_stage_index':next_stage,
        'model_revision':model_revision,'parent_checkpoint_PRIVATE':str(parent_checkpoint.resolve()),
        'parent_checkpoint_step':step,'parent_adapter_file_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
        'parent_proof_sha256':hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        'parent_final_sync_original_step':step,
        'syncs':[{'global_step':0,'adapter_hash':final['adapter_hash'],'engine_hash':final['engine_hash']}],
        'mapping':'Previous completed-stage final student weights -> new stage local step0; do not use previous initial step0.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--order-id',required=True)
    p.add_argument('--next-stage',type=int,choices=[2,3],required=True)
    p.add_argument('--model-revision',required=True)
    p.add_argument('--engineering-only',action='store_true')
    a=p.parse_args()
    value=build(a.parent_checkpoint,a.output,a.order_id,a.next_stage,a.model_revision,a.engineering_only)
    print(json.dumps({'reference_created':True,'parent_final_step':value['parent_checkpoint_step'],
        'next_stage_local_step':0,'engineering_only':a.engineering_only,
        'scope':'No training, language selection or actual GPU synchronization performed.'}))


if __name__=='__main__':main()
