"""New explicitly separate merged-serving BN/English development protocol.

Base and BN20 share prompts, source, question seeds, serving backend and budget.
The prior native16 evaluation/release stays immutable. No training is performed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''):h.update(block)
    return h.hexdigest()
def stat(p):
    s=p.stat();return [s.st_ino,s.st_size,s.st_mtime_ns]


def prepare(root,case,controller=False):
    code=Path(__file__).parent;fp=code/'canonical_merged_bn4b_eval_frozen_20261023.PUBLIC.json';f=read(fp);spec=f['cases'][case]
    for k,v in f['runtime_code_sha256'].items():assert sha(code/k)==v,k
    for k,v in f['installed_source_sha256'].items():assert sha(root/k)==v,k
    lp=root/f['model_lock_relative'];assert sha(lp)==f['model_lock_sha256'];lock=read(lp);assert lock['revision']==f['model_revision']
    validation={'case':case,'freeze_sha256':sha(fp),'model_lock_sha256':sha(lp),'files':{}}
    prior=None
    if controller:
        proof=Path(os.environ['COPSD_PARENT_MODEL_VALIDATION']);assert sha(proof)==os.environ['COPSD_PARENT_MODEL_VALIDATION_SHA256'];prior=read(proof)
        assert all(prior[k]==validation[k] for k in ['case','freeze_sha256','model_lock_sha256'])
    for name,info in lock['files'].items():
        path=root/lock['model_relative']/name
        if controller:assert prior['files'][name]=={'sha256':info['sha256'],'stat':stat(path)},name
        else:assert sha(path)==info['sha256'],name
        validation['files'][name]={'sha256':info['sha256'],'stat':stat(path)}
    canonical=root/spec['canonical_reference_relative'];assert sha(canonical)==spec['canonical_reference_sha256']
    with canonical.open() as stream:weights=[json.loads(x) for x in stream]
    assert len(weights)==spec['canonical_reference_rows'] and len(weights)%398==0
    assert {x['name'] for x in weights[-398:]}==set(read(root/'models/Qwen3-4B/model.safetensors.index.json')['weight_map'])
    adapter=root/spec['adapter_relative'] if spec['adapter_relative'] else None
    if adapter:
        assert sha(adapter/'adapter_model.safetensors')==spec['adapter_sha256']
        release_path=root/spec['training_release_relative'];assert sha(release_path)==spec['training_release_sha256'];release=read(release_path)
        assert release['adapter_sha256']==spec['adapter_sha256'] and release['global_step']==20
        assert release['evaluation_freeze_sha256']==f['original_native16_evaluation_freeze_sha256']
        assert sha(code/'bn4b_eval_frozen_20261023.PUBLIC.json')==f['original_native16_evaluation_freeze_sha256']
        assert read(adapter/'trainer_state.json')['global_step']==20
        cfg=read(adapter/'adapter_config.json');assert cfg['r']==64 and cfg['lora_alpha']==128 and cfg['bias']=='none'
    source=root/'COPSD-official-20261015/polymath_eval/evaluate_math.py';assert sha(source)==f['official_source_sha256']
    items=[];aligned_inputs={}
    for lang in f['languages']:
        path=root/f['inputs'][lang]['relative'];assert sha(path)==f['inputs'][lang]['sha256'];rows=read(path)
        assert [str(x['problem_id']) for x in rows]==f['selected_IDs'] and len(rows)==200
        aligned_inputs[lang]=rows
        for row in rows:items.append({'id':str(row['problem_id']),'language':lang,'question':row['problem' if lang=='en' else 'problem_bn'],'answer':row['Answer']})
    assert len(items)==len({(x['language'],x['id']) for x in items})==400
    assert all(x['problem']==y['problem'] and x['Answer']==y['Answer'] for x,y in zip(aligned_inputs['bn'],aligned_inputs['en'],strict=True))
    assert all(isinstance(x['question'],str) and x['question'].strip() for x in items)
    output=root/spec['output_relative'];assert not output.exists()
    return f,spec,items,output,validation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--case',choices=['Base','BN20'],required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    f,spec,items,output,_=prepare(a.root,a.case,controller=not a.prepare_only)
    if a.prepare_only:print(a.case+': fullmodel/source/input/release preflight passed, noGPU',flush=True);return
    assert os.environ['GPU_QUEUE_GPU_COUNT']=='1' and os.environ['CUDA_VISIBLE_DEVICES']==''
    from transformers import AutoTokenizer
    from trl.extras.vllm_client import VLLMClient
    from evaluate_vllm import official_api,question_seed
    from trl_server_eval_seeded_worker32 import SEED_KEY
    code=Path(__file__).parent;fp=code/'canonical_merged_bn4b_eval_frozen_20261023.PUBLIC.json'
    output.mkdir(parents=True)
    initialization=read(Path(os.environ['COPSD_DIRECT_INIT_RECEIPT']))
    assert initialization['all398_reference_weights_match'] and initialization['all_weights_loaded']==398
    assert initialization['adapted_weights_loaded']==(252 if spec['adapter_relative'] else 0)
    tokenizer=AutoTokenizer.from_pretrained(a.root/'models/Qwen3-4B');api=official_api(a.root/'COPSD-official-20261015/polymath_eval/evaluate_math.py')
    prompts=[api['build_qwen3_prompt'](tokenizer,x['question'],x['language'].upper(),True) for x in items]
    seeds=[question_seed(42,x['language'],x['id']) for x in items];prompt_ids=[tokenizer.encode(x,add_special_tokens=False) for x in prompts]
    assert max(map(len,prompt_ids))+8192<=40960
    config={'protocol':f['protocol'],'case':a.case,'model_revision':f['model_revision'],'model':str(a.root/'models/Qwen3-4B'),
        'adapter':str(a.root/spec['adapter_relative']) if spec['adapter_relative'] else None,'adapter_sha256':spec['adapter_sha256'],
        'freeze_sha256':sha(fp),'evaluator_sha256':sha(Path(__file__)),'official_source_sha256':f['official_source_sha256'],
        'samples':1,'max_new_tokens':8192,'max_num_seqs':32,'temperature':1.,'top_p':.95,'top_k':-1,'seed':42,
        'initialization':initialization,'canonical_reference_sha256':spec['canonical_reference_sha256'],
        'model_verification':'ParentverifiedfullfileSHAs; childverifiedparentproofandunchangedfileidentity/stat; servingworkerverifiedall398canonicalweights.',
        'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','vllm','math-verify']},
        'source_inputs':f['inputs'],'items':400,'formal_test':False,'old_native16_scores_or_release_modified':False}
    (output/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');(output/'PHASE.json').write_text(json.dumps({'phase':'generation','items':400})+'\n')
    client=VLLMClient(host='127.0.0.1',server_port=int(os.environ['COPSD_SERVER_HTTP_PORT']),connection_timeout=120)
    tick=time.monotonic()
    try:response=client.generate(prompts=prompts,n=1,temperature=1.,top_p=.95,top_k=-1,min_p=0.,max_tokens=8192,generation_kwargs={'presence_penalty':0.,'extra_args':{SEED_KEY:seeds}})
    finally:client.session.close()
    seconds=time.monotonic()-tick;assert response['prompt_ids']==prompt_ids and len(response['completion_ids'])==400
    seed_path=Path(os.environ['COPSD_EVAL_SEED_RECEIPTS'])
    with seed_path.open() as stream:receipts=[json.loads(x) for x in stream]
    assert len(receipts)==1 and receipts[0]['per_question_params_reached_LLM_generate'];metadata=receipts[0]['records'];assert len(metadata)==400
    cached=[]
    for item,prompt,seed,tokens,meta in zip(items,prompts,seeds,response['completion_ids'],metadata,strict=True):
        assert meta['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest() and meta['seed']==seed
        assert meta['token_ids_sha256']==hashlib.sha256(json.dumps(tokens).encode()).hexdigest() and meta['generated_tokens']==len(tokens)
        assert 0<len(tokens)<=8192 and meta['finish_reason'] in ['stop','length']
        cached.append({**item,'prompt':prompt,'seed':seed,'token_ids':tokens,'finish_reason':meta['finish_reason'],'raw_output':tokenizer.decode(tokens,skip_special_tokens=False)})
    raw_path=output/'GENERATED.PRIVATE.jsonl'
    with raw_path.open('x') as stream:
        for row in cached:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    (output/'GENERATION_DONE.PRIVATE.json').write_text(json.dumps({'items':400,'sha256':sha(raw_path),'freeze_sha256':sha(fp),'generation_seconds':seconds,'seed_receipt_sha256':sha(seed_path)})+'\n')
    (output/'PHASE.json').write_text(json.dumps({'phase':'scoring','items':400})+'\n');records={lang:[] for lang in f['languages']}
    for row in cached:
        answer=api['extract_boxed_answer'](row['raw_output']);correct=bool(api['grade_answer'](answer,row['answer']))
        sample={k:row[k] for k in ['seed','token_ids','finish_reason','raw_output']}
        sample.update(tokens=len(row['token_ids']),boxed_answer=answer,correct=correct)
        records[row['language']].append({k:row[k] for k in ['id','language','question','answer','prompt']}|{'samples':[sample]})
    metrics={}
    for lang,rows in records.items():
        assert len(rows)==200;folder=output/lang;folder.mkdir()
        with (folder/'ITEMS.jsonl').open('x') as stream:
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        samples=[x['samples'][0] for x in rows]
        metrics[lang]={'items':200,'single_correct':sum(x['correct'] for x in samples),'single_accuracy':sum(x['correct'] for x in samples)/200,
            'generated_tokens':sum(x['tokens'] for x in samples),'normal_terminations':sum(x['finish_reason']=='stop' for x in samples),
            'truncated':sum(x['finish_reason']=='length' for x in samples),'answer_extraction_failures':sum(x['boxed_answer'] is None for x in samples),
            'n':1,'formal_test':False,'protocol':f['protocol'],'ITEMS_sha256':sha(folder/'ITEMS.jsonl')}
        (folder/'DONE.json').write_text(json.dumps(metrics[lang],indent=2)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    tb=a.root/'logs/copsd_official_10lang/tensorboard/canonical_merged_bn4b_dev200_20261023'/a.case;writer=SummaryWriter(str(tb))
    for lang,m in metrics.items():writer.add_scalar(lang+'/accuracy',m['single_accuracy'],20 if a.case=='BN20' else 0)
    writer.close()
    done={'case':a.case,'protocol':f['protocol'],'items':400,'languages':metrics,'generation_seconds':seconds,'generated_tokens':sum(x['generated_tokens'] for x in metrics.values()),'seed_receipt_sha256':sha(seed_path),'no_new_training':True,'formal_test':False}
    (output/'DONE.json').write_text(json.dumps(done,indent=2)+'\n');(output/'PHASE.json').write_text(json.dumps({'phase':'done'})+'\n')
    print('CANONICAL_MERGED_BN4B_CASE_DONE',json.dumps(done),flush=True)


if __name__=='__main__':main()
