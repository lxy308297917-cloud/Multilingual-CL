"""Freeze32 prefixes from first4 originaldev questions perSW/EN, no outcome selection."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();h=a.root;code=Path(__file__).parent
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(h/'models/Qwen3-1.7B',local_files_only=True)
    records=[];sources={};adapter=None
    for lang,name in [('sw','sw_clip20_dev100_8192_20261020_v1'),('en','en_sw_clip20_dev100_8192_20261020_v1')]:
        folder=h/'logs/copsd_official_10lang/development_vllm8192'/name
        cfg=read(folder/'CONFIG.json');assert cfg['language']==lang and cfg['checkpoint_step']==20
        if adapter is None:adapter=Path(cfg['adapter'])
        assert Path(cfg['adapter'])==adapter
        rows=[json.loads(s) for s in (folder/'ITEMS.jsonl').read_text().splitlines()][:4]
        sources[lang]={'relative':str(folder.relative_to(h)),'items_sha256':sha(folder/'ITEMS.jsonl'),'config_sha256':sha(folder/'CONFIG.json')}
        for row in rows:
            response=row['samples'][0]['token_ids'];assert len(response)>128
            prompt=tokenizer.encode(row['prompt'],add_special_tokens=False)
            for pos in [0,32,64,128]:
                records.append({'key':f'{lang}:{row["id"]}:{pos}','id':row['id'],'language':lang,'response_position':pos,'input_ids':prompt+response[:pos+1],'target_token_id':response[pos]})
    assert len(records)==32 and len({r['id'] for r in records})==4
    directory=h/'logs/copsd_official_10lang/engine_probability_probe_20261022_v1';directory.mkdir(exist_ok=False)
    path=directory/'INPUTS.PRIVATE.json';path.write_text(json.dumps(records,indent=2)+'\n')
    old=read(code/'eval_dense8192_frozen_20261021.PUBLIC.json');export=h/old['export_relative'];manifest=read(export/'EXPORT.MANIFEST.PRIVATE.json')
    assert sha(export/'EXPORT.MANIFEST.PRIVATE.json')==old['export_manifest_sha256'] and manifest['source_adapter_sha256']==sha(adapter/'adapter_model.safetensors')
    f={'frozen_at':datetime.datetime.now().isoformat(),'protocol':'fixed_prefix_HF_native_dense_probability_diagnostic32_v1','official_commit':'4f4410730e9b6775a30d20f4b6e141cba2e87590','model_revision':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
        'input_relative':str(path.relative_to(h)),'input_sha256':sha(path),'adapter_relative':str(adapter.relative_to(h)),'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
        'dense_export_relative':old['export_relative'],'dense_export_manifest_sha256':old['export_manifest_sha256'],'source_records':sources,
        'original_question_IDs':sorted({r['id'] for r in records}),'response_positions':[0,32,64,128],'prefixes':32,'distinct_original_questions':4,'languages':['sw','en'],
        'output_relative':'logs/copsd_official_10lang/engine_probability_probe_20261022_v1/results',
        'code_sha256':{n:sha(code/n) for n in ['engine_probability_probe.py','run_engine_probability_probe.sh','prepare_engine_probability_probe.py','evaluate_vllm.py','evaluate_transformers.py','benchmark_eval_concurrency8192.py','freeze_vllm_eval.py']},
        'analysis_policy':'Report sampled-token logprob andargmax differences, adapter-minusBase delta differences and correlations. Descriptive fixed-prefix software diagnostic; no arbitrary numerical pass criterion, answer scoring or condition/checkpoint selection. Do not infer full-generation equivalence.',
        'scope':'One fixed existingSW20 adapter,32 prefixes=4 original questions x2 languages x4 positions; not32 independent questions. Compare HFPEFT/Base with nativevLLM LoRA/Base andexisting BF16 denseexport vLLM. No model updates, source teacher change or new benchmark correctness. One dummy generated token pervLLM request only enables prompt-logprob extraction; not a short-answer efficacy experiment. HFSDPA, nativeLoRA and BF16 merging can differ numerically; measure rather than assume bitwise equality.'}
    destination=code/'engine_probability_probe_frozen_20261022.PUBLIC.json'
    with destination.open('x') as z:json.dump(f,z,indent=2);z.write('\n')
    print('Fixed32 prefixes over4 shared originalquestions, sameSW20 adapter andexisting denseexport.')


if __name__=='__main__':main()
