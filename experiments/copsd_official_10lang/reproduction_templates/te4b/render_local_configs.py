"""Render scientific templates using explicitly supplied local bindings.

No GPU, model load, queue start, storage migration or existing file overwrite.
This project-local entry assumes preceding shared4B resource locks and accepted
BaseEN evidence already exist. It is not a standalone reproduction of the study.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint-storage',type=Path,required=True);p.add_argument('--base-english-folder',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--site-packages-relative',type=Path,required=True);a=p.parse_args()
    template=Path(__file__).parent;code=template.parent.parent;root=a.root.resolve()
    train=read(template/'training.template.json');evaluation=read(template/'evaluation.template.json')
    pool=read(code/'te4b_pool2700_prepared_20261024.PUBLIC.json')
    assert pool['language']=='te' and pool['rows']==2700 and pool['development_items']==200
    manifest=root/train['training_manifest_relative'];assert read(manifest)['data_sha256']==pool['data_sha256']==sha(root/pool['data_relative'])
    train.update(frozen_at=datetime.now().isoformat(),checkpoint_actual_container=str(a.checkpoint_storage.resolve()),training_manifest_sha256=sha(manifest))
    train['code_sha256']={n:sha(code/n) for n in train['code_sha256']}
    site=a.site_packages_relative;assert not site.is_absolute() and '..' not in site.parts
    runtime_files=[site/'trl/scripts/vllm_serve.py',site/'trl/extras/vllm_client.py']
    train['installed_source_sha256']={str(n):sha(root/n) for n in runtime_files}
    evaluation['selected_IDs']=read(code/'followup200_frozen_20261021.PUBLIC.json')['selected_IDs']
    evaluation['inputs']['te']['sha256']=sha(root/evaluation['inputs']['te']['relative']);assert evaluation['inputs']['te']['sha256']==pool['development_sha256']
    evaluation['model_lock_sha256']=sha(root/evaluation['model_lock_relative'])
    accepted=read(code/'sw4b_followup200_completed_20261022.PUBLIC.json')['languages']['en']['conditions']['Base']['source'];cache=a.base_english_folder.resolve()
    assert sha(cache/'ITEMS.jsonl')==accepted['items_sha256'] and sha(cache/'CONFIG.json')==accepted['config_sha256']
    evaluation['cached_Base_English']={'run_id':accepted['run_id'],'relative':str(cache.relative_to(root)),'ITEMS_sha256':sha(cache/'ITEMS.jsonl'),'CONFIG_sha256':sha(cache/'CONFIG.json'),'cost_already_counted':True}
    evaluation['runtime_code_sha256']={n:sha(code/n) for n in evaluation['runtime_code_sha256']}
    train_text=json.dumps(train,ensure_ascii=False,indent=2)+'\n';evaluation['training_freeze_sha256']=hashlib.sha256(train_text.encode()).hexdigest();evaluation['frozen_at']=datetime.now().isoformat();eval_text=json.dumps(evaluation,ensure_ascii=False,indent=2)+'\n'
    assert '@LOCAL_' not in train_text+eval_text and '@CHECKPOINT_STORAGE@' not in train_text+eval_text
    names=['te4b_server20_frozen_20261024.PUBLIC.json','te4b_eval_frozen_20261024.PUBLIC.json'];a.output_dir.mkdir(parents=True,exist_ok=True)
    assert all(not (a.output_dir/n).exists() for n in names),'Existing runtime configs must remain unchanged'
    for name,text in zip(names,[train_text,eval_text]):
        with (a.output_dir/name).open('x') as stream:stream.write(text)
    print('Two local runtime configs written; no experiment or GPU task started.')


if __name__=='__main__':main()
