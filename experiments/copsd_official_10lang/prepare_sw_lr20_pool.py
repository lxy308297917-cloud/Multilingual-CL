"""Create one heldout-safe SW2700 pool for both fixed learning-rate conditions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def norm(s): return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();h=a.root
    code=Path(__file__).parent;frozen=code/'followup200_frozen_20261021.PUBLIC.json';f=read(frozen)
    source=h/'data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_sw.json'
    exclusion=h/f['future_training_exclusion']['relative'];assert sha(exclusion)==f['future_training_exclusion']['sha256']
    all_rows=read(source);ban=set(f['selected_IDs']);lookup={str(r['problem_id']):r for r in all_rows};assert len(lookup)==2900 and len(ban)==200
    old_dev=h/'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json'
    forbidden=ban|{str(r['problem_id']) for r in read(old_dev)}
    banned_English={norm(r['problem']) for r in read(old_dev)}|{norm(lookup[i]['problem']) for i in ban}
    rows=[r for r in all_rows if str(r['problem_id']) not in forbidden and norm(r['problem']) not in banned_English]
    assert len(rows)==2700 and len({str(r['problem_id']) for r in rows})==2700
    directory=h/'data/copsd_official_20261015/sw_lr20_pool_20261022_v1';directory.mkdir(exist_ok=False)
    data=directory/'project_train_2700_sw.PRIVATE.json'
    data.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    from accelerate import Accelerator,DataLoaderConfiguration
    from torch.utils.data import DataLoader,RandomSampler
    from transformers import set_seed
    set_seed(42);acc=Accelerator(cpu=True,dataloader_config=DataLoaderConfiguration(split_batches=False,dispatch_batches=False,even_batches=True,use_seedable_sampler=True,data_seed=42))
    loader=acc.prepare_data_loader(DataLoader(range(2700),batch_size=1,sampler=RandomSampler(range(2700)),num_workers=0),device_placement=False)
    order=[int(b.item()) for _,b in zip(range(640),loader)];assert len(set(order))==640
    m={'data_relative':str(data.relative_to(h)),'data_sha256':sha(data),'training_IDs':[str(r['problem_id']) for r in rows],
       'first640_expected_IDs':[str(rows[i]['problem_id']) for i in order],
       'original_source_relative':str(source.relative_to(h)),'original_source_sha256':sha(source),
       'exclusion_relative':str(exclusion.relative_to(h)),'exclusion_sha256':sha(exclusion),
       'original_dev_sha256':sha(old_dev),'followup_data_freeze_sha256':sha(frozen),
       'scope':'Same2700-row pool for fresh paired conditions, old100 plus fixed followup200 excluded byID and exact normalized English. Original retained order, no text rewrites. Initial640 exposure sequence reconstructed by actual CPUAccelerate sampler; actual prompt hashes logged during training for matched comparison. It differs from legacy2900 sampling, hence a fresh5e-6 control is required.'}
    (directory/'MANIFEST.PRIVATE.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'expected_exposures':len(order),'ID_and_normalized_English_holdout_overlap':0,'manifest':str(directory/'MANIFEST.PRIVATE.json')}))


if __name__=='__main__':main()
