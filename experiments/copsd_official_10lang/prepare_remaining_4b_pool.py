"""Prepare one remaining 4B language against frozen SW2700 and dev200 IDs.

No scores select questions/language. Preserve official translations and English
references. Historical1.7B exposure is not evidence of 4B training exposure.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import unicodedata
import pyarrow.parquet as pq


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).split())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--language',choices=['te','th','ru','de','fr','es'],required=True);p.add_argument('--output-report',type=Path,required=True);a=p.parse_args();root=a.root;code=Path(__file__).parent;lang=a.language
    audit_path=root/'data/copsd_official_20261015/adapted_v2_ordered/DATA_AUDIT.json';audit=read(audit_path)
    assert audit['official_repo']=='yihongLiu/COPSD-PolyMath-TrainDataset' and audit['revision']=='519ea2be68fec00c8a033cb2040059ff595b06a0'
    raw_path=Path(audit['input_files'][lang]['path']);assert sha(raw_path)==audit['input_files'][lang]['sha256'];raw=pq.read_table(raw_path).to_pylist();assert len(raw)==3000
    original={hashlib.sha256(x['problem'].strip().encode()).hexdigest():x for x in raw};assert len(original)==3000
    source=root/'data/copsd_official_20261015/adapted_v2_ordered'/f'project_train_2900_{lang}.json';all_rows=read(source);by_id={str(x['problem_id']):x for x in all_rows};assert len(all_rows)==len(by_id)==2900
    for x in all_rows:
        y=original[str(x['problem_id'])]
        assert all(x[k]==y[k] for k in ['source','problem','solution','Answer']) and x['problem_'+lang]==y['problem_translated']
        assert all(isinstance(x[k],str) and x[k].strip() for k in ['problem','solution','Answer','problem_'+lang])
    frozen=read(code/'followup200_frozen_20261021.PUBLIC.json');lr=read(code/'sw_lr20_frozen_20261022.PUBLIC.json');manifest=root/lr['training_manifest_relative'];assert sha(manifest)==lr['training_manifest_sha256'];sw=read(manifest);sw_path=root/sw['data_relative'];assert sha(sw_path)==sw['data_sha256'];shared=read(sw_path)
    train=[by_id[str(x['problem_id'])] for x in shared];dev=[by_id[k] for k in frozen['selected_IDs']]
    assert len(train)==2700 and len(dev)==200 and [str(x['problem_id']) for x in train]==sw['training_IDs']
    assert all(all(x[k]==y[k] for k in ['problem','solution','Answer']) for x,y in zip(train,shared))
    english={norm(x['problem']) for x in train}
    old=read(root/'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json')
    for heldout in [dev,old]:
        assert not {str(x['problem_id']) for x in train}&{str(x['problem_id']) for x in heldout}
        assert not english&{norm(x['problem']) for x in heldout}
    base_dev=read(root/frozen['inputs']['en']['relative']);assert all(all(x[k]==y[k] for k in ['problem_id','problem','solution','Answer']) for x,y in zip(dev,base_dev))
    directory=root/'data/copsd_official_20261015'/f'{lang}4b_pool2700_20261024_v1';assert not directory.exists() and not a.output_report.exists();directory.mkdir()
    data=directory/f'project_train_2700_{lang}.PRIVATE.json';development=directory/f'project_followup_200_{lang}.PRIVATE.json'
    data.write_text(json.dumps(train,ensure_ascii=False,indent=2)+'\n');development.write_text(json.dumps(dev,ensure_ascii=False,indent=2)+'\n')
    report={'prepared_at':datetime.now().isoformat(),'language':lang,'rows':2700,'source_relative':str(source.relative_to(root)),'source_sha256':sha(source),'source_audit_sha256':sha(audit_path),'official_dataset_revision':audit['revision'],'official_raw_file_sha256':sha(raw_path),'all2900_adapted_rows_match_official_question_translation_reference':True,'data_relative':str(data.relative_to(root)),'data_sha256':sha(data),'development_relative':str(development.relative_to(root)),'development_sha256':sha(development),'development_items':200,'same_order_and_English_question_solution_answer_as_SW2700':True,'ordered_problem_IDs_sha256':hashlib.sha256(json.dumps(sw['training_IDs']).encode()).hexdigest(),'normalized_English_and_ID_overlap_with_original100_and_reserved200':0,'normalized_English_duplicate_training_rows_preserved':len(train)-len(english),'translation_replacement_character_rows':sum('\ufffd' in x['problem_'+lang] for x in train),'maximum_target_question_characters':max(len(x['problem_'+lang]) for x in train),'new_training_started':False,'builder_sha256':sha(Path(__file__)),'scope':'One independent4B candidate. Same2700pool and200heldout IDs as preceding4B languages; no new sampling or synthetic answers. Preserves official content. Original100 and reserved200 excluded from4Btraining. These are reused development questions, not newblindtest; earlier1.7B TE exposure may differ and is not mixed with this cohort.'}
    (directory/'MANIFEST.PRIVATE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with a.output_report.open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
