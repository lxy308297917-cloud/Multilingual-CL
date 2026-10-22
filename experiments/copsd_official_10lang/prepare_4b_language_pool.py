"""Prepare a new4B language pool using the frozen SW2700 original-question order."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import unicodedata


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).split())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--language',choices=['ja','zh','bn'],required=True)
    p.add_argument('--output-report',type=Path,required=True)
    a=p.parse_args();code=Path(__file__).parent;root=a.root;lang=a.language
    old=read(code/'followup200_frozen_20261021.PUBLIC.json')
    source=root/old['source_inputs'][lang]['relative'];assert sha(source)==old['source_inputs'][lang]['sha256']
    lr=read(code/'sw_lr20_frozen_20261022.PUBLIC.json');manifest=root/lr['training_manifest_relative'];assert sha(manifest)==lr['training_manifest_sha256']
    sw=read(manifest);sw_path=root/sw['data_relative'];assert sha(sw_path)==sw['data_sha256']
    shared=read(sw_path);assert len(shared)==2700 and [str(r['problem_id']) for r in shared]==sw['training_IDs']
    original=read(source);by_id={str(r['problem_id']):r for r in original};assert len(original)==len(by_id)==2900
    rows=[by_id[str(r['problem_id'])] for r in shared]
    assert all(all(x[k]==y[k] for k in ['problem','solution','Answer']) for x,y in zip(shared,rows))
    assert all(isinstance(r['problem_'+lang],str) and r['problem_'+lang].strip() for r in rows)
    english={norm(r['problem']) for r in rows}
    for heldout in [read(root/old['inputs']['sw']['relative']),read(root/'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json')]:
        assert not {str(r['problem_id']) for r in rows}&{str(r['problem_id']) for r in heldout}
        assert not english&{norm(r['problem']) for r in heldout}
    folder=root/'data/copsd_official_20261015'/f'{lang}4b_pool2700_20261022_v1';folder.mkdir(exist_ok=False)
    data=folder/f'project_train_2700_{lang}.PRIVATE.json'
    data.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    report={'prepared_at':datetime.now().isoformat(),'language':lang,'rows':2700,
        'source_relative':str(source.relative_to(root)),'source_sha256':sha(source),
        'data_relative':str(data.relative_to(root)),'data_sha256':sha(data),
        'ordered_problem_IDs_sha256':hashlib.sha256(json.dumps(sw['training_IDs']).encode()).hexdigest(),
        'same_order_and_English_question_solution_answer_as_SW2700':True,
        'normalized_English_and_ID_overlap_with_original100_and_reserved200':0,
        'source_rows_preserved_except_filtering_and_order':True,'new_training_started':False,
        'scope':'Data interface preparation only for a separate4B extension, not a trainedmodel or selectedlanguage. JA uses the previously recorded one-row prefix repair; no new rewriting or target-language solution synthesis.',
        'builder_sha256':sha(Path(__file__))}
    (folder/'MANIFEST.PRIVATE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with a.output_report.open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
