from pathlib import Path
import json,sys
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
cli=parser.parse_args()
from evaluate_vllm import official_api,question_seed
from evaluate_transformers import load_items,build_prompt,official_prompts
from transformers import AutoTokenizer
root=cli.root;source=root/'COPSD-official-20261015/polymath_eval/evaluate_math.py'
api=official_api(source);tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen3-1.7B',local_files_only=True)
old=official_prompts(source);count=0;ids_ref=None
for lang in 'sw th te ja bn en'.split():
 p=root/f'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_{lang if lang!="en" else "sw"}.json'
 rows=json.loads(p.read_text());items=[{'id':r['problem_id'],'question':r['problem'] if lang=='en' else r['problem_'+lang],'answer':r['Answer']} for r in rows]
 if lang!='en':assert items==list(load_items(p,'dev',lang))
 ids=[x['id'] for x in items]
 if ids_ref is None:ids_ref=ids
 else:assert ids==ids_ref
 for item in items:
  new=api['build_qwen3_prompt'](tokenizer,item['question'],lang.upper(),True)
  assert new==build_prompt(tokenizer,item['question'],lang,old)
  assert 0<=question_seed(42,lang,item['id'])<2**32
  count+=1
result={'scope':'CPU input/prompt compatibility only, no inference or score parity claim','languages':['sw','th','te','ja','bn','en'],'heldout_items_each':100,'same_ordered_problem_ids_across_languages':True,'target_fields_match_existing_dev_loader':True,'english_uses_original_problem_field':True,'original_official_prompts_equal_existing_HF_prompts_for_items':count,'gpu_validated':False}
f=cli.output;f.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
