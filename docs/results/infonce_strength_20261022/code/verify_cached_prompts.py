"""CPU token-level identity checks on all reused512answers per model."""
from evaluation_common import *
from transformers import AutoTokenizer

def main():
 m,rr=inputs();tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);expected={}
 for r in rr:
  msgs=([{'role':'system','content':r['system']}] if r.get('system') else [])+[{'role':'user','content':r['prompt']}];expected[r['id']]=len(tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True))
 for name in ('Base','B','D0'):
  folder=RUN/'evaluation'/name;records=rows(folder/'PREDICTIONS.jsonl');assert len(records)==512
  for r in records:
   assert r['prompt_tokens']==expected[r['id']]
   assert tok.decode(r['token_ids'],skip_special_tokens=True).strip()==r['text']
   assert r['generated_tokens']==len(r['token_ids'])
 proof={'passed':True,'models':['Base','B','D0'],'answers':1536,'all_prompt_lengths_match_frozen_templates_and_tokenizer':True,'all_raw_tokens_decode_to_saved_text':True,'all_generated_counts_match_raw_token_ids':True,'data_sha256':m['data_sha256'],'tokenizer_sha256':sha(BASE/'tokenizer.json'),'code_sha256':sha(__file__)};write(REPORT/'CACHE_TOKEN_IDENTITY_VERIFIED.json',proof);print('1536 cached prompt lengths/token counts/decoded outputs verified')
if __name__=='__main__':main()
