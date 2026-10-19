"""Reconstruct a fresh single-GPU sampler prefix and check real prompt totals.

Private output contains source IDs and hashes. It is reconstructed from trusted
local checkpoint arguments and the pinned sampler, not a direct runtime ID log.
"""
import argparse,hashlib,importlib.metadata,json,sys
from pathlib import Path
import torch
from accelerate.data_loader import SeedableRandomSampler
from transformers import AutoTokenizer


def main():
 p=argparse.ArgumentParser()
 for name in ['root','source','model','data','checkpoint','output']:
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--language',required=True)
 a=p.parse_args()
 assert a.checkpoint.resolve().is_relative_to((a.root/'models/checkpoint').resolve())
 # Only checkpoints created by this local experiment are accepted here.
 args=torch.load(a.checkpoint/'training_args.bin',map_location='cpu',weights_only=False)
 ledger=json.loads((a.checkpoint/'COST_SEGMENT_PROGRESS.json').read_text())
 assert ledger['start_step']==0 and args.per_device_train_batch_size==1 and args.gradient_accumulation_steps==32
 assert args.data_seed is not None and not args.group_by_length and not args.dataloader_drop_last and args.dataloader_num_workers==0
 cfg=args.accelerator_config
 assert (cfg.get('use_seedable_sampler') if isinstance(cfg,dict) else cfg.use_seedable_sampler)
 rows=json.loads(a.data.read_text());ids=[r['problem_id'] for r in rows];assert len(set(ids))==len(ids)
 sampler=SeedableRandomSampler(rows,generator=torch.Generator(device='cpu'),data_seed=args.data_seed)
 groups=[];indices=[];epoch=0;limit=ledger['global_step']
 while len(groups)<limit:
  sampler.set_epoch(epoch);block=[]
  for i in sampler:
   block.append(i)
   if len(block)==32:
    groups.append({'global_step':len(groups)+1,'epoch':epoch,'indices':block});indices.extend(block);block=[]
    if len(groups)==limit:break
  if block and len(groups)<limit:
   groups.append({'global_step':len(groups)+1,'epoch':epoch,'indices':block});indices.extend(block)
  epoch+=1
 assert len(indices)==ledger['loss_calls']
 sys.path.insert(0,str(a.source));from multilingual_data_collator import MultilingualSelfDistillationDataCollator
 tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 collator=MultilingualSelfDistillationDataCollator(tokenizer,max_length=args.max_length,reason_first=False,student_enable_thinking=True,include_problem_en=True,include_reference_solution_en=True)
 student=teacher=student_attended=teacher_attended=0
 for i in indices:
  row=dict(rows[i],target_lang=a.language.upper(),problem_en=rows[i]['problem']);batch=collator([row])
  text=tokenizer.batch_decode(batch['student_prompts'],skip_special_tokens=False)[0]
  if tokenizer.pad_token:text=text.replace(tokenizer.pad_token,'')
  tokens=tokenizer(text,add_special_tokens=False,truncation=True,max_length=max(1,args.max_length-args.max_completion_length))['input_ids']
  student+=len(tokens);teacher+=batch['teacher_prompts'].shape[1]
  student_attended+=sum(t!=tokenizer.pad_token_id for t in tokens)
  teacher_attended+=int((batch['teacher_prompts']!=tokenizer.pad_token_id).sum())
 n=len(indices);budget=args.max_completion_length
 assert student+n*budget==ledger['student_processed_input_positions']
 assert teacher+n*budget==ledger['teacher_processed_input_positions']
 assert student_attended+ledger['valid_response_loss_positions']==ledger['student_attended_input_positions']
 assert teacher_attended+ledger['valid_response_loss_positions']==ledger['teacher_attended_input_positions']
 report={'scope':'Reconstructed fresh sampler prefix; not a direct per-microstep runtime ID log',
         'generator':'CPU SeedableRandomSampler, pinned library implementation, saved data_seed',
         'checkpoint_global_step':limit,'actual_observed_loss_calls':n,'data_seed':args.data_seed,
         'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
         'source_ids_in_reconstructed_order':[ids[i] for i in indices],
         'updates':[{**g,'problem_ids':[ids[i] for i in g['indices']]} for g in groups],
         'actual_student_teacher_processed_and_attended_position_totals_match':True,
         'student_prompt_position_sum':student,'teacher_prompt_position_sum':teacher,
         'packages':{n:importlib.metadata.version(n) for n in ['torch','accelerate','transformers']}}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as stream:json.dump(report,stream,indent=2)
 print(json.dumps({k:v for k,v in report.items() if k not in ['source_ids_in_reconstructed_order','updates','data_sha256']}))


if __name__=='__main__':main()
