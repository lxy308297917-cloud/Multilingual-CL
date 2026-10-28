"""Freeze complete benchmark scope and perform train/calibration content overlap audit."""
from settings import *
import collections,re,unicodedata,pyarrow.parquet as pq
from datasets import Dataset

assert not (REPORT/'EVALUATION_FROZEN.json').exists()
out=RUN/'evaluation_inputs';out.mkdir(exist_ok=True)
old=ROOT/'outputs/benchmark_direction_20261026'
bridge=read(old/'bridge/BENCHMARK_BRIDGE_PROTOCOL.json');sources={}
bench=[]
def jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def save(name,rows):
 p=out/(name+'.jsonl');p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));sources[name]=dict(path=str(p),n=len(rows),sha256=sha(p));return p
for lang,r in bridge['source_files'].items():
 rows=jsonl(r['path']);assert sha(r['path'])==r['sha256'];save('belebele_'+lang,rows)
 bench.extend((f'belebele_{lang}',str(i),[str(r.get(k,'')) for k in ('flores_passage','question')]) for i,r in enumerate(rows))
enroot=H/'ssu-migration/data/ig-baseline-eval-public/mmlu'
swroot=ROOT/'work/benchmark_direction_20261026/bridge/global_mmlu_sw/sw'
swtest=pq.read_table(swroot/'test-00000-of-00001.parquet').to_pylist();swdev=pq.read_table(swroot/'dev-00000-of-00001.parquet').to_pylist()
swmap={r['sample_id']:r for r in swtest};knowledge={l:[] for l in ('en','sw')};source_hashes={}
for subject in sorted({r['subject'] for r in swtest}):
 tests=enroot/subject/'test-00000-of-00001.parquet';dev=enroot/subject/'validation-00000-of-00001.parquet'
 english=pq.read_table(tests).to_pylist();shots=pq.read_table(dev).to_pylist()[:5]
 swshots=sorted([r for r in swdev if r['subject']==subject],key=lambda r:int(r['sample_id'].split('/')[-1]))
 assert len(shots)==len(swshots)==5
 for lang,examples in [('en',shots),('sw',swshots)]:
  formatted=[]
  for r in examples:
   formatted.append(dict(subject=subject,question=r['question'],choices=r['choices'] if lang=='en' else [r['option_'+c] for c in 'abcd'],answer='ABCD'[r['answer']] if lang=='en' else r['answer']))
  folder=out/'knowledge'/lang/subject;folder.mkdir(parents=True,exist_ok=True)
  write(folder/'validation.json',formatted)
 for i,r in enumerate(english):
  key=f'{subject}/test/{i}';s=swmap[key];assert s['answer']=='ABCD'[r['answer']]
  for lang,row in [('en',dict(id=key,subject=subject,question=r['question'],choices=r['choices'],answer='ABCD'[r['answer']])),('sw',dict(id=key,subject=subject,question=s['question'],choices=[s['option_'+c] for c in 'abcd'],answer=s['answer']))]:
   knowledge[lang].append(row);bench.append(('knowledge_'+lang,key,[row['question']]))
  for lang in ('en','sw'):
   pass
 for lang in ('en','sw'):write(out/'knowledge'/lang/subject/'test.json',[r for r in knowledge[lang] if r['subject']==subject])
 source_hashes[str(tests)]=sha(tests);source_hashes[str(dev)]=sha(dev)
assert len(knowledge['en'])==len(knowledge['sw'])==14042
for l in knowledge:save('knowledge_'+l,knowledge[l])
flores=H/'ssu-migration/data/multilingual_cl_v3_2_lowresource_bn_te_sw/eval_data/flores200'
mt=[]
for task,folder in [('en_to_sw','eng_Latn-swh_Latn'),('sw_to_en','swh_Latn-eng_Latn')]:
 p=flores/folder/'test.jsonl';rows=jsonl(p);assert len(rows)==1012
 source_hashes[str(p)]=sha(p)
 for r in rows:
  instruction='Translate the following English text into Swahili (Kiswahili). Output only the translation.' if task=='en_to_sw' else 'Translate the following Swahili (Kiswahili) text into English. Output only the translation.'
  row=dict(id=task+':'+r['id'],group_id=r['id'],task=task,prompt=instruction+'\n\n'+r['source_text'],source_text=r['source_text'],reference=r['target_text'])
  mt.append(row);bench.append((task,r['id'],[r['source_text'],r['target_text']]))
save('translation',mt)
c=read(ROOT/'work/benchmark_direction_20261026/bridge/legacy_config.json')
ife=jsonl(c['ifeval_data']);assert len(ife)==541;save('ifeval',ife)
bench.extend(('ifeval',str(r['key']),[r['prompt']]) for r in ife)
gsm=Dataset.from_file(c['gsm8k_data']).to_list();assert len(gsm)==1319;save('gsm8k',gsm)
bench.extend(('gsm8k',str(i),[r['question']]) for i,r in enumerate(gsm))
source_hashes[str(swroot/'test-00000-of-00001.parquet')]=sha(swroot/'test-00000-of-00001.parquet')
def tokens(s):return re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold())
def grams(s):
 w=tokens(s);return {' '.join(w[i:i+16]) for i in range(len(w)-15)}
reference=collections.defaultdict(set);exact=collections.defaultdict(set)
for task,key,texts in bench:
 for text in texts:
  norm=' '.join(tokens(text))
  if len(tokens(text))>=5:exact[norm].add(task+':'+key)
  for gram in grams(text):reference[gram].add(task+':'+key)
frozen=read(REPORT/'DATA_FROZEN.json');indices=set(read(RUN/'INDICES.json'));collisions=[];same_group={};dupes=collections.Counter()
with Path(frozen['source']).open() as f:
 for i,s in enumerate(f):
  if i not in indices:continue
  r=json.loads(s);matches=set()
  for field in ('prompt','answer'):
   text=str(r.get(field,''));norm=' '.join(tokens(text));matches.update(exact.get(norm,()))
   for gram in grams(text):matches.update(reference.get(gram,()))
  if matches:collisions.append(dict(where='main',id=r['id'],matches=sorted(matches)))
  key=r.get('group_key',r['dedup_group']);dupes[key]+=1;same_group[key]=r['bucket']
for r in read(RUN/'CALIBRATION_POOL.json'):
 matches=set()
 for field in ('prompt','answer'):
  text=str(r.get(field,''));matches.update(exact.get(' '.join(tokens(text)),()))
  for gram in grams(text):matches.update(reference.get(gram,()))
 if matches:collisions.append(dict(where='calibration',id=r['id'],matches=sorted(matches)))
write(REPORT/'OVERLAP_AUDIT.json',dict(passed=not collisions,collisions=collisions,
 rule='NFKC/casefold normalized exact content>=5words OR common16 consecutive words; prompt/answer against full test content',
 main_unique_source_rows=len(indices),calibration_rows=len(read(RUN/'CALIBRATION_POOL.json')),
 repeated_content_groups=sum(v>1 for v in dupes.values()),max_presentations_per_content_group=max(dupes.values()),
 cross_language_dual_direction_may_share_original_group=True,
 limitations=['Lexical screening is not proof of semantic decontamination.','Base pretraining unknown.','Parallel benchmark and previously exposed test material are supplementary benchmark comparisons.']))
write(REPORT/'EVALUATION_FROZEN.json',dict(tasks={'belebele_en':900,'belebele_sw':900,'knowledge_en':14042,'knowledge_sw':14042,'en_to_sw':1012,'sw_to_en':1012,'ifeval':541,'gsm8k':1319},
 sources=sources,original_source_hashes=source_hashes,
 reading='MCF3shot, raw label likelihood ACC; exact same-question and source-link-disjoint patched sampler',
 knowledge='Full57subjects14042, MCF5shot official subject dev; raw label likelihood microACC and macro subjectACC',
 translation='Entire FLORES devtest/test1012,0shot fixed explicit translation prompt,corpuschrF++6char2word beta2',
 instruction='Entire officialIFEval541, official strict prompt/instruction checks',
 mathematics='Entire GSM8K1319; frozen lm_eval0.4.8 gsm8k5shot; primary official strict-match, flexible-match secondary, raw answers retained',
 generation=dict(do_sample=False,num_beams=1,translation_max_new_tokens=256,ifeval_max_new_tokens=1280,gsm8k_max_new_tokens=1024,batch=4),
 no_test_checkpoint_selection=True,old_exposed_material_not_independent_final_confirmation=True,created_unix=time.time()))
assert not collisions,'Overlap found: inspect before training; no silent data replacement'
print('Full benchmark input identities frozen; overlap audit passes')
