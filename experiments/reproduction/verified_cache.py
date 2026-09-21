"""Read-only current-protocol completion check; never launches GPU evaluation."""
from pathlib import Path
import ast,json,sys,subprocess,math
E=Path(__file__).resolve().parents[1];R=E.parent/'ssu-main';sys.path.insert(0,str(R/'scripts'))
from run_baseline_eval import model_identity,digest,sha,valid_done,protocol,jobs

def unique_done(paths,ident):
 found=[p for p in paths if (p/'DONE.json').exists()]
 if len(found)>1:raise RuntimeError('Multiple completed attempts for same unit')
 if not found:return None
 assert valid_done(found[0],ident);return found[0]

def verify(language,model):
 model=Path(model).resolve();evidence=[];missing=[]
 if language=='sw':
  W=E/'sw_fineweb2_aya_v1';source=W/'scripts/evaluate_cl.py';tree=ast.parse(source.read_text());nodes=[]
  for node in tree.body:
   if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='ArgumentParser':break
   nodes.append(node)
  else:raise RuntimeError('Frozen SW module boundary changed')
  ns={'__file__':str(source),'__name__':'sw_cache_verifier'};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
  prot=ns['protocol']();assert prot==json.loads((W/'EVAL_PROTOCOL_LOCK.json').read_text());c=ns['C'];expected=c['tasks'];mid=model_identity(model,c);dest=Path(c['output_root'])/digest(mid)[:16]
 else:
  cp=R/'configs/ig_baseline_eval_v3.json';c=json.loads(cp.read_text());prot=protocol(c,cp);assert prot==json.loads((Path(c['output_root'])/'protocol_lock.json').read_text());mid=model_identity(model,c);dest=Path(c['output_root'])/'evaluation'/digest(mid)[:16]/'primary';expected=[j for j in jobs(c,False) if j['name'] not in ['gsm8k','mmlu_en','gmmlu_ig']]
 for j in expected:
  ident={'protocol':digest(prot),'model':digest(mid),'job':j,'smoke':False};found=unique_done((dest/(j['name']+'-seed42')).glob('attempt-*'),ident)
  if found:evidence.append({'task':j['name'],'done':str(found/'DONE.json'),'done_sha256':sha(found/'DONE.json')})
  else:missing.append(j['name'])
 if language=='sw':
  diag=W/'diagnostics';p=json.loads((diag/'protocol.json').read_text());assert p['code_sha256']==sha(diag/'evaluate_cl.py')
  for lang,path in {'en':'/root/autodl-tmp/eval_datasets_local/sum_ssu/en/test.jsonl','sw':'/root/autodl-tmp/sw_fineweb2_aya_v1/eval_data/sum_sw_test.jsonl'}.items():assert p['source_sha256'][lang]==sha(Path(path))
  weights={k:v for k,v in mid['files'].items() if k.endswith('.safetensors')};unit=diag/digest(weights)[:16];done=unit/'DONE.json'
  if not done.exists():missing.append('ppl')
  else:
   d=json.loads(done.read_text());assert d['weights']==weights and d['protocol']==digest(p);assert sha(unit/'scores.json')==d['scores_sha256'];scores=json.loads((unit/'scores.json').read_text());details=json.loads((unit/'details.json').read_text())
   for lang in ['en','sw']:
    rr=[r for r in details if r['language']==lang];assert len(rr)==16 and sorted(r['block'] for r in rr)==list(range(16));assert all(r['scored_tokens']==511 for r in rr);nll=sum(r['nll'] for r in rr)/16
    assert abs(nll-scores[lang+'_nll'])<1e-10 and math.isclose(math.exp(nll),scores[lang+'_ppl'],rel_tol=1e-12)
   evidence.append({'task':'ppl','done':str(done),'done_sha256':sha(done),'details_sha256':sha(unit/'details.json')})
 G=E/'ig_gsm8k_harness048_v1';gc=json.loads((R/'configs/ig_gsm8k_harness048_v1.json').read_text());gp=json.loads((G/'protocol_lock.json').read_text())
 code="import sys,json;sys.path.insert(0,"+repr(str(G/'scripts'))+");import run;print(json.dumps(run.identity()))"
 actual=json.loads(subprocess.check_output([gc['evaluation_python'],'-c',code],text=True));assert actual==gp
 gm=model_identity(model,gc);ident={'protocol':digest(gp),'model':digest(gm),'job':{'name':'gsm8k','seed':42,'kind':'general','generative':True},'smoke':False};found=unique_done((G/'evaluation'/digest(gm)[:16]).glob('attempt-*'),ident)
 if found:
  assert json.loads((found/'contract_audit.json').read_text())['passed'];evidence.append({'task':'gsm8k','done':str(found/'DONE.json'),'done_sha256':sha(found/'DONE.json')})
 else:missing.append('gsm8k')
 return {'language':language,'model':str(model),'complete':not missing,'missing':missing,'verified_units':evidence,'verification':'live model/tokenizer identities, current protocol code/data/environment, all frozen DONE artifacts; PPL details recomputed; no GPU execution'}
