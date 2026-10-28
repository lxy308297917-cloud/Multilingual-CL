"""Scientific figures from verified real caches only; no plot-driven selection."""
from pathlib import Path
import sys,os,json,csv,hashlib,collections
ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'docs/results/representation_drift_functional_forgetting_20261028'
WORK=ROOT/'work/representation_drift_functional_forgetting_20261028'
DEPS=Path(__file__).resolve().parents[1]/'plot_dependencies';sys.path.insert(0,str(DEPS))
os.environ.setdefault('MPLCONFIGDIR',str(WORK/'matplotlib_config'))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,trustworthiness
from sklearn.neighbors import NearestNeighbors
from metrics import functional_state,paired_metrics

def read(path):return json.loads(path.read_text())
def records(path):return [json.loads(line) for line in path.open()]
def save(fig,name):fig.tight_layout();fig.savefig(REPORT/name,dpi=180);plt.close(fig)

def main():
 analysis=read(REPORT/'DRIFT_FORGETTING_ANALYSIS.json');assert analysis['status']=='completed_primary_association_analysis'
 protocol=read(REPORT/'PROTOCOL_FROZEN.json');models=protocol['models'];adapted=models[1:]
 assert (WORK/'MODEL_CACHE_IDENTITIES_VERIFIED.json').exists()
 rows=list(csv.DictReader((REPORT/'DRIFT_LAYERWISE.csv').open()))
 tasks=[('belebele','en'),('knowledge','en'),('gsm8k','en'),('belebele','sw'),('knowledge','sw')]
 colors={'Dense':'#4477AA','Element':'#EE6677','Protected':'#228833','Base':'#888888'}
 for field,title,filename in [('cosine_drift','1 - cosine(Base, adapted)','LAYERWISE_COSINE_DRIFT.png'),('normalized_l2','||adapted - Base|| / ||Base||','LAYERWISE_NORMALIZED_L2.png')]:
  fig,axes=plt.subplots(2,3,figsize=(13,7))
  for ax,(task,lang) in zip(axes.flat,tasks):
   for model in adapted:
    subset=sorted([r for r in rows if r['model']==model and (r['task'],r['language'])==(task,lang) and r['phase']=='confirmation' and r['state']=='All' and r['position']=='0'],key=lambda r:int(r['block']))
    x=[int(r['block']) for r in subset];ax.plot(x,[float(r[field]) for r in subset],label=model,color=colors[model])
    ax.fill_between(x,[float(r[field+'_lower']) for r in subset],[float(r[field+'_upper']) for r in subset],color=colors[model],alpha=.14)
   ax.set_title(task+' '+lang);ax.set_xlabel('Decoder block (post-block residual)');ax.set_ylabel(title);ax.grid(alpha=.15)
  axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center')
  fig.suptitle('Fixed confirmation groups; primary decision position; source-bootstrap 95% CI')
  save(fig,filename)
 meta=records(Path(protocol['input_manifest']['path']));base_preds=records(WORK/'Base/first/PREDICTIONS_CONTENT_V2.jsonl');base=np.load(WORK/'Base/first/REPRESENTATIONS.npy',mmap_mode='r')
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,model in zip(axes,adapted):
  arr=np.load(WORK/model/'first/REPRESENTATIONS.npy',mmap_mode='r');pred=records(WORK/model/'first/PREDICTIONS_CONTENT_V2.jsonl')
  # Mean-all-layer per-item primary cosine; descriptive distribution, no causal decomposition.
  drift=np.zeros(len(meta))
  for block in range(28):drift+=paired_metrics(base[:,block,0],arr[:,block,0])['cosine_drift']/28
  states=[functional_state(b['correct'],a['correct']) for b,a in zip(base_preds,pred)]
  values=[];labels=[]
  for state in ('Retained','Forgotten','Acquired','Still Wrong'):
   ix=[i for i,r in enumerate(meta) if r['phase']=='confirmation' and r['language']=='en' and r['task']!='gsm8k' and states[i]==state]
   if ix:values.append(drift[ix]);labels.append(state+'\nn='+str(len(ix)))
  if values:ax.boxplot(values,tick_labels=labels,showfliers=False)
  ax.set_title(model);ax.set_ylabel('Mean decision cosine drift, all 28 blocks');ax.tick_params(axis='x',rotation=15)
 fig.suptitle('Observed functional states; English MC confirmation examples; no claim of stored-knowledge loss')
 save(fig,'FUNCTIONAL_STATE_DRIFT_DISTRIBUTIONS.png')
 fig,ax=plt.subplots(figsize=(8,4));positions=[];labels=[]
 for i,model in enumerate(adapted):
  v=analysis['prediction'][model];positions.append(i)
  if v['status']!='evaluated':labels.append(model+' (insufficient classes)');continue
  effect=v['primary_drift_over_controls'];d=effect['difference'];lo,hi=effect['paired95CI'];labels.append(model)
  ax.hlines(i,lo,hi,color=colors[model]);ax.plot(d,i,'o',color=colors[model])
 ax.axvline(0,color='black',lw=.8);ax.set_yticks(positions,labels);ax.set_xlabel('Confirmation AUROC gain: controls + drift minus controls');ax.grid(axis='x',alpha=.15)
 ax.set_title('Controls include task, length, Base confidence and activation norm change')
 save(fig,'DRIFT_PREDICTION_AUROC_GAIN.png')
 # All models share exactly the same selected input IDs and ONE embedding per layer/seed.
 subset=[]
 for task,lang in [('belebele','en'),('belebele','sw'),('knowledge','en'),('knowledge','sw')]:
  candidates=[i for i,r in enumerate(meta) if (r['task'],r['language'],r['phase'])==(task,lang,'confirmation')]
  candidates.sort(key=lambda i:hashlib.sha256(('tsne-42:'+meta[i]['id']).encode()).hexdigest());subset+=candidates[:32]
 arrays={m:np.load(WORK/m/'first/REPRESENTATIONS.npy',mmap_mode='r') for m in models}
 predictions={m:records(WORK/m/'first/PREDICTIONS_CONTENT_V2.jsonl') for m in models}
 report=dict(samples_per_model=len(subset),joint_models=models,input_ids=[meta[i]['id'] for i in subset],
  parameters=protocol['tsne'],layer_results={},not_a_forgetting_metric=True)
 language_colors={'en':'#4477AA','sw':'#EE6677'};state_colors={'Retained':'#228833','Forgotten':'#CC3311','Acquired':'#4477AA','Still Wrong':'#AA3377','Base':'#BBBBBB'}
 for seed in (42,43):
  fig,axes=plt.subplots(3,3,figsize=(12,11))
  for row,block in enumerate((0,13,27)):
   x=np.concatenate([arrays[m][subset,block,0] for m in models]).astype(np.float64)
   x=(x-x.mean(0))/np.maximum(x.std(0),1e-12)
   reduced=PCA(n_components=50,random_state=42,svd_solver='full').fit_transform(x)
   fit=TSNE(n_components=2,perplexity=30,learning_rate=200,init='pca',n_iter=1000,random_state=seed)
   coords=fit.fit_transform(reduced)
   np.savez(WORK/f'TSNE_block{block}_seed{seed}.npz',coordinates=coords,input_indices=np.array(subset),models=np.array(models))
   report['layer_results'][f'{block}/seed{seed}']=dict(trustworthiness_10=float(trustworthiness(reduced,coords,n_neighbors=10)),kl_divergence=float(fit.kl_divergence_))
   offset=0
   for model in models:
    xy=coords[offset:offset+len(subset)];axes[row,0].scatter(xy[:,0],xy[:,1],s=8,alpha=.6,label=model,color=colors[model])
    for lang in ('en','sw'):
     ix=[j for j,i in enumerate(subset) if meta[i]['language']==lang]
     axes[row,1].scatter(xy[ix,0],xy[ix,1],s=8,alpha=.5,color=language_colors[lang],label=lang if model=='Base' else None)
    states=['Base' if model=='Base' else functional_state(base_preds[i]['correct'],predictions[model][i]['correct']) for i in subset]
    for state in dict.fromkeys(states):
     ix=[j for j,s in enumerate(states) if s==state];axes[row,2].scatter(xy[ix,0],xy[ix,1],s=8,alpha=.5,color=state_colors.get(state,'black'),label=state)
    offset+=len(subset)
   for col,title in enumerate(('Model identity','Input language','Observed transition (Base gray)')):
    ax=axes[row,col];ax.set_title(f'Block {block}: {title}');ax.set_xticks([]);ax.set_yticks([])
    h,l=ax.get_legend_handles_labels();unique=dict(zip(l,h));ax.legend(unique.values(),unique.keys(),fontsize=7)
  fig.suptitle(f'Joint t-SNE, fixed IDs, seed {seed}; coordinates are NOT forgetting scores')
  save(fig,f'TSNE_FIXED_LAYERS_seed{seed}.png')
 write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 for block in (0,13,27):
  xy=[np.load(WORK/f'TSNE_block{block}_seed{seed}.npz')['coordinates'] for seed in (42,43)]
  neighbors=[NearestNeighbors(n_neighbors=11).fit(x).kneighbors(x,return_distance=False)[:,1:] for x in xy]
  overlap=[len(set(a)&set(b))/10 for a,b in zip(*neighbors)]
  report['layer_results'][f'{block}/seed_sensitivity']=dict(mean_10NN_overlap=float(np.mean(overlap)),
   definition='Neighborhood overlap after independent seed42/43 joint fits; no coordinate-distance comparison',
   interpretation='Low overlap signals visualization instability; never changes primary statistical gate')
 write(REPORT/'TSNE_DIAGNOSTIC.json',report)

if __name__=='__main__':main()
