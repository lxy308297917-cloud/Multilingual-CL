"""Standalone one-seed task contrasts and update-strength figures."""
import pathlib,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=pathlib.Path(__file__).resolve().parents[1];d=json.loads((r/'DOWNSTREAM_COMPARISON.json').read_text());cost=json.loads((r/'TRAINING_COST_AND_DISPLACEMENT.json').read_text());tasks=['en_to_sw','sw_to_en','sw_reading','en_reading','en_instruction'];labels=['EN→SW chrF++','SW→EN chrF++','SW reading F1','EN reading accuracy','EN instruction strict'];fig,ax=plt.subplots(figsize=(9,4.5))
for j,(left,color) in enumerate([('S-Attn','#1765a3'),('S-MLP','#e07b22')]):
 rr=[d['comparisons'][left+'-minus-S-All'][t] for t in tasks];x=np.array([v['difference'] for v in rr]);lo=np.array([v['low'] for v in rr]);hi=np.array([v['high'] for v in rr]);y=np.arange(5)+(j-.5)*.16;ax.errorbar(x,y,xerr=np.array([x-lo,hi-x]),fmt='o',capsize=3,label=left+' − S-All',color=color)
ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(np.arange(5),labels);ax.invert_yaxis();ax.set_xlabel('Metric-point difference; source-paired95% bootstrap CI');ax.set_title('Equal77,070,336 selected coordinates/update; seed42 development screen');ax.legend();ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(r/'TASK_CONTRASTS.png',dpi=180);plt.close(fig)
names=['S-All','S-Attn','S-MLP'];fig,axes=plt.subplots(1,3,figsize=(11,3.8));colors=['#777777','#1765a3','#e07b22']
for ax,key,title in [(axes[0],'cumulative_path_length','Sum of actual BF16 update norms'),(axes[1],'final_base_delta_L2','Final displacement from Base (L2)'),(axes[2],'cumulative_nonzero_write_events','Nonzero coordinate-write events')]:
 ax.bar(names,[cost[n]['training'][key] for n in names],color=colors);ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.2)
fig.suptitle('Same nominalK; actual displacement and coverage can differ',fontsize=11);fig.tight_layout();fig.savefig(r/'ACTUAL_UPDATES.png',dpi=180)
