"""Standalone frozen comparison figures; no model or metric selection."""
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parents[1];d=json.loads((p/'DOWNSTREAM_COMPARISON.json').read_text());c=json.loads((p/'TRAINING_COST_AND_DISPLACEMENT.json').read_text());s=json.loads((p/'UPDATE_STRENGTH.json').read_text());new='S-All-LR077';tasks=['en_to_sw','sw_to_en','sw_reading','en_reading','en_instruction'];labels=['EN→SW chrF++','SW→EN chrF++','SW reading F1','EN reading content accuracy','EN instruction strict'];fig,ax=plt.subplots(figsize=(9.5,4.8))
for j,(key,label,color) in enumerate([(new+'-minus-S-All','LR077 − S-All','#2875ab'),('S-Attn-minus-'+new,'S-Attn − LR077','#d97b21')]):
 vals=[d['comparisons'][key][t] for t in tasks];x=np.array([v['difference'] for v in vals]);lo=np.array([v['low'] for v in vals]);hi=np.array([v['high'] for v in vals]);ax.errorbar(x,np.arange(5)+(j-.5)*.17,xerr=np.array([x-lo,hi-x]),fmt='o',capsize=3,label=label,color=color)
ax.axvline(0,color='black',lw=.8);ax.set_yticks(np.arange(5),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_xlabel('Metric-point difference; source-paired 95% bootstrap CI');ax.set_title('One-seed follow-up diagnostic; complete frozen endpoint');ax.legend();fig.tight_layout();fig.savefig(p/'TASK_CONTRASTS.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,4));names=['S-All','S-Attn','S-MLP',new];colors=['#777777','#2875ab','#d97b21','#428752'];fields=[('cumulative_path_length','Sum of actual BF16 update norms'),('final_base_delta_L2','Final net displacement from Base (L2)'),('cumulative_nonzero_write_events','Nonzero coordinate-write events')]
for ax,(field,title) in zip(axes,fields):
 ax.bar(names,[c['models'][n][field] for n in names],color=colors);ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
axes[0].axhspan(*s['range'],color='#428752',alpha=.12,label='Frozen ±10% path band');axes[0].axhline(s['reference_target_rounded'],color='black',ls='--',lw=.8);axes[0].legend(fontsize=8);fig.suptitle('Same K: only cumulative path is targeted; other update properties can differ',fontsize=11);fig.tight_layout();fig.savefig(p/'ACTUAL_UPDATES.png',dpi=180)
