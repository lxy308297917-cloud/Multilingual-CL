"""Standalone scientific figure; use existing plot-env, not training environment installs."""
import pathlib,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(__file__).resolve().parents[1]
d=json.loads((root/'DOWNSTREAM_COMPARISON.json').read_text());r=json.loads((root/'REPRESENTATION_COMPARISON.json').read_text());tasks=['en_to_sw','sw_to_en','sw_reading','en_reading','en_instruction'];labels=['EN→SW chrF++','SW→EN chrF++','SW reading F1','EN reading accuracy','EN instruction strict'];fig,ax=plt.subplots(figsize=(9,4.5))
for j,(control,color) in enumerate([('B','#1765a3'),('D0','#e07b22')]):
 vals=[d['comparisons']['D1-minus-'+control][t] for t in tasks];x=np.array([v['difference'] for v in vals]);low=np.array([v['low'] for v in vals]);high=np.array([v['high'] for v in vals]);y=np.arange(5)+(j-.5)*.16;ax.errorbar(x,y,xerr=np.array([x-low,high-x]),fmt='o',capsize=3,color=color,label='D1 − '+control)
ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(np.arange(5),labels);ax.invert_yaxis();ax.set_xlabel('Difference in metric points; source-paired 95% bootstrap CI');ax.set_title('Fixed one-seed development endpoint; mathematics excluded');ax.grid(axis='x',alpha=.2);ax.legend();fig.tight_layout();fig.savefig(root/'DOWNSTREAM_DIFFERENCES.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,7));names=['Base','B','D0','D1'];colors=['#aaaaaa','#1765a3','#e07b22','#34834b']
for ax,kind,field,title in [(axes[0,0],'fixed_Base_EN','bidirectional_P1','Primary diagnostic: fixed Base-EN ↔ current-SW (%)'),(axes[1,0],'current_EN_SW','bidirectional_P1','Secondary: current EN ↔ SW retrieval (%)'),(axes[1,1],'fixed_Base_EN','positive_negative_gap','Fixed-anchor positive − off-pair cosine gap')]:
 ax.bar(names,[r['models'][n][kind][field] for n in names],color=colors);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
ax=axes[0,1];xx=np.arange(4);width=.36
for j,(field,label,color) in enumerate([('ENtoSW_P1','EN → SW','#6f629f'),('SWtoEN_P1','SW → EN','#359e92')]):
 ax.bar(xx+(j-.5)*width,[r['models'][n]['fixed_Base_EN'][field] for n in names],width,label=label,color=color)
ax.set_xticks(xx,names);ax.set_title('Fixed-anchor retrieval: directional asymmetry (%)',fontsize=10);ax.legend();ax.grid(axis='y',alpha=.2)
fig.suptitle('Layer10 mean pooling; fixed independent1000-pair candidate pool',fontsize=11);fig.tight_layout();fig.savefig(root/'REPRESENTATION_ENDPOINT.png',dpi=180)
