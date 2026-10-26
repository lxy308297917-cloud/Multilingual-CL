from settings import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=read(REPORT/'DATA_ORDER_AUDIT.json');tasks=sorted(r['orders']['42']['total']['tasks']);colors=['#4477AA','#EE6677','#228833','#CCBB44'];fig,axs=plt.subplots(2,3,figsize=(15,8))
for row,seed in enumerate(('42','43')):
 bins=r['orders'][seed]['bins'];xs=np.arange(1,11)
 for col,kind in enumerate(('input_tokens','supervised_tokens')):
  ax=axs[row,col];bottom=np.zeros(10)
  for color,task in zip(colors,tasks):
   vals=np.array([b['tasks'].get(task,{}).get(kind,0)/b[kind]*100 for b in bins]);ax.bar(xs,vals,bottom=bottom,color=color,label=task);bottom+=vals
  ax.set_ylim(0,100);ax.set_xticks(xs);ax.set_xlabel('Training-progress decile');ax.set_ylabel('% tokens');ax.set_title(f'Order {seed}: {kind.replace("_"," ")}');ax.legend(fontsize=8)
 ax=axs[row,2];ax.plot(xs,[b['main_CE_mean_of_updates'] for b in bins],marker='o',color='black');ax.set_xlabel('Training-progress decile');ax.set_ylabel('Mean observed main CE');ax.set_title(f'Order {seed}: mixed-task loss (diagnostic)');ax.set_xticks(xs);ax.grid(alpha=.2)
fig.suptitle('Same sample multiset and within-update pairs; different positions on the same LR schedule');fig.tight_layout();fig.savefig(REPORT/'DATA_COMPOSITION.png',dpi=150);plt.close(fig)
if (REPORT/'RESULTS.json').exists():
 r=read(REPORT/'RESULTS.json');fig,axs=plt.subplots(1,3,figsize=(13,4.2))
 for ax,task in zip(axs,('sw_reading','en_reading','en_instruction')):
  for seed,color in [('42','#4477AA'),('43','#EE6677')]:
   xy=[(step,r['scores'].get(f'order{seed}-step{step}',{}).get(task,{}).get('score')) for step in (2146,4293)];xy=[(x,y) for x,y in xy if y is not None]
   if xy:ax.plot([x for x,y in xy],[y for x,y in xy],marker='o',label=f'Order {seed}',color=color)
  ax.set_title(task.replace('_',' '));ax.set_xticks([2146,4293]);ax.set_xlabel('Training update');ax.set_ylabel('F1 / percentage points');ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Frozen 256-item development diagnosis; midpoint is not a selected checkpoint');fig.tight_layout();fig.savefig(REPORT/'MIDPOINT_COMPARISON.png',dpi=150);plt.close(fig)
print('Plots written')
