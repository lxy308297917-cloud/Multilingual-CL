"""Standalone scientific figures; each task stays on its own scale."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from settings import *

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#777777','#3973ac','#2c8b68','#dc9638','#ba5965'];models=['Base','A','B','C','D'];tasks=['sw_reading','en_reading','en_math'];titles=['SW reading: token F1','EN reading: content accuracy','Math: extraction coverage ONLY'];allrows=json.loads((REPORT/'V2_SCORES.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.4),sharey=True)
    for ax,task,title in zip(axes,tasks,titles):
        points=[];errlo=[];errhi=[]
        for name in models:
            rr=[r for r in rows(RUN/'generation'/name/'PREDICTIONS.jsonl') if r['task']==task];key='content_f1' if task=='sw_reading' else 'content_correct';v=np.array([int(r['scores']['status']=='extracted') if task=='en_math' else r['scores'][key] for r in rr])*100;rng=np.random.default_rng(42);sample=rng.integers(0,len(v),size=(10000,len(v)));ci=np.quantile(v[sample].mean(1),[.025,.975]);point=v.mean();points.append(point);errlo.append(point-ci[0]);errhi.append(ci[1]-point)
        ax.bar(models,points,color=colors,yerr=[errlo,errhi],capsize=4,alpha=.9,hatch='//' if task=='en_math' else None);ax.set_title(title);ax.set_ylim(0,108);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if task=='en_math':ax.text(2.4,75,'SCORER FAILED\nNot ability accuracy',ha='center',color='#a13030',fontsize=11,fontweight='bold')
        for i,p in enumerate(points):ax.text(i,p+errhi[i]+2,f'{p:.1f}',ha='center',fontsize=9)
    axes[0].set_ylabel('Score / coverage (0–100)');fig.suptitle('Fresh development v2: same 192 source groups per model',fontweight='bold');fig.text(.5,.02,'n=64/task; bootstrap intervals. One malformed EN reading source flagged; denominator unchanged. No formal test.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(REPORT/'V2_COMPARISON.png',dpi=220);plt.close(fig)
    grad=json.loads((REPORT/'MECHANISM_GRADIENTS.json').read_text());states=['Base','B-mid','B','C-mid','C','D-mid','D'];fig,ax=plt.subplots(figsize=(10,4.5))
    for j,(kind,color,label) in enumerate(zip('BCD',colors[2:],['Translation CE','Cosine alignment','InfoNCE alignment'])):
        v=[next(r['scaled_aux_to_main_ratio'] for r in grad if r['state']==state and r['auxiliary_kind']==kind) for state in states];ax.plot(np.arange(len(states)),v,'o-',color=color,label=label)
    ax.set_xticks(np.arange(len(states)),states);ax.set_yscale('log');ax.set_ylabel('||0.1 × auxiliary gradient|| / ||main gradient||');ax.set_title('Same coefficient does not match auxiliary signal strength');ax.grid(axis='y',which='both',alpha=.2);ax.legend();fig.text(.5,.01,'One identical true main batch + eight true parallel pairs at each saved state; not a trajectory average.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(REPORT/'GRADIENT_STRENGTH.png',dpi=220);plt.close(fig)
    print('Two publication figures saved')
if __name__=='__main__':main()
