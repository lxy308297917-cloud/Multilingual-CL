"""Standalone research curves; retrieval is diagnostic, never the effectiveness endpoint."""
import json
from pathlib import Path
OUT=Path('/data0/users/lixiaoyang/models/checkpoint/representation_alignment_20261021')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv,shutil

def main():
 fig,axes=plt.subplots(1,2,figsize=(11,4));table=[]
 for folder in sorted((OUT/'diagnostics').iterdir()):
  p=folder/'diagnostic_metrics.json'
  if not p.exists():continue
  d=json.loads(p.read_text())
  for method in ('mean','last','weighted'):
   for i,v in enumerate(d[method]):table.append({'model':folder.name,'pooling':method,'layer':i,**v})
  vals=d['mean'];axes[0].plot(range(29),[100*(v['en_to_sw_p1']+v['sw_to_en_p1'])/2 for v in vals],label=folder.name);axes[1].plot(range(29),[v['positive_negative_gap'] for v in vals],label=folder.name)
 for ax,title in zip(axes,['Bidirectional retrieval P@1 (%)','Positive minus off-diagonal cosine']):ax.set_xlabel('HF hidden-state index');ax.set_title(title);ax.grid(alpha=.3);ax.legend()
 fig.tight_layout();fig.savefig(OUT/'ALIGNMENT_CURVES.png',dpi=180);plt.close(fig)
 with (OUT/'ALIGNMENT_METRICS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 destination=Path(__file__).resolve().parents[4]/'outputs/representation_alignment_report';destination.mkdir(parents=True,exist_ok=True)
 for name in ('ALIGNMENT_CURVES.png','ALIGNMENT_METRICS.csv'):shutil.copy2(OUT/name,destination/name)
if __name__=='__main__':main()
