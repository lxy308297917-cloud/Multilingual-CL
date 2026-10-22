"""Export the frozen paired task effects; no selection or mixed-task score."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/representation_alignment_report'
report = json.loads((OUT / 'COMPARISON.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for ax, task, title in zip(axes, ['en_to_sw', 'sw_reading_kenswquad'], ['EN to SW: delta chrF++', 'SW reading: delta token F1 (points)']):
    for y, (method, color) in enumerate([('C', '#355c9a'), ('D', '#ba6424')]):
        value = report['paired_deltas'][method + '-B'][task]
        point = value['delta_left_minus_right']
        lo, hi = value['paired95percentCI']
        ax.errorbar(point, 1-y, xerr=[[point-lo], [hi-point]], fmt='o', color=color, capsize=5, markersize=7)
        ax.annotate(f'{point:+.2f} [{lo:+.2f}, {hi:+.2f}]', (point, 1-y), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=9)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    if task == 'sw_reading_kenswquad':
        ax.axvline(-2, color='#777777', linestyle=':', linewidth=1, label='Noninferiority bound: -2')
        ax.legend(loc='lower left', fontsize=8)
    ax.set_yticks([1, 0], ['Cosine C - CE B', 'InfoNCE D - CE B'])
    ax.set_ylim(-.6, 1.6)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='x', alpha=.2)
fig.suptitle('Explicit alignment versus the same-pool translation CE control', fontsize=13)
fig.text(.5, .025, 'One seed; fixed 4,293-step endpoint. Paired source-cluster 95% CIs, 10,000 resamples. Development only.', ha='center', fontsize=9)
fig.tight_layout(rect=[0, .07, 1, .92])
fig.savefig(OUT / 'FINAL_TASK_EFFECTS.png', dpi=200)
fig.savefig(OUT / 'FINAL_TASK_EFFECTS.pdf')
plt.close(fig)
