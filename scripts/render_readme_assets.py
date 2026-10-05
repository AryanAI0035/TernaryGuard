#!/usr/bin/env python3
"""Render README diagrams and a chart from the final, recorded results."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets'
NAVY = '#18324b'
TEAL = '#087f8c'
BLUE = '#3479b5'
GRAY = '#53677b'
PALE = '#edf5f8'
AMBER = '#9d6000'
plt.rcParams['svg.hashsalt'] = 'ternaryguard-readme'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': NAVY,
                     'axes.labelcolor': NAVY, 'axes.edgecolor': '#c5d3de',
                     'xtick.color': GRAY, 'ytick.color': GRAY})


def base(title, subtitle, height=5):
    fig, ax = plt.subplots(figsize=(14, height), facecolor='white')
    ax.set(xlim=(0, 14), ylim=(0, height))
    ax.axis('off')
    ax.text(.4, height-.55, title, fontsize=22, fontweight='bold')
    ax.text(.4, height-1.0, subtitle, fontsize=11, color=GRAY)
    return fig, ax


def box(ax, x, y, w, h, title, detail, color=TEAL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.035,rounding_size=0.12',
                              facecolor=PALE, edgecolor='#c9dbe5', linewidth=1))
    ax.text(x+.17, y+h-.34, title, fontsize=12, fontweight='bold', color=color)
    ax.text(x+.17, y+h-.72, detail, fontsize=10, color=NAVY, va='top', linespacing=1.6)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=15,
                               linewidth=1.6, color=GRAY, connectionstyle='arc3'))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/(name+'.svg'), bbox_inches='tight', facecolor='white', metadata={'Date': None})
    svg = OUT/(name+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(OUT/(name+'.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def overview():
    fig, ax = base('From traffic statistics to a Nano prediction',
                   'One frozen classifier, checked in Python, workstation C and physical Arduino firmware.', 5.3)
    box(ax, .4, 1.9, 2.8, 1.65, 'Traffic measurements', '115 numerical statistics\nfrom the N-BaIoT dataset')
    box(ax, 3.65, 1.9, 2.8, 1.65, 'Prepare 20 features', 'Select, signed-log transform\nand normalize the values')
    box(ax, 6.9, 1.9, 2.8, 1.65, 'Ternary neural network', '20 → 64 → 32 → 11\nPacked weights: −1, 0, +1')
    box(ax, 10.15, 2.72, 3.4, 1.25, 'Workstation C', '69,040 test rows\nZero prediction disagreements')
    box(ax, 10.15, 1.22, 3.4, 1.25, 'Physical Arduino Nano', '5,855 unique test rows\nZero prediction disagreements')
    arrow(ax, (3.23, 2.72), (3.58, 2.72))
    arrow(ax, (6.48, 2.72), (6.83, 2.72))
    arrow(ax, (9.73, 2.94), (10.08, 3.2))
    arrow(ax, (9.73, 2.5), (10.08, 1.95))
    ax.text(.4, .8, 'Output: one of 11 classes — benign traffic or a known botnet attack type.', fontsize=12)
    ax.text(.4, .38, 'Dataset statistics are the inputs; live packet capture and feature extraction are outside this prototype.', fontsize=10, color=GRAY)
    save(fig, 'system-overview')


def ternary():
    fig, ax = base('Three weight values, three simple operations',
                   'The packed dot-product loop adds, subtracts or skips integer-scaled activations.', 4.8)
    box(ax, .4, 1.72, 3.95, 1.52, '+1  →  ADD', 'Keep the input\nExample: +4')
    box(ax, 5.02, 1.72, 3.95, 1.52, '0  →  SKIP', 'Ignore the input\nExample: 0', GRAY)
    box(ax, 9.64, 1.72, 3.95, 1.52, '−1  →  SUBTRACT', 'Subtract the input\nExample: −5', AMBER)
    ax.text(.4, 1.02, 'Example: inputs [4, 2, 5, 3] · weights [+1, 0, −1, +1] = 4 + 0 − 5 + 3 = 2', fontsize=12)
    ax.text(.4, .6, 'Each weight uses 2 bits: 00 = 0, 01 = +1, 10 = −1. Code 11 is rejected.', fontsize=11, color=GRAY)
    ax.text(.4, .2, 'Illustrative integer example. Preprocessing, RMSNorm and final scaling still use floating point.', fontsize=10, color=GRAY)
    save(fig, 'ternary-explained')


def comparison():
    active = json.loads((ROOT/'model/active_model.json').read_text())
    with (ROOT/'results.csv').open(newline='') as f:
        rows = {r['model_type']: r for r in csv.DictReader(f) if r['run_id']==active['run_id']}
    assert set(rows)=={'fp32','int8','ternary'}
    types=['fp32','int8','ternary'];names=['FP32 baseline','INT8 weight simulation','Ternary model']
    sizes=[int(rows[t]['model_size_bytes']) for t in types]
    accuracies=[float(rows[t]['accuracy'])*100 for t in types]
    assert sizes==[15484,4456,1696]
    fig, axes=plt.subplots(1,2,figsize=(14,4.8),facecolor='white',gridspec_kw={'width_ratios':[1.1,1]})
    fig.subplots_adjust(left=.20,right=.97,top=.73,bottom=.22,wspace=.47)
    fig.text(.04,.91,'Smaller storage comes with an accuracy tradeoff',fontsize=20,fontweight='bold')
    fig.text(.04,.82,'Same 20 → 64 → 32 → 11 architecture and held-out-device test; final run only.',fontsize=11,color=GRAY)
    colors=[BLUE,'#8b9dab',TEAL]
    for ax,values,limit,title,fmt in [(axes[0],sizes,19000,'Model parameter storage (bytes)',lambda v:f'{v:,} B'),
                                      (axes[1],accuracies,106,'Test accuracy (%)',lambda v:f'{v:.2f}%')]:
        ax.barh(names,values,color=colors,height=.55)
        ax.invert_yaxis();ax.set_xlim(0,limit);ax.set_xlabel(title,fontsize=10)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,labelsize=10)
        ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(v+limit*.012,i,fmt(v),va='center',fontsize=10,fontweight='bold')
    axes[1].set_yticklabels([])
    fig.text(.04,.1,'Storage excludes firmware and runtime RAM. INT8 uses simulated quantized weights with FP32 execution.',fontsize=10,color=GRAY)
    fig.text(.04,.045,'Known weakness: ternary BASHLITE TCP recall is 1 / 5,555 (0.018%). Aggregate accuracy does not resolve it.',fontsize=10,color=AMBER)
    save(fig, 'model-tradeoff')
    (OUT/'README.md').write_text('''# README visuals

`system-overview` and `ternary-explained` are explanatory diagrams, not hardware photographs or measured traces. `model-tradeoff` reads only the active final run in the root results.csv. Parameter storage excludes firmware/RAM; INT8 is weight simulation with FP32 execution.

Regenerate SVG and PNG versions with `python3 scripts/render_readme_assets.py`. SVGs are used in the GitHub README; PNGs allow visual inspection. No legacy exploratory images are used.
''')


if __name__=='__main__':
    overview();ternary();comparison()
    print('Rendered 3 README visuals (SVG + PNG) from diagrams and final recorded results.')
