#!/usr/bin/env python3
"""Plot the final model's recorded training history for the README."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/benchmarks/phase3_final_seed42/ternary_2bit.json'
OUTPUT = ROOT / 'docs/assets/training-loss.png'


def main():
    report = json.loads(SOURCE.read_text())
    history = report['history']
    epochs = [row['epoch'] for row in history]
    train_loss = [row['train_loss'] for row in history]
    val_loss = [row['val_loss'] for row in history]
    checkpoint_epoch = report['best_epoch']
    if not epochs or checkpoint_epoch not in epochs or epochs != sorted(set(epochs)):
        raise ValueError('Invalid recorded epoch sequence')

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.plot(epochs, train_loss, color='black', linewidth=1.4, label='Training')
    ax.plot(epochs, val_loss, color='0.45', linestyle='--', linewidth=1.4, label='Validation')
    ax.axvline(checkpoint_epoch, color='0.7', linestyle=':', linewidth=1,
               label=f'Saved checkpoint: epoch {checkpoint_epoch}')
    ax.set_title('Training and validation loss', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_ylim(0, max(train_loss + val_loss) * 1.08)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='0.9', linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'Plotted {len(history)} recorded epochs from {SOURCE.relative_to(ROOT)}')
    print(f'Saved checkpoint: epoch {checkpoint_epoch}; output: {OUTPUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
