# utils/metrics.py
import os
import matplotlib.pyplot as plt

def plot_metrics(results, save_dir=None, show_inline=True):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'loss_curves.png')
    else:
        filepath = 'loss_curves.png'

    plt.figure(figsize=(12, 6))
    for i, tag in enumerate(['train_losses', 'val_losses']):
        plt.subplot(1, 2, i + 1)
        for fold, fdet in results['folds'].items():
            plt.plot(fdet[tag], label=f"Fold {fold}")
        plt.plot(results[f'avg_{tag}'], 'k--', linewidth=2, label='Avg')
        plt.title(tag)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.savefig(filepath)

    if show_inline:
        plt.show()
    else:
        plt.close()

    return filepath