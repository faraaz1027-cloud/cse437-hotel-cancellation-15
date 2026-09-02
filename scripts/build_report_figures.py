"""Report-only plots from frozen predictions and coefficients; no model fitting."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ['step13/evaluation_summary.json', 'step15/comparison_summary.json']:
        record = json.loads((ROOT / 'data/results' / name).read_text())
        for relative, expected in record['output_sha256'].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    lr = pd.concat([pd.read_csv(ROOT / f'data/results/step13/test_predictions_{i:02d}.csv.gz')
                    for i in range(1, 5)], ignore_index=True)
    rf = pd.read_csv(ROOT / 'data/results/step15/random_forest_test_probabilities.csv.gz')
    assert np.array_equal(lr.source_row_id, rf.source_row_id)
    assert np.array_equal(lr.actual, rf.actual)
    coefficients = pd.read_csv(ROOT / 'data/results/step13/feature_coefficients.csv').head(6)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for label, probability, color in [
        ('Logistic Regression', lr.cancellation_probability, '#187a87'),
        ('Random Forest', rf.random_forest_probability, '#b57532')]:
        fpr, tpr, _ = roc_curve(lr.actual, probability)
        axes[0, 0].plot(fpr, tpr, label=f'{label}: {roc_auc_score(lr.actual, probability):.3f}', color=color)
        precision, recall, _ = precision_recall_curve(lr.actual, probability)
        axes[0, 1].plot(recall, precision, label=label, color=color)
    axes[0, 0].plot([0, 1], [0, 1], '--', color='#9a9a9a', lw=1)
    axes[0, 0].set(title='ROC curves', xlabel='False positive rate', ylabel='True positive rate')
    axes[0, 0].legend(fontsize=8, frameon=False, loc='lower right')
    axes[0, 1].axhline(lr.actual.mean(), linestyle='--', color='#9a9a9a', label='Test prevalence')
    axes[0, 1].set(title='Precision-recall curves', xlabel='Recall', ylabel='Precision')
    axes[0, 1].legend(fontsize=8, frameon=False, loc='lower left')
    matrix = np.array([[9543, 4526], [1164, 8562]])
    ax = axes[1, 0]
    ax.imshow(matrix, cmap='Blues', vmin=0, vmax=11000)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f'{matrix[row,col]:,}', ha='center', va='center', fontsize=14,
                    color='white' if matrix[row,col] > 6000 else '#182535')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Not canceled', 'Canceled'],
           yticklabels=['Not canceled', 'Canceled'], xlabel='Predicted', ylabel='Actual',
           title='Selected model: confusion matrix')
    labels = ['deposit: Non Refund', 'agent: 89', 'agent: 152', 'agent: 17', 'agent: 11', 'parking spaces']
    ax = axes[1, 1]
    ax.barh(labels[::-1], coefficients.coefficient.to_numpy()[::-1], color='#187a87')
    ax.axvline(0, color='#49596a', lw=.7)
    ax.set(title='Six largest absolute LR coefficients', xlabel='Coefficient (not causal importance)')
    fig.tight_layout(h_pad=2.2, w_pad=2.5)
    path = ROOT / 'figures/11_report_test_diagnostics.png'
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
