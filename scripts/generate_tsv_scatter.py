#!/usr/bin/env python3
"""Generate TSV scatter plot by recomputing predictions from LOTO coefficients."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from scipy.stats import pearsonr
import os
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parent.parent))

# Set style with serif fonts (LaTeX-like without requiring LaTeX installation)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'cm',  # Computer Modern for math
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

# Paths
METRICS_PATH = Path(PROJECT_ROOT / 'results/mergeability/ViT-B-16/pairwise_metrics_N20.json')
PERFORMANCE_PATH = Path(PROJECT_ROOT / 'results/ViT-B-16/tsv/all_pairwise_summary_N20.json')
LOTO_RESULTS_PATH = Path(PROJECT_ROOT / 'results/metric_linear_optimization/loto_cv/tsv_loto_results.json')
FIGS_DIR = Path(PROJECT_ROOT / 'results/figs')

TARGET_METRIC = 'acc/test/avg'


def load_data():
    """Load metrics and performance data."""
    with open(METRICS_PATH, 'r') as f:
        metrics_data = json.load(f)

    with open(PERFORMANCE_PATH, 'r') as f:
        performance_data = json.load(f)

    with open(LOTO_RESULTS_PATH, 'r') as f:
        loto_results = json.load(f)

    return metrics_data, performance_data, loto_results


def get_performance(perf_data, pair_name):
    """Extract performance value from nested structure."""
    if pair_name not in perf_data:
        return None
    pair_data = perf_data[pair_name]
    if 'avg' in pair_data and len(pair_data['avg']) > 0:
        return pair_data['avg'][0].get(TARGET_METRIC)
    return None


def main():
    print("Loading data...")
    metrics_data, performance_data, loto_results = load_data()

    # Get tasks list
    tasks = metrics_data['datasets']
    n_tasks = len(tasks)
    print(f"Tasks: {tasks}")

    # Get metric names
    metric_names = sorted(metrics_data['metrics'].keys())
    print(f"Metrics ({len(metric_names)}): {metric_names}")

    # Build pair-to-index mapping
    pair_to_indices = {}
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            pair_name = f"{tasks[i]}__{tasks[j]}"
            pair_to_indices[pair_name] = (i, j)

    # Get all valid pairs (those with performance data)
    valid_pairs = []
    for pair_name in performance_data.keys():
        perf = get_performance(performance_data, pair_name)
        if perf is not None and pair_name in pair_to_indices:
            valid_pairs.append(pair_name)

    valid_pairs = sorted(valid_pairs)
    print(f"Found {len(valid_pairs)} valid pairs")

    # Build metrics matrix and performance array
    metrics_matrix = np.zeros((len(valid_pairs), len(metric_names)))
    performance_array = np.zeros(len(valid_pairs))

    for idx, pair_name in enumerate(valid_pairs):
        i, j = pair_to_indices[pair_name]
        for m_idx, metric in enumerate(metric_names):
            value = metrics_data['metrics'][metric]['matrix'][i][j]
            if value is None:
                value = 0.0  # Handle None values
            metrics_matrix[idx, m_idx] = value
        performance_array[idx] = get_performance(performance_data, pair_name)

    # Normalize metrics to [-1, 1]
    metrics_min = metrics_matrix.min(axis=0)
    metrics_max = metrics_matrix.max(axis=0)
    metrics_range = metrics_max - metrics_min
    metrics_range[metrics_range == 0] = 1  # Avoid division by zero
    metrics_normalized = 2 * (metrics_matrix - metrics_min) / metrics_range - 1

    # For each pair, find which fold it belongs to (validation set)
    # and use that fold's coefficients to make prediction
    fold_results = loto_results['fold_results']

    all_predictions = []
    all_actuals = []
    all_pair_names = []

    for fold in fold_results:
        held_out_task = fold['held_out_task']
        coefficients = fold['coefficients']

        # Get coefficient vector in same order as metric_names
        coef_vector = np.array([coefficients.get(m, 0) for m in metric_names])

        # Find validation pairs (pairs involving the held-out task)
        for idx, pair_name in enumerate(valid_pairs):
            i, j = pair_to_indices[pair_name]
            task1, task2 = tasks[i], tasks[j]

            if task1 == held_out_task or task2 == held_out_task:
                # This pair is in the validation set for this fold
                prediction = np.dot(metrics_normalized[idx], coef_vector)
                actual = performance_array[idx]
                all_predictions.append(prediction)
                all_actuals.append(actual)
                all_pair_names.append(pair_name)

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    print(f"Computed {len(all_predictions)} validation predictions")
    print(f"Raw predictions range: [{all_predictions.min():.2f}, {all_predictions.max():.2f}]")
    print(f"Actuals range: [{all_actuals.min():.2f}, {all_actuals.max():.2f}]")

    # Rescale predictions to [0, 1] range to match accuracy scale
    pred_min, pred_max = all_predictions.min(), all_predictions.max()
    all_predictions_scaled = (all_predictions - pred_min) / (pred_max - pred_min)

    # Compute correlation (same whether scaled or not)
    corr, p_value = pearsonr(all_predictions_scaled, all_actuals)
    print(f"Validation correlation: r={corr:.4f}, p={p_value:.2e}")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(all_predictions_scaled, all_actuals, alpha=0.6, s=50,
               edgecolors='k', linewidths=0.5, c='steelblue')

    # Best fit line (using scaled predictions)
    z = np.polyfit(all_predictions_scaled, all_actuals, 1)
    p = np.poly1d(z)
    pred_sorted = np.sort(all_predictions_scaled)
    ax.plot(pred_sorted, p(pred_sorted), 'r-', linewidth=2,
            label='Best fit ($r$={:.3f})'.format(corr))

    ax.set_xlabel('Predicted Mergeability Score', fontsize=12)
    ax.set_ylabel('Actual Post-Merge Accuracy', fontsize=12)
    ax.set_title('TSV: LOTO Validation Predictions\n($r$={:.3f}, $p$={:.2e}, $n$={})'.format(corr, p_value, len(all_predictions)),
                 fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'tsv_scatter.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'tsv_scatter.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved TSV scatter plot to {FIGS_DIR / 'tsv_scatter.pdf'}")


if __name__ == '__main__':
    main()
