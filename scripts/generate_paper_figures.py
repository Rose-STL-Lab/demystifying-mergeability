#!/usr/bin/env python3
"""Generate figures for the mergeability prediction paper."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
import os
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parent.parent))

# Set style with serif fonts (LaTeX-like without requiring LaTeX installation)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'cm',  # Computer Modern for math
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 25,
    'figure.dpi': 600,
})

# Paths
RESULTS_DIR = Path(PROJECT_ROOT / 'results/metric_linear_optimization/loto_cv_no_leakage')
FIGS_DIR = Path(PROJECT_ROOT / 'results/figs')
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Method names for display
METHOD_NAMES = {
    'arithmetic': 'Task Arithmetic',
    'weight_avg': 'Weight Averaging',
    'isotropic': 'Isotropic',
    'tsv': 'TSV'
}


METRIC_COUNTS_PER_CATEGORY = {
    'Task Vector Geometry': 5,
    'Effective Rank': 7,
    'Subspace Overlap': 6,
    'Activation-Based': 4,
    'Gradient-Based': 6
}

# Metric categories
METRIC_CATEGORIES = {
    'Task Vector Geometry': [
        'task_vector_cosine_similarity',
        'task_vector_l2_distance',
        'task_vector_dot_product',
        'weight_space_angle',
        'task_vector_magnitude_ratio',
    ],
    'Effective Rank': [
        'effective_rank',
        'effective_rank_mergeability_score',
        'stable_rank',
        'spectral_gap',
        'singular_value_ratio',
        'layerwise_effective_rank',
        'layerwise_effective_rank_mergeability_score',
    ],
    'Subspace Overlap': [
        'singular_value_overlap',
        'subspace_overlap',
        'right_subspace_overlap_top_k',
        'right_subspace_overlap_bottom_k',
        'interaction_matrix_overlap_top_k',
        'interaction_matrix_overlap_bottom_k',
    ],
    'Activation-Based': [
        'activation_l2_distance',
        'activation_cosine_similarity',
        'activation_magnitude_ratio',
        'activation_dot_product',
    ],
    'Gradient-Based': [
        'encoder_gradient_cosine_similarity',
        'encoder_gradient_l2_distance',
        'encoder_gradient_dot_product',
        'input_gradient_cosine_similarity',
        'input_gradient_l2_distance',
        'input_gradient_dot_product',
    ],
}

# Short metric names for display (LaTeX-safe)
METRIC_SHORT_NAMES = {
    'task_vector_cosine_similarity': r'TV Cosine Sim.',
    'task_vector_l2_distance': r'TV L2 Dist.',
    'task_vector_dot_product': r'TV Dot',
    'weight_space_angle': r'TV Angle',
    'task_vector_magnitude_ratio': r'TV Magn. Ratio',
    'effective_rank': r'Effective Rank',
    'effective_rank_mergeability_score': r'Eff Rank Score',
    'stable_rank': r'Stable Rank',
    'spectral_gap': r'Spectral Gap',
    'singular_value_ratio': r'Singular Value Ratio',
    'layerwise_effective_rank': r'Layer Eff. Rank',
    'layerwise_effective_rank_mergeability_score': r'Layer Eff. Rank Score',
    'singular_value_overlap': r'SV Overlap',
    'subspace_overlap': r'Left Sub. Top-$k$',
    'right_subspace_overlap_top_k': r'Right Sub. Top-$k$',
    'right_subspace_overlap_bottom_k': r'Right Sub. Bottom-$k$',
    'interaction_matrix_overlap_top_k': r'Interaction Top-$k$',
    'interaction_matrix_overlap_bottom_k': r'Interaction Bottom-$k$',
    'activation_l2_distance': r'Activation L2 Dist.',
    'activation_cosine_similarity': r'Activation Cos. Sim.',
    'activation_magnitude_ratio': r'Activation Magn. Ratio',
    'activation_dot_product': r'Activation Dot',
    'encoder_gradient_cosine_similarity': r'Enc. Gradd. Cos. Sim.',
    'encoder_gradient_l2_distance': r'Enc. Grad. L2 Dist.',
    'encoder_gradient_dot_product': r'Enc. Grad. Dot',
    'input_gradient_cosine_similarity': r'Input Grad. Cos. Sim.',
    'input_gradient_l2_distance': r'Input Grad. L2 Dist.',
    'input_gradient_dot_product': r'Input Grad. Dot',
}


def load_loto_results():
    """Load LOTO results for all methods."""
    results = {}
    for method in ['arithmetic', 'weight_avg', 'isotropic', 'tsv']:
        filepath = RESULTS_DIR / f'{method}_loto_results.json'
        with open(filepath, 'r') as f:
            results[method] = json.load(f)
    return results


def plot_coefficient_heatmap(results):
    """Generate coefficient heatmap across methods."""
    print("Generating coefficient heatmap...")

    methods = ['arithmetic', 'weight_avg', 'isotropic', 'tsv']

    # Get all metrics in category order
    all_metrics = []
    category_boundaries = []
    category_labels = []
    for cat_name, metrics in METRIC_CATEGORIES.items():
        category_boundaries.append(len(all_metrics))
        category_labels.append(cat_name)
        all_metrics.extend(metrics)
    category_boundaries.append(len(all_metrics))

    # Build coefficient matrix (transposed: rows=methods, cols=metrics)
    coef_matrix = np.zeros((len(methods), len(all_metrics)))
    for j, method in enumerate(methods):
        avg_coefs = results[method]['average_coefficients']
        for i, metric in enumerate(all_metrics):
            if metric in avg_coefs:
                coef_matrix[j, i] = avg_coefs[metric]

    # Create figure with extra space at bottom for colorbar
    fig, ax = plt.subplots(figsize=(16, 5))

    # Normalize for better visualization (clip extreme values)
    vmax = np.percentile(np.abs(coef_matrix), 95)

    # Plot heatmap
    im = ax.imshow(coef_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax)

    # Labels (swapped: x=metrics, y=methods)
    ax.set_xticks(range(len(all_metrics)))
    ax.set_xticklabels([METRIC_SHORT_NAMES.get(m, m) for m in all_metrics], rotation=45, ha='right', fontsize=18)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_NAMES[m] for m in methods], fontsize=18)

    # Add category separators (vertical lines instead of horizontal)
    for boundary in category_boundaries[1:-1]:
        ax.axvline(x=boundary - 0.5, color='black', linewidth=1.5)

    # Colorbar - position it at the top
    cbar_ax = fig.add_axes([0.25, 0.92, 0.5, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Average Coefficient', fontsize=18)
    cbar_ax.tick_params(labelsize=16)
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.xaxis.set_ticks_position('top')

    ax.set_ylabel('Merging Method', fontsize=18)
    #ax.set_title('Learned Coefficients Across Merging Methods', fontsize=20)

    # Adjust layout
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(FIGS_DIR / 'coefficient_heatmap.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'coefficient_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved coefficient heatmap to {FIGS_DIR / 'coefficient_heatmap.pdf'}")


def plot_tsv_scatter(results):
    """Generate scatter plot for TSV predictions vs actual."""
    print("Generating TSV scatter plot...")

    tsv_results = results['tsv']
    fold_results = tsv_results['fold_results']

    # Collect all validation predictions and actuals
    all_val_predictions = []
    all_val_actuals = []

    for fold in fold_results:
        if 'val_predictions' in fold and 'val_actuals' in fold:
            all_val_predictions.extend(fold['val_predictions'])
            all_val_actuals.extend(fold['val_actuals'])

    # If predictions not stored, we need to recompute or skip
    if not all_val_predictions:
        print("Warning: Validation predictions not found in results. Using per-fold correlations instead.")
        # Create a summary plot instead
        fig, ax = plt.subplots(figsize=(6, 5))

        val_correlations = [fold['val_r'] for fold in fold_results]
        held_out_tasks = [fold['held_out_task'] for fold in fold_results]

        colors = plt.cm.viridis(np.linspace(0, 1, len(val_correlations)))
        bars = ax.bar(range(len(val_correlations)), val_correlations, color=colors)

        ax.set_xticks(range(len(held_out_tasks)))
        ax.set_xticklabels(held_out_tasks, rotation=90, ha='center', fontsize=7)
        ax.set_ylabel('Validation Correlation ($r$)')
        ax.set_xlabel('Held-Out Task')
        ax.set_title('TSV: Per-Fold Validation Correlations')
        ax.axhline(y=np.mean(val_correlations), color='red', linestyle='--',
                   label='Mean: {:.3f}'.format(np.mean(val_correlations)))
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'tsv_validation_by_fold.pdf', bbox_inches='tight')
        plt.savefig(FIGS_DIR / 'tsv_validation_by_fold.png', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved TSV per-fold plot to {FIGS_DIR / 'tsv_validation_by_fold.pdf'}")
        return

    all_val_predictions = np.array(all_val_predictions)
    all_val_actuals = np.array(all_val_actuals)

    # Compute correlation
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(all_val_predictions, all_val_actuals)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(all_val_predictions, all_val_actuals, alpha=0.6, s=40,
               edgecolors='k', linewidths=0.5)

    # Best fit line
    z = np.polyfit(all_val_predictions, all_val_actuals, 1)
    p = np.poly1d(z)
    pred_sorted = np.sort(all_val_predictions)
    ax.plot(pred_sorted, p(pred_sorted), 'r-', linewidth=2,
            label=f'Best fit (r={corr:.3f})')

    ax.set_xlabel('Predicted Mergeability Score')
    ax.set_ylabel('Actual Post-Merge Accuracy')
    ax.set_title(f'TSV: Predicted vs Actual Mergeability\n(LOTO Validation, r={corr:.3f}, p={p_value:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'tsv_scatter.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'tsv_scatter.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved TSV scatter plot to {FIGS_DIR / 'tsv_scatter.pdf'}")


def plot_category_importance(results):
    """Generate metric category importance bar chart normalized by metric counts."""
    print("Generating metric category importance chart...")

    methods = ['arithmetic', 'weight_avg', 'isotropic', 'tsv']
    categories = list(METRIC_CATEGORIES.keys())
    
    # 1. Compute sum of |coefficients| per category per method
    importance_raw = np.zeros((len(methods), len(categories)))

    for j, method in enumerate(methods):
        avg_coefs = results[method]['average_coefficients']
        for i, (cat_name, metrics) in enumerate(METRIC_CATEGORIES.items()):
            cat_importance = sum(abs(avg_coefs.get(m, 0)) for m in metrics)
            importance_raw[j, i] = cat_importance

    # 2. Normalize by the number of metrics in each category
    # This converts "Cumulative Sum" to "Mean Magnitude"
    counts = np.array([METRIC_COUNTS_PER_CATEGORY[cat] for cat in categories])
    importance_per_metric = importance_raw / counts

    # 3. Normalize per method for the relative plot
    importance_normalized = importance_per_metric / importance_per_metric.sum(axis=1, keepdims=True)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    x_methods = np.arange(len(methods))
    width_bar = 0.6

    # Plot 1: Mean Absolute importance (stacked bar)
    bottom = np.zeros(len(methods))
    for i, (cat, color) in enumerate(zip(categories, colors)):
        ax1.bar(x_methods, importance_per_metric[:, i], width_bar, 
                bottom=bottom, label=cat, color=color)
        bottom += importance_per_metric[:, i]

    ax1.set_xticks(x_methods)
    ax1.set_xticklabels([METHOD_NAMES[m] for m in methods], rotation=45, ha='right', fontsize=18)
    ax1.set_ylabel('Mean $|$Coefficients$|$', fontsize=20)
    ax1.set_title('Mean Category Importance', fontsize=22)
    ax1.legend(loc='upper left', fontsize=16)
    ax1.tick_params(axis='y', labelsize=18)

    # Plot 2: Relative importance (grouped bar)
    x_cats = np.arange(len(categories))
    width_group = 0.2

    for i, method in enumerate(methods):
        offset = (i - 1.5) * width_group
        ax2.bar(x_cats + offset, importance_normalized[i], width_group,
                label=METHOD_NAMES[method], color=colors[i] if i < len(colors) else f'C{i}')

    ax2.set_xticks(x_cats)
    ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=18)
    ax2.set_ylabel('Relative Importance (Normalized)', fontsize=20)
    ax2.set_title('Relative Importance per Metric Category', fontsize=22)
    ax2.legend(loc='upper right', fontsize=16)
    ax2.set_ylim(0, 0.6) # Bumped slightly as normalization might concentrate weights
    ax2.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'category_importance.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'category_importance.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved normalized category importance chart to {FIGS_DIR / 'category_importance.pdf'}")


def main():
    print("Loading LOTO results...")
    results = load_loto_results()

    # Generate all figures
    plot_coefficient_heatmap(results)
    #plot_tsv_scatter(results)
    #plot_validation_boxplots(results)
    #plot_category_importance(results)

    print("\nAll figures generated successfully!")
    print(f"Figures saved to: {FIGS_DIR}")


if __name__ == '__main__':
    main()
