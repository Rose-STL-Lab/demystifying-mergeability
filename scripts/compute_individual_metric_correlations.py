#!/usr/bin/env python3
"""Compute Pearson correlation between individual metrics and post-merge performance.

This script computes the correlation of each metric with post-merge performance
for each merging method, without any linear combination optimization.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import os
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parent.parent))

# Paths
RESULTS_DIR = Path(PROJECT_ROOT / 'results/ViT-B-16')
METRICS_PATH = Path(PROJECT_ROOT / 'results/mergeability/ViT-B-16/pairwise_metrics_N20.json')

# Methods to analyze (main ones used in the paper)
METHODS = ['arithmetic', 'weight_avg', 'isotropic', 'tsv']

METHOD_NAMES = {
    'arithmetic': 'Task Arithmetic',
    'weight_avg': 'Weight Averaging',
    'isotropic': 'Isotropic',
    'tsv': 'TSV'
}

# Metric display names
METRIC_SHORT_NAMES = {
    'task_vector_cosine_similarity': 'TV Cosine Sim',
    'task_vector_l2_distance': 'TV L2 Dist',
    'task_vector_dot_product': 'TV Dot Prod',
    'weight_space_angle': 'Weight Angle',
    'task_vector_magnitude_ratio': 'TV Mag Ratio',
    'effective_rank': 'Eff Rank',
    'effective_rank_mergeability_score': 'Eff Rank Score',
    'stable_rank': 'Stable Rank',
    'spectral_gap': 'Spectral Gap',
    'singular_value_ratio': 'SV Ratio',
    'layerwise_effective_rank': 'Layer Eff Rank',
    'layerwise_effective_rank_mergeability_score': 'Layer Eff Rank Score',
    'singular_value_overlap': 'SV Overlap',
    'subspace_overlap': 'Left Sub Top-k',
    'right_subspace_overlap_top_k': 'Right Sub Top-k',
    'right_subspace_overlap_bottom_k': 'Right Sub Bot-k',
    'interaction_matrix_overlap_top_k': 'Interact Top-k',
    'interaction_matrix_overlap_bottom_k': 'Interact Bot-k',
    'activation_l2_distance': 'Act L2 Dist',
    'activation_cosine_similarity': 'Act Cosine Sim',
    'activation_magnitude_ratio': 'Act Mag Ratio',
    'activation_dot_product': 'Act Dot Prod',
    'encoder_gradient_cosine_similarity': 'Enc Grad Cos',
    'encoder_gradient_l2_distance': 'Enc Grad L2',
    'encoder_gradient_dot_product': 'Enc Grad Dot',
    'input_gradient_cosine_similarity': 'Input Grad Cos',
    'input_gradient_l2_distance': 'Input Grad L2',
    'input_gradient_dot_product': 'Input Grad Dot',
}

# Metric categories for organization
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


def load_metrics():
    """Load pairwise metrics."""
    with open(METRICS_PATH, 'r') as f:
        data = json.load(f)
    return data


def load_performance(method):
    """Load post-merge performance for a method."""
    perf_path = RESULTS_DIR / method / 'all_pairwise_summary_N20.json'
    with open(perf_path, 'r') as f:
        data = json.load(f)
    return data


def extract_pairs(metrics_data, performance_data):
    """Extract aligned (metric_values, performance) pairs for all dataset pairs.

    Returns a dict mapping metric_name -> list of (metric_value, normalized_acc_avg) tuples
    """
    datasets = metrics_data['datasets']
    n = len(datasets)

    # Build mapping from pair name to performance
    perf_map = {}
    for pair_key, pair_data in performance_data.items():
        if 'avg' in pair_data:
            avg_acc = pair_data['avg'][0].get('normalized_acc/test/avg',
                      pair_data['avg'][0].get('acc/test/avg', 0))
            perf_map[pair_key] = avg_acc

    # For each metric, extract aligned values
    result = {}
    for metric_name, metric_info in metrics_data['metrics'].items():
        matrix = metric_info['matrix']
        pairs = []

        for i in range(n):
            for j in range(i+1, n):
                # Get metric value (upper triangle)
                metric_val = matrix[i][j]
                if metric_val is None:
                    continue

                # Get corresponding performance
                pair_key = f"{datasets[i]}__{datasets[j]}"
                if pair_key in perf_map:
                    perf_val = perf_map[pair_key]
                    pairs.append((metric_val, perf_val))

        result[metric_name] = pairs

    return result


def compute_correlations(pairs_dict):
    """Compute Pearson correlation for each metric.

    Returns dict mapping metric_name -> (correlation, p_value, n_samples)
    """
    correlations = {}
    for metric_name, pairs in pairs_dict.items():
        if len(pairs) < 3:
            continue

        metric_vals = np.array([p[0] for p in pairs])
        perf_vals = np.array([p[1] for p in pairs])

        # Remove any NaN values
        valid_mask = ~(np.isnan(metric_vals) | np.isnan(perf_vals))
        metric_vals = metric_vals[valid_mask]
        perf_vals = perf_vals[valid_mask]

        if len(metric_vals) < 3:
            continue

        r, p = stats.pearsonr(metric_vals, perf_vals)
        correlations[metric_name] = (r, p, len(metric_vals))

    return correlations


def main():
    print("Loading metrics...")
    metrics_data = load_metrics()

    # Store correlations for all methods
    all_correlations = {}

    for method in METHODS:
        print(f"\nProcessing {METHOD_NAMES[method]}...")
        performance_data = load_performance(method)
        pairs_dict = extract_pairs(metrics_data, performance_data)
        correlations = compute_correlations(pairs_dict)
        all_correlations[method] = correlations

    # Print results
    print("\n" + "="*100)
    print("INDIVIDUAL METRIC CORRELATIONS WITH POST-MERGE PERFORMANCE")
    print("="*100)

    # Get all metrics in category order
    all_metrics = []
    for cat_metrics in METRIC_CATEGORIES.values():
        all_metrics.extend(cat_metrics)

    # Print table header
    print(f"\n{'Metric':<30} | " + " | ".join(f"{METHOD_NAMES[m]:^18}" for m in METHODS))
    print("-"*30 + "-+-" + "-+-".join("-"*18 for _ in METHODS))

    for category, cat_metrics in METRIC_CATEGORIES.items():
        print(f"\n--- {category} ---")
        for metric in cat_metrics:
            short_name = METRIC_SHORT_NAMES.get(metric, metric)
            row = f"{short_name:<30} |"
            for method in METHODS:
                if metric in all_correlations[method]:
                    r, p, n = all_correlations[method][metric]
                    sig = "*" if p < 0.05 else ""
                    sig = "**" if p < 0.01 else sig
                    sig = "***" if p < 0.001 else sig
                    row += f" {r:>6.3f}{sig:<3} (n={n:>3}) |"
                else:
                    row += f" {'N/A':^18} |"
            print(row)

    # Generate LaTeX table
    print("\n\n" + "="*100)
    print("LATEX TABLE")
    print("="*100)

    latex = generate_latex_table(all_correlations, METRIC_CATEGORIES, METRIC_SHORT_NAMES, METHODS, METHOD_NAMES)
    print(latex)

    # Save results
    output_path = Path(PROJECT_ROOT / 'results/individual_metric_correlations.json')
    with open(output_path, 'w') as f:
        # Convert to serializable format
        serializable = {}
        for method, corrs in all_correlations.items():
            serializable[method] = {
                metric: {'r': r, 'p': p, 'n': n}
                for metric, (r, p, n) in corrs.items()
            }
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save LaTeX table
    latex_path = Path(PROJECT_ROOT / 'results/individual_metric_correlations.tex')
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved to {latex_path}")


def generate_latex_table(all_correlations, metric_categories, metric_short_names, methods, method_names):
    """Generate LaTeX table for the appendix."""

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pearson correlation between individual mergeability metrics and normalized post-merge accuracy for each merging method. Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. The generally low correlations demonstrate that no single metric is sufficient for predicting mergeability, motivating our learned linear combination approach.}")
    lines.append(r"\label{tab:individual_metric_correlations}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    # Add extra column for rotated category labels
    lines.append(r"\begin{tabular}{cl" + "r"*len(methods) + "}")
    lines.append(r"\toprule")

    # Header (empty cell for category column)
    header = r" & \textbf{Metric} & " + " & ".join(rf"\textbf{{{method_names[m]}}}" for m in methods) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    category_list = list(metric_categories.items())
    for cat_idx, (category, cat_metrics) in enumerate(category_list):
        n_metrics = len(cat_metrics)

        for i, metric in enumerate(cat_metrics):
            short_name = metric_short_names.get(metric, metric).replace('_', r'\_')
            # Escape special LaTeX characters
            short_name = short_name.replace('$', r'\$').replace('-k', r'-$k$')

            row_vals = []
            for method in methods:
                if metric in all_correlations[method]:
                    r, p, n = all_correlations[method][metric]
                    sig = ""
                    if p < 0.001:
                        sig = r"$^{***}$"
                    elif p < 0.01:
                        sig = r"$^{**}$"
                    elif p < 0.05:
                        sig = r"$^{*}$"
                    row_vals.append(f"{r:.3f}{sig}")
                else:
                    row_vals.append("---")

            # Add rotated category label only for the middle row of each category
            if i == n_metrics // 2:
                cat_label = rf"\rotatebox{{90}}{{\textbf{{{category}}}}}"
                row = f"\\multirow{{{n_metrics}}}{{*}}{{{cat_label}}} & {short_name} & " + " & ".join(row_vals) + r" \\"
            else:
                row = f" & {short_name} & " + " & ".join(row_vals) + r" \\"
            lines.append(row)

        # Add horizontal line between categories (except after the last one)
        if cat_idx < len(category_list) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == '__main__':
    main()