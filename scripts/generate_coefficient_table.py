#!/usr/bin/env python3
"""Generate LaTeX table of learned coefficients for all methods."""

import json
from pathlib import Path
import os
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(PROJECT_ROOT / 'results/metric_linear_optimization/loto_cv')
OUTPUT_DIR = Path(PROJECT_ROOT / 'results')

METHODS = ['arithmetic', 'weight_avg', 'isotropic', 'tsv']
METHOD_NAMES = {
    'arithmetic': 'Task Arithmetic',
    'weight_avg': 'Weight Averaging',
    'isotropic': 'Isotropic',
    'tsv': 'TSV'
}

# Metric categories (same order as heatmap)
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
    'subspace_overlap': 'Left Sub Top-$k$',
    'right_subspace_overlap_top_k': 'Right Sub Top-$k$',
    'right_subspace_overlap_bottom_k': 'Right Sub Bot-$k$',
    'interaction_matrix_overlap_top_k': 'Interact Top-$k$',
    'interaction_matrix_overlap_bottom_k': 'Interact Bot-$k$',
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


def load_coefficients():
    """Load average coefficients for all methods."""
    all_coefs = {}
    for method in METHODS:
        filepath = RESULTS_DIR / f'{method}_loto_results.json'
        with open(filepath, 'r') as f:
            data = json.load(f)
        all_coefs[method] = data['average_coefficients']
    return all_coefs


def generate_latex_table(all_coefs):
    """Generate LaTeX table with rotated category labels."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Average learned coefficients for each mergeability metric across merging methods, obtained via leave-one-task-out cross-validation. Positive coefficients indicate that higher metric values predict better post-merge performance, while negative coefficients indicate the opposite. Coefficients are on normalized metrics (min-max scaled to $[0, 1]$).}")
    lines.append(r"\label{tab:learned_coefficients}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{clrrrr}")
    lines.append(r"\toprule")

    # Header
    header = r" & \textbf{Metric} & " + " & ".join(rf"\textbf{{{METHOD_NAMES[m]}}}" for m in METHODS) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    category_list = list(METRIC_CATEGORIES.items())
    for cat_idx, (category, cat_metrics) in enumerate(category_list):
        n_metrics = len(cat_metrics)

        for i, metric in enumerate(cat_metrics):
            short_name = METRIC_SHORT_NAMES.get(metric, metric)

            # Get coefficient values for each method
            row_vals = []
            for method in METHODS:
                if metric in all_coefs[method]:
                    val = all_coefs[method][metric]
                    row_vals.append(f"{val:.2f}")
                else:
                    row_vals.append("---")

            # Add rotated category label only for the middle row
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


def generate_appendix_section(all_coefs):
    """Generate the full appendix section with text and table."""

    # Compute some statistics for the text
    max_coefs = {}
    min_coefs = {}
    for method in METHODS:
        coefs = all_coefs[method]
        max_metric = max(coefs.keys(), key=lambda k: coefs[k])
        min_metric = min(coefs.keys(), key=lambda k: coefs[k])
        max_coefs[method] = (max_metric, coefs[max_metric])
        min_coefs[method] = (min_metric, coefs[min_metric])

    text = r"""\section{Learned Coefficient Values}
\label{app:learned_coefficients}

Table~\ref{tab:learned_coefficients} reports the average learned coefficients for each mergeability metric across all merging methods. These coefficients are obtained by averaging across all 20 folds of the leave-one-task-out cross-validation procedure. The coefficients operate on min-max normalized metrics (scaled to $[0, 1]$), so their magnitudes reflect both the importance of each metric and its original scale.

\paragraph{Interpretation.}
A positive coefficient indicates that higher values of the corresponding metric predict better post-merge performance, while a negative coefficient indicates the opposite relationship. The magnitude of a coefficient reflects how strongly that metric influences the prediction, though direct comparison across metrics requires accounting for their different variances.

\paragraph{Key Observations.}
Several patterns emerge from the coefficient values:

\begin{itemize}
    \item \textbf{Gradient-based metrics} receive consistently large-magnitude coefficients across methods, with input gradient L2 distance showing strong negative coefficients for all methods (ranging from $-25.0$ to $-42.3$). This suggests that large gradient differences between task vectors are detrimental to merging success.

    \item \textbf{Distance metrics} (L2 distances) generally receive negative coefficients, indicating that greater dissimilarity between models predicts worse merging outcomes. Conversely, similarity metrics (cosine similarity, overlap measures) tend to receive positive coefficients.

    \item \textbf{Method-specific patterns}: Weight Averaging shows distinctively large coefficients for right subspace overlap ($41.5$) and input gradient cosine similarity ($45.0$), suggesting these metrics are particularly informative for this method. Task Arithmetic and Isotropic show similar coefficient patterns, while TSV exhibits generally smaller coefficient magnitudes.

    \item \textbf{Sign consistency}: Some metrics show consistent signs across all methods (e.g., input gradient L2 distance is always negative), while others flip signs depending on the merging method, highlighting the importance of method-specific prediction models.
\end{itemize}

"""

    table = generate_latex_table(all_coefs)

    return text + table


def main():
    print("Loading coefficients...")
    all_coefs = load_coefficients()

    print("Generating appendix section...")
    appendix = generate_appendix_section(all_coefs)

    # Save to file
    output_path = OUTPUT_DIR / 'appendix_learned_coefficients.tex'
    with open(output_path, 'w') as f:
        f.write(appendix)
    print(f"Saved to {output_path}")

    # Also print the table
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print(appendix)


if __name__ == '__main__':
    main()