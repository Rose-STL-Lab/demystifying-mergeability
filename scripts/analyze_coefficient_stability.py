#!/usr/bin/env python3
"""
Analyze coefficient stability across LOTO folds.

Identifies:
1. Most stable coefficients (low variance across folds)
2. Most consistent direction (always positive or always negative)
3. Highest average impact (high mean absolute coefficient)
4. Method-specific stable features
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parent.parent))

sns.set_style("whitegrid")


def load_loto_results(results_dir):
    """Load LOTO results for all methods."""
    results_dir = Path(results_dir)
    methods = ['weight_avg', 'arithmetic', 'tsv', 'isotropic']
    results = {}

    for method in methods:
        result_file = results_dir / f'{method}_loto_results.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[method] = json.load(f)
        else:
            print(f"Warning: {result_file} not found")

    return results


def extract_fold_coefficients(results):
    """Extract coefficient matrix for each method (folds × metrics)."""
    fold_coefficients = {}

    for method, result in results.items():
        fold_results = result['fold_results']

        # Get metric names from first fold
        metric_names = list(fold_results[0]['coefficients'].keys())

        # Create matrix: rows=folds, cols=metrics
        n_folds = len(fold_results)
        n_metrics = len(metric_names)
        coef_matrix = np.zeros((n_folds, n_metrics))

        for i, fold in enumerate(fold_results):
            for j, metric in enumerate(metric_names):
                coef_matrix[i, j] = fold['coefficients'][metric]

        fold_coefficients[method] = {
            'matrix': coef_matrix,
            'metric_names': metric_names,
            'n_folds': n_folds
        }

    return fold_coefficients


def compute_stability_metrics(fold_coefficients):
    """
    Compute stability metrics for coefficients.

    Returns:
        DataFrame with columns: metric, mean_coef, std_coef, cv, consistency_score, sign_agreement
    """
    stability_results = {}

    for method, data in fold_coefficients.items():
        matrix = data['matrix']
        metric_names = data['metric_names']

        stats = []
        for j, metric in enumerate(metric_names):
            coefs = matrix[:, j]

            mean_coef = np.mean(coefs)
            std_coef = np.std(coefs)
            abs_mean = np.abs(mean_coef)

            # Coefficient of variation (normalized std)
            cv = std_coef / (abs_mean + 1e-8)

            # Sign consistency: fraction of folds with same sign as mean
            sign_agreement = np.mean(np.sign(coefs) == np.sign(mean_coef))

            # Consistency score: low CV + high sign agreement
            consistency_score = sign_agreement / (cv + 0.1)  # Higher is more stable

            stats.append({
                'metric': metric,
                'mean_coef': mean_coef,
                'abs_mean_coef': abs_mean,
                'std_coef': std_coef,
                'cv': cv,
                'sign_agreement': sign_agreement,
                'consistency_score': consistency_score
            })

        df = pd.DataFrame(stats)
        stability_results[method] = df

    return stability_results


def identify_stable_features(stability_results, top_k=10):
    """Identify most stable features per method."""
    stable_features = {}

    for method, df in stability_results.items():
        # Sort by consistency score (high = stable)
        df_sorted = df.sort_values('consistency_score', ascending=False)

        # Filter: only keep features with high sign agreement (>0.7) and moderate impact
        df_filtered = df_sorted[
            (df_sorted['sign_agreement'] >= 0.7) &
            (df_sorted['abs_mean_coef'] >= 0.5)  # Has some impact
        ]

        stable_features[method] = df_filtered.head(top_k)

    return stable_features


def plot_coefficient_stability(fold_coefficients, stability_results, output_dir):
    """Create visualizations of coefficient stability."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, data in fold_coefficients.items():
        matrix = data['matrix']
        metric_names = data['metric_names']
        df_stats = stability_results[method]

        # Get top 15 most stable features
        top_stable = df_stats.nsmallest(15, 'cv')
        top_indices = [metric_names.index(m) for m in top_stable['metric'].values]

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Coefficient distributions (violin plot)
        ax = axes[0]
        top_matrix = matrix[:, top_indices]
        positions = np.arange(len(top_indices))

        parts = ax.violinplot(top_matrix, positions=positions, widths=0.7,
                             showmeans=True, showmedians=True)

        # Color violins by mean coefficient sign
        means = top_matrix.mean(axis=0)
        for i, pc in enumerate(parts['bodies']):
            color = 'green' if means[i] > 0 else 'red'
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(positions)
        ax.set_xticklabels([metric_names[i][:30] + '...' if len(metric_names[i]) > 30
                           else metric_names[i] for i in top_indices],
                          rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Coefficient Value', fontsize=11)
        ax.set_title(f'{method.upper()}: Top 15 Most Stable Features\n(Low Coefficient Variation)',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: Stability metrics scatter
        ax = axes[1]
        scatter = ax.scatter(df_stats['cv'], df_stats['sign_agreement'],
                           s=df_stats['abs_mean_coef']*100,
                           c=df_stats['abs_mean_coef'],
                           cmap='viridis', alpha=0.6)

        # Annotate top stable features
        for _, row in top_stable.head(5).iterrows():
            ax.annotate(row['metric'][:20],
                       (row['cv'], row['sign_agreement']),
                       fontsize=8, alpha=0.7)

        ax.set_xlabel('Coefficient of Variation (lower = more stable)', fontsize=11)
        ax.set_ylabel('Sign Agreement (higher = consistent direction)', fontsize=11)
        ax.set_title(f'{method.upper()}: Feature Stability\n(size = |mean coefficient|)',
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('|Mean Coefficient|', fontsize=10)

        # Add quadrants
        cv_median = df_stats['cv'].median()
        sign_median = df_stats['sign_agreement'].median()
        ax.axvline(cv_median, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(sign_median, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f'{method}_stability.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved stability plot: {output_dir / f'{method}_stability.png'}")


def create_comparison_heatmap(stability_results, output_dir):
    """Create heatmap comparing stability across methods."""
    output_dir = Path(output_dir)

    # Get all metric names
    all_metrics = stability_results[list(stability_results.keys())[0]]['metric'].values

    # Create matrix of consistency scores
    methods = list(stability_results.keys())
    consistency_matrix = np.zeros((len(all_metrics), len(methods)))

    for j, method in enumerate(methods):
        df = stability_results[method]
        for i, metric in enumerate(all_metrics):
            row = df[df['metric'] == metric]
            if not row.empty:
                consistency_matrix[i, j] = row['consistency_score'].values[0]

    # Sort by average consistency across methods
    avg_consistency = consistency_matrix.mean(axis=1)
    sorted_indices = np.argsort(avg_consistency)[::-1][:20]  # Top 20

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(consistency_matrix[sorted_indices],
                xticklabels=[m.upper() for m in methods],
                yticklabels=[all_metrics[i][:45] for i in sorted_indices],
                cmap='YlOrRd', annot=True, fmt='.1f',
                cbar_kws={'label': 'Consistency Score'},
                ax=ax)

    ax.set_title('Feature Stability Across Methods\n(Top 20 Most Consistent Features)',
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel('Merge Method', fontsize=11)
    ax.set_ylabel('Mergeability Metric', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'stability_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison heatmap: {output_dir / 'stability_comparison_heatmap.png'}")


def main():
    # Paths
    results_dir = Path(PROJECT_ROOT / 'results/metric_linear_optimization/loto_cv')
    output_dir = results_dir / 'stability_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Coefficient Stability Analysis (LOTO Cross-Validation)")
    print("="*70)
    print()

    # Load results
    print("Loading LOTO results...")
    results = load_loto_results(results_dir)
    print(f"Loaded results for: {list(results.keys())}")
    print()

    # Extract fold coefficients
    print("Extracting fold coefficients...")
    fold_coefficients = extract_fold_coefficients(results)
    print()

    # Compute stability metrics
    print("Computing stability metrics...")
    stability_results = compute_stability_metrics(fold_coefficients)
    print()

    # Identify stable features
    print("Identifying most stable features...")
    stable_features = identify_stable_features(stability_results, top_k=10)
    print()

    # Print results
    print("="*70)
    print("MOST STABLE FEATURES PER METHOD")
    print("="*70)
    print("\nCriteria: High sign agreement (>70%) + Low coefficient variation + Moderate impact")
    print()

    for method, df in stable_features.items():
        print(f"\n{method.upper()}:")
        print("-"*80)
        if len(df) > 0:
            print(f"{'Rank':<5} {'Metric':<45} {'Mean':<10} {'Std':<10} {'Sign%':<8}")
            print("-"*80)
            for i, (_, row) in enumerate(df.iterrows(), 1):
                print(f"{i:<5} {row['metric'][:43]:<45} {row['mean_coef']:>9.3f} "
                     f"{row['std_coef']:>9.3f} {row['sign_agreement']*100:>6.0f}%")
        else:
            print("  No features meet stability criteria")

    # Print overall statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    for method, df in stability_results.items():
        print(f"\n{method.upper()}:")
        print(f"  Features with >80% sign agreement: {len(df[df['sign_agreement'] > 0.8])}/29")
        print(f"  Features with >70% sign agreement: {len(df[df['sign_agreement'] > 0.7])}/29")
        print(f"  Average coefficient std: {df['std_coef'].mean():.2f}")
        print(f"  Median sign agreement: {df['sign_agreement'].median()*100:.0f}%")

    # Save results to CSV
    print("\n" + "="*70)
    print("Saving results...")

    for method, df in stability_results.items():
        csv_path = output_dir / f'{method}_stability_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved {method} stability metrics to: {csv_path}")

    # Create visualizations
    print("\n" + "="*70)
    print("Creating visualizations...")
    plot_coefficient_stability(fold_coefficients, stability_results, output_dir)
    create_comparison_heatmap(stability_results, output_dir)

    # Identify method-specific stable features
    print("\n" + "="*70)
    print("METHOD-SPECIFIC STABLE FEATURES")
    print("="*70)
    print("\n(Features stable for one method but not others)")

    for method, df_method in stable_features.items():
        if len(df_method) == 0:
            continue

        method_specific = []
        for _, row in df_method.iterrows():
            metric = row['metric']

            # Check if unstable in other methods
            is_specific = True
            for other_method, df_other in stability_results.items():
                if other_method == method:
                    continue

                other_row = df_other[df_other['metric'] == metric]
                if not other_row.empty:
                    # If sign agreement is also high in other method, not specific
                    if other_row['sign_agreement'].values[0] > 0.7:
                        is_specific = False
                        break

            if is_specific:
                method_specific.append(metric)

        if method_specific:
            print(f"\n{method.upper()}-SPECIFIC:")
            for metric in method_specific[:5]:
                row = df_method[df_method['metric'] == metric].iloc[0]
                print(f"  • {metric:<45} (mean={row['mean_coef']:+.3f}, sign={row['sign_agreement']*100:.0f}%)")

    print("\n" + "="*70)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
