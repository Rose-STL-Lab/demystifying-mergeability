#!/usr/bin/env python3
"""
Analyze linear optimization coefficients to understand method-specific mergeability properties.

This script:
1. Loads linear optimization results for all merge methods
2. Compares coefficients across methods
3. Identifies method-specific important features
4. Provides interpretations for top features
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parent.parent))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_linear_results(results_dir):
    """Load linear optimization results for all methods."""
    results_dir = Path(results_dir)

    methods = ['weight_avg', 'arithmetic', 'tsv', 'isotropic']
    results = {}

    for method in methods:
        result_file = results_dir / f'{method}_pearson.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[method] = json.load(f)
        else:
            print(f"Warning: {result_file} not found")

    return results


def create_coefficient_comparison(results):
    """Create DataFrame comparing coefficients across methods."""
    # Get all metric names (should be same across methods)
    metric_names = list(results[list(results.keys())[0]]['coefficients'].keys())

    # Create DataFrame
    data = {}
    for method, result in results.items():
        data[method] = [result['coefficients'][metric] for metric in metric_names]

    df = pd.DataFrame(data, index=metric_names)

    return df


def identify_top_features(df, top_k=10):
    """Identify top-k most important features for each method."""
    top_features = {}

    for method in df.columns:
        # Sort by absolute coefficient value
        sorted_features = df[method].abs().sort_values(ascending=False)
        top_features[method] = sorted_features.head(top_k)

    return top_features


def compute_feature_importance_stats(df):
    """Compute statistics about feature importance across methods."""
    stats = {
        'mean_abs_coef': df.abs().mean(axis=1).sort_values(ascending=False),
        'std_abs_coef': df.abs().std(axis=1).sort_values(ascending=False),
        'max_abs_coef': df.abs().max(axis=1).sort_values(ascending=False),
        'consistency': (df.abs().std(axis=1) / (df.abs().mean(axis=1) + 1e-8)).sort_values()  # Low = consistent across methods
    }

    return stats


def plot_coefficient_heatmap(df, output_path):
    """Create heatmap of coefficients across methods."""
    # Sort by mean absolute coefficient
    mean_abs = df.abs().mean(axis=1)
    df_sorted = df.loc[mean_abs.sort_values(ascending=False).index]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 16))

    # Plot heatmap
    sns.heatmap(df_sorted, cmap='RdBu_r', center=0,
                annot=True, fmt='.3f',
                cbar_kws={'label': 'Coefficient Value'},
                ax=ax, vmin=-0.6, vmax=0.6)

    ax.set_title('Linear Coefficients Across Merge Methods', fontsize=16, pad=20)
    ax.set_xlabel('Merge Method', fontsize=12)
    ax.set_ylabel('Mergeability Metric', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to: {output_path}")


def plot_top_features_comparison(top_features, output_path):
    """Plot top features for each method side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    methods = list(top_features.keys())

    for idx, method in enumerate(methods):
        ax = axes[idx]

        features = top_features[method]

        # Create bar plot
        colors = ['green' if x > 0 else 'red' for x in features.values]
        bars = ax.barh(range(len(features)), features.values, color=colors, alpha=0.7)

        # Customize
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([name[:40] + '...' if len(name) > 40 else name
                           for name in features.index], fontsize=9)
        ax.set_xlabel('Coefficient Value', fontsize=11)
        ax.set_title(f'Top 10 Features for {method.upper()}', fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, features.values)):
            ax.text(val, i, f' {val:.3f}', va='center',
                   ha='left' if val > 0 else 'right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved top features comparison to: {output_path}")


def analyze_method_specificity(df, threshold=0.1):
    """Identify features that are specific to certain methods."""
    # Find features with high coefficient for one method, low for others
    specific_features = {}

    for method in df.columns:
        # Features with high absolute coefficient for this method
        high_this_method = df[df[method].abs() > threshold].index

        # Among those, find ones with low coefficient for other methods
        other_methods = [m for m in df.columns if m != method]
        specific = []

        for feature in high_this_method:
            # Check if low for all other methods
            if all(df.loc[feature, m] < threshold for m in other_methods):
                specific.append({
                    'feature': feature,
                    'coefficient': df.loc[feature, method],
                    'other_max': df.loc[feature, other_methods].abs().max()
                })

        specific_features[method] = specific

    return specific_features


def provide_interpretations():
    """Provide interpretations for key metric types."""
    interpretations = {
        'effective_rank': 'Dimensionality of weight space - higher = more diverse representations',
        'stable_rank': 'Effective number of significant singular values - stability indicator',
        'spectral_gap': 'Gap between largest singular values - indicates structure',
        'singular_value_ratio': 'Ratio of max to min singular values - dynamic range',
        'task_vector_cosine_similarity': 'Alignment of task-specific directions - higher = similar updates',
        'task_vector_l2_distance': 'Distance between task vectors - higher = more different',
        'weight_space_angle': 'Angle between weight spaces - geometric divergence',
        'activation_cosine_similarity': 'Alignment of activation patterns - functional similarity',
        'interaction_matrix_overlap': 'Overlap in how layers interact - architectural compatibility',
        'subspace_overlap': 'Overlap in weight subspaces - representational similarity',
        'encoder_gradient_cosine_similarity': 'Alignment of gradient directions in encoder',
        'input_gradient_cosine_similarity': 'Alignment of input sensitivities'
    }

    return interpretations


def main():
    # Paths
    results_dir = Path(PROJECT_ROOT / 'results/metric_linear_optimization')
    output_dir = results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("Linear Coefficient Analysis for Method-Specific Mergeability")
    print("="*70)
    print()

    # Load results
    print("Loading linear optimization results...")
    results = load_linear_results(results_dir)

    # Print validation correlations
    print("\nValidation Correlations:")
    print("-" * 50)
    for method, result in results.items():
        r = result['correlation']['validation']
        p = result['correlation']['validation_p_value']
        print(f"{method:15s}: r={r:.4f}, p={p:.2e}")
    print()

    # Create coefficient comparison
    print("Creating coefficient comparison...")
    df_coef = create_coefficient_comparison(results)

    # Save to CSV
    csv_path = output_dir / 'coefficients_comparison.csv'
    df_coef.to_csv(csv_path)
    print(f"Saved coefficients to: {csv_path}")
    print()

    # Identify top features
    print("Identifying top features per method...")
    top_features = identify_top_features(df_coef, top_k=10)

    print("\nTop 10 Features by Method:")
    print("="*70)
    for method, features in top_features.items():
        print(f"\n{method.upper()}:")
        print("-" * 50)
        for i, (feature, coef) in enumerate(features.items(), 1):
            sign = "+" if coef > 0 else ""
            print(f"{i:2d}. {feature:45s} {sign}{coef:7.4f}")
    print()

    # Compute statistics
    print("Computing feature importance statistics...")
    stats = compute_feature_importance_stats(df_coef)

    print("\nMost Important Features Overall (by mean |coefficient|):")
    print("-" * 60)
    for i, (feature, val) in enumerate(stats['mean_abs_coef'].head(15).items(), 1):
        print(f"{i:2d}. {feature:50s} {val:.4f}")
    print()

    print("Most Consistent Features Across Methods (low std/mean):")
    print("-" * 60)
    for i, (feature, val) in enumerate(stats['consistency'].head(10).items(), 1):
        avg_coef = df_coef.loc[feature].abs().mean()
        print(f"{i:2d}. {feature:45s} consistency={val:.3f}, avg|coef|={avg_coef:.3f}")
    print()

    # Analyze method specificity
    print("Analyzing method-specific features...")
    specific = analyze_method_specificity(df_coef, threshold=0.1)

    print("\nMethod-Specific Features (high for one method, low for others):")
    print("="*70)
    for method, features in specific.items():
        if features:
            print(f"\n{method.upper()}-SPECIFIC:")
            print("-" * 60)
            for feat in features:
                print(f"  {feat['feature']:45s} coef={feat['coefficient']:7.4f}, other_max={feat['other_max']:.4f}")
        else:
            print(f"\n{method.upper()}: No highly specific features found")
    print()

    # Create visualizations
    print("Creating visualizations...")
    plot_coefficient_heatmap(df_coef, output_dir / 'coefficient_heatmap.png')
    plot_top_features_comparison(top_features, output_dir / 'top_features_comparison.png')
    print()

    # Feature overlap analysis
    print("Analyzing feature overlap across methods...")
    top_5_sets = {method: set(features.head(5).index)
                  for method, features in top_features.items()}

    print("\nTop-5 Feature Overlap:")
    print("-" * 60)
    methods = list(top_5_sets.keys())
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            overlap = top_5_sets[m1] & top_5_sets[m2]
            overlap_pct = len(overlap) / 5 * 100
            print(f"{m1:12s} vs {m2:12s}: {len(overlap)}/5 overlap ({overlap_pct:.0f}%)")
            if overlap:
                print(f"  Common: {', '.join(list(overlap)[:3])}")
    print()

    # Summary for paper
    print("="*70)
    print("SUMMARY FOR PAPER")
    print("="*70)
    print()
    print("KEY FINDING: Different merge methods rely on different properties!")
    print()

    print("Method-Specific Dominant Features:")
    print("-" * 60)
    for method in ['tsv', 'weight_avg', 'arithmetic', 'isotropic']:
        top_3 = top_features[method].head(3)
        print(f"\n{method.upper()}:")
        for feature, coef in top_3.items():
            print(f"  • {feature:45s} ({coef:+.3f})")

    print("\n" + "="*70)
    print("Analysis complete! Check the output directory for visualizations.")
    print("="*70)


if __name__ == "__main__":
    main()
