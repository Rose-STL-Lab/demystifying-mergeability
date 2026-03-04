#!/usr/bin/env python3
"""
Analyze statistical significance of LOTO cross-validation results.
Computes confidence intervals, paired t-tests, and permutation tests.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import argparse


def load_fold_results(results_path: Path, method_name: str) -> dict:
    """Load per-fold val_r values from LOTO results."""
    file_path = results_path / f"{method_name}_loto_results.json"
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        data = json.load(f)

    fold_vals = [fold['val_r'] for fold in data['fold_results']]
    return {
        'method': method_name,
        'val_r_values': np.array(fold_vals),
        'mean': np.mean(fold_vals),
        'std': np.std(fold_vals, ddof=1),
        'n_folds': len(fold_vals)
    }


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple:
    """Compute confidence interval for the mean."""
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h


def paired_t_test(values1: np.ndarray, values2: np.ndarray) -> dict:
    """Perform paired t-test between two methods."""
    t_stat, p_value = stats.ttest_rel(values1, values2)
    diff = values1 - values2
    mean_diff = np.mean(diff)
    ci_low, ci_high = compute_confidence_interval(diff)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'significant_at_0.05': p_value < 0.05
    }


def permutation_test(values1: np.ndarray, values2: np.ndarray, n_permutations: int = 10000) -> dict:
    """Perform permutation test to compare two methods."""
    observed_diff = np.mean(values1) - np.mean(values2)

    combined = np.stack([values1, values2], axis=1)
    n_samples = len(values1)

    count = 0
    permuted_diffs = []

    for _ in range(n_permutations):
        # For each sample, randomly swap the two values
        swap_mask = np.random.randint(0, 2, size=n_samples).astype(bool)
        permuted = combined.copy()
        permuted[swap_mask] = permuted[swap_mask][:, ::-1]

        perm_diff = np.mean(permuted[:, 0]) - np.mean(permuted[:, 1])
        permuted_diffs.append(perm_diff)

        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations

    return {
        'observed_difference': observed_diff,
        'p_value': p_value,
        'significant_at_0.05': p_value < 0.05,
        'n_permutations': n_permutations
    }


def bootstrap_confidence_interval(values: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval for the mean."""
    bootstrap_means = []
    n = len(values)

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(description='Analyze LOTO statistical significance')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing LOTO results')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['weight_avg', 'arithmetic', 'tsv', 'ties', 'dare'],
                        help='Methods to analyze')
    parser.add_argument('--compare_dirs', type=str, nargs=2, default=None,
                        help='Two directories to compare (e.g., baseline vs L1)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("=" * 70)
    print("LOTO Cross-Validation Statistical Analysis")
    print("=" * 70)

    # Load results for each method
    results = {}
    for method in args.methods:
        data = load_fold_results(results_dir, method)
        if data:
            results[method] = data

    # Print confidence intervals for each method
    print("\n1. CONFIDENCE INTERVALS (95%)")
    print("-" * 70)
    print(f"{'Method':<15} {'Mean val_r':<12} {'Std':<10} {'95% CI':<25} {'SE':<10}")
    print("-" * 70)

    for method, data in results.items():
        ci_low, ci_high = compute_confidence_interval(data['val_r_values'])
        se = stats.sem(data['val_r_values'])
        print(f"{method:<15} {data['mean']:<12.4f} {data['std']:<10.4f} [{ci_low:.4f}, {ci_high:.4f}]  {se:<10.4f}")

    # Bootstrap confidence intervals
    print("\n2. BOOTSTRAP CONFIDENCE INTERVALS (95%, 10000 samples)")
    print("-" * 70)
    print(f"{'Method':<15} {'Mean val_r':<12} {'Bootstrap 95% CI':<25}")
    print("-" * 70)

    for method, data in results.items():
        ci_low, ci_high = bootstrap_confidence_interval(data['val_r_values'])
        print(f"{method:<15} {data['mean']:<12.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    # Pairwise comparisons between methods
    if len(results) > 1:
        print("\n3. PAIRED T-TESTS (comparing methods)")
        print("-" * 70)

        methods = list(results.keys())
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                test_result = paired_t_test(
                    results[method1]['val_r_values'],
                    results[method2]['val_r_values']
                )
                sig_marker = "*" if test_result['significant_at_0.05'] else ""
                print(f"\n{method1} vs {method2}:")
                print(f"  Mean difference: {test_result['mean_difference']:.4f}")
                print(f"  95% CI of diff:  [{test_result['ci_95_low']:.4f}, {test_result['ci_95_high']:.4f}]")
                print(f"  t-statistic:     {test_result['t_statistic']:.4f}")
                print(f"  p-value:         {test_result['p_value']:.4f} {sig_marker}")

    # Compare two experiment directories if provided
    if args.compare_dirs:
        dir1, dir2 = Path(args.compare_dirs[0]), Path(args.compare_dirs[1])

        print("\n" + "=" * 70)
        print(f"COMPARISON: {dir1.name} vs {dir2.name}")
        print("=" * 70)

        for method in args.methods:
            data1 = load_fold_results(dir1, method)
            data2 = load_fold_results(dir2, method)

            if data1 and data2:
                print(f"\n{method.upper()}")
                print("-" * 40)
                print(f"  {dir1.name}: {data1['mean']:.4f} ± {data1['std']:.4f}")
                print(f"  {dir2.name}: {data2['mean']:.4f} ± {data2['std']:.4f}")

                # Paired t-test
                t_result = paired_t_test(data2['val_r_values'], data1['val_r_values'])
                print(f"  Improvement: {t_result['mean_difference']:.4f}")
                print(f"  p-value (paired t-test): {t_result['p_value']:.4f}", end="")
                if t_result['significant_at_0.05']:
                    print(" *")
                else:
                    print("")

                # Permutation test
                perm_result = permutation_test(data2['val_r_values'], data1['val_r_values'])
                print(f"  p-value (permutation):   {perm_result['p_value']:.4f}", end="")
                if perm_result['significant_at_0.05']:
                    print(" *")
                else:
                    print("")

    # Effect size (Cohen's d) for method comparisons
    if len(results) > 1:
        print("\n4. EFFECT SIZES (Cohen's d)")
        print("-" * 70)

        methods = list(results.keys())
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                vals1 = results[method1]['val_r_values']
                vals2 = results[method2]['val_r_values']

                # Pooled standard deviation
                pooled_std = np.sqrt(((len(vals1)-1)*np.var(vals1, ddof=1) +
                                      (len(vals2)-1)*np.var(vals2, ddof=1)) /
                                     (len(vals1) + len(vals2) - 2))

                cohens_d = (np.mean(vals1) - np.mean(vals2)) / pooled_std

                # Interpret effect size
                if abs(cohens_d) < 0.2:
                    effect = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect = "small"
                elif abs(cohens_d) < 0.8:
                    effect = "medium"
                else:
                    effect = "large"

                print(f"{method1} vs {method2}: d = {cohens_d:.4f} ({effect})")


if __name__ == '__main__':
    main()
