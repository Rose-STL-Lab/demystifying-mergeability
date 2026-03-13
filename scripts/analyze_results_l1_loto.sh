#!/bin/bash
cd /home/ubuntu/thesis/MM/Mergeability-Bench && python3 << 'EOF'
import json
import numpy as np

# Load results
results_dir = '/home/ubuntu/thesis/MM/Mergeability-Bench/results/metric_linear_optimization_v2/vit-b-16_AdamW/loto_cv_l1_lambda1.0'
methods = ['weight_avg', 'arithmetic', 'tsv', 'ties', 'dare']

all_results = {}
for method in methods:
    with open(f'{results_dir}/{method}_loto_results.json') as f:
        all_results[method] = json.load(f)

# 1. Compute metric overlap between methods
print("=" * 70)
print("1. METRIC OVERLAP BETWEEN METHODS (by coefficient magnitude)")
print("=" * 70)

def get_top_metrics(coefficients, n=10):
    """Get top n metrics by absolute coefficient magnitude."""
    sorted_metrics = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    return set([m[0] for m in sorted_metrics[:n]])

for n_top in [5, 10]:
    print(f"\nTop-{n_top} metrics overlap:")
    top_metrics = {m: get_top_metrics(all_results[m]['average_coefficients'], n_top) for m in methods}

    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            overlap = len(top_metrics[m1] & top_metrics[m2])
            pct = overlap / n_top * 100
            print(f"  {m1} vs {m2}: {overlap}/{n_top} ({pct:.0f}%)")

# 2. Validation Pearson correlation range
print("\n" + "=" * 70)
print("2. VALIDATION PEARSON CORRELATION RANGE")
print("=" * 70)

val_r_means = []
for method in methods:
    val_r_mean = all_results[method]['per_fold_stats']['val_r_mean']
    val_r_std = all_results[method]['per_fold_stats']['val_r_std']
    val_r_means.append(val_r_mean)
    print(f"{method}: val_r = {val_r_mean:.4f} ± {val_r_std:.4f}")

print(f"\nRange: [{min(val_r_means):.4f}, {max(val_r_means):.4f}]")

# 3. Per-fold mean and std for train and validation
print("\n" + "=" * 70)
print("3. PER-FOLD MEAN AND STD FOR TRAIN/VAL CORRELATIONS")
print("=" * 70)

print(f"\n{'Method':<15} {'Train r mean':<15} {'Train r std':<15} {'Val r mean':<15} {'Val r std':<15} {'N nonzero':<15}")
print("-" * 90)
for method in methods:
    stats = all_results[method]['per_fold_stats']
    n_nonzero = stats.get('n_nonzero_mean', 'N/A')
    if isinstance(n_nonzero, float):
        print(f"{method:<15} {stats['train_r_mean']:<15.4f} {stats['train_r_std']:<15.4f} {stats['val_r_mean']:<15.4f} {stats['val_r_std']:<15.4f} {n_nonzero:<15.1f}")
    else:
        print(f"{method:<15} {stats['train_r_mean']:<15.4f} {stats['train_r_std']:<15.4f} {stats['val_r_mean']:<15.4f} {stats['val_r_std']:<15.4f} {n_nonzero:<15}")

# 4. Top-5 metrics by average coefficient magnitude
print("\n" + "=" * 70)
print("4. TOP-5 METRICS BY AVERAGE COEFFICIENT MAGNITUDE")
print("=" * 70)

for method in methods:
    coefs = all_results[method]['average_coefficients']
    stds = all_results[method]['coefficient_std']
    freqs = all_results[method].get('nonzero_frequency', {})
    sorted_metrics = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"\n{method.upper()}:")
    for i, (metric, coef) in enumerate(sorted_metrics, 1):
        std = stds[metric]
        freq = freqs.get(metric, 1.0)
        print(f"  {i}. {metric}: {coef:+.6f} (±{std:.6f}) [freq={freq:.0%}]")

# 5. TOP-5 METRICS BY IMPORTANCE SCORE (|coef| × frequency)
print("\n" + "=" * 70)
print("5. TOP-5 METRICS BY IMPORTANCE SCORE (|avg_coef| × nonzero_freq)")
print("=" * 70)

for method in methods:
    coefs = all_results[method]['average_coefficients']
    stds = all_results[method]['coefficient_std']
    freqs = all_results[method].get('nonzero_frequency', {})

    # Compute importance score
    importance_scores = {}
    for metric, coef in coefs.items():
        freq = freqs.get(metric, 1.0)
        importance_scores[metric] = abs(coef) * freq

    sorted_metrics = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n{method.upper()}:")
    for i, (metric, score) in enumerate(sorted_metrics, 1):
        coef = coefs[metric]
        freq = freqs.get(metric, 1.0)
        print(f"  {i}. {metric}: score={score:.6f} (coef={coef:+.6f}, freq={freq:.0%})")

# 6. CONSISTENTLY SELECTED METRICS (high frequency across all mergers)
print("\n" + "=" * 70)
print("6. CONSISTENTLY SELECTED METRICS (freq >= 80% for ALL mergers)")
print("=" * 70)

all_metrics = list(all_results['weight_avg']['average_coefficients'].keys())

consistent_freq_metrics = []
for metric in all_metrics:
    freqs = [all_results[m].get('nonzero_frequency', {}).get(metric, 0.0) for m in methods]
    min_freq = min(freqs)
    if min_freq >= 0.8:
        avg_freq = np.mean(freqs)
        consistent_freq_metrics.append((metric, min_freq, avg_freq, freqs))

consistent_freq_metrics.sort(key=lambda x: x[2], reverse=True)

print(f"\nFound {len(consistent_freq_metrics)} metrics selected >= 80% across ALL mergers:")
for metric, min_freq, avg_freq, freqs in consistent_freq_metrics:
    freq_str = ", ".join([f"{f:.0%}" for f in freqs])
    print(f"  {metric}: min={min_freq:.0%}, avg={avg_freq:.0%} [{freq_str}]")

# 6b. Metrics with 100% frequency across all mergers
print("\n" + "-" * 70)
print("UNIVERSALLY SELECTED METRICS (freq = 100% for ALL mergers):")
print("-" * 70)

universal_metrics = [(m, mf, af, fs) for m, mf, af, fs in consistent_freq_metrics if mf >= 1.0]
if universal_metrics:
    for metric, min_freq, avg_freq, freqs in universal_metrics:
        coefs = [all_results[m]['average_coefficients'][metric] for m in methods]
        coef_str = ", ".join([f"{c:+.6f}" for c in coefs])
        print(f"  {metric}: [{coef_str}]")
else:
    print("  (None found)")

# 7. Stable metrics (consistent sign across methods)
print("\n" + "=" * 70)
print("7. STABLE METRICS ANALYSIS (consistent sign across all mergers)")
print("=" * 70)

print("\nMetrics with CONSISTENT SIGN across all methods:")
consistent_sign_metrics = []
for metric in all_metrics:
    coefs = [all_results[m]['average_coefficients'][metric] for m in methods]
    # Filter out zero coefficients for sign analysis
    nonzero_coefs = [c for c in coefs if abs(c) > 1e-10]
    if len(nonzero_coefs) >= len(methods) // 2:  # At least half must be nonzero
        if all(c > 0 for c in nonzero_coefs) or all(c < 0 for c in nonzero_coefs):
            avg_coef = np.mean(coefs)
            sign = "+" if avg_coef > 0 else "-"
            freqs = [all_results[m].get('nonzero_frequency', {}).get(metric, 0.0) for m in methods]
            avg_freq = np.mean(freqs)
            consistent_sign_metrics.append((metric, avg_coef, sign, coefs, avg_freq))

consistent_sign_metrics.sort(key=lambda x: abs(x[1]), reverse=True)

if consistent_sign_metrics:
    print(f"\nFound {len(consistent_sign_metrics)} metrics with consistent sign:")
    for metric, avg_coef, sign, coefs, avg_freq in consistent_sign_metrics:
        coef_str = ", ".join([f"{c:+.6f}" for c in coefs])
        print(f"  {sign} {metric}: [{coef_str}] (avg: {avg_coef:+.6f}, freq: {avg_freq:.0%})")
else:
    print("\n  (No metrics found with consistent sign across all mergers)")

# 8. METRICS THAT ARE BOTH STABLE AND CONSISTENTLY SELECTED
print("\n" + "=" * 70)
print("8. CORE PREDICTORS (stable sign + freq >= 80% across all mergers)")
print("=" * 70)

core_predictors = []
for metric, avg_coef, sign, coefs, avg_freq in consistent_sign_metrics:
    freqs = [all_results[m].get('nonzero_frequency', {}).get(metric, 0.0) for m in methods]
    min_freq = min(freqs)
    if min_freq >= 0.8:
        core_predictors.append((metric, avg_coef, sign, coefs, freqs))

if core_predictors:
    print(f"\nFound {len(core_predictors)} CORE PREDICTORS:")
    for metric, avg_coef, sign, coefs, freqs in core_predictors:
        freq_str = ", ".join([f"{f:.0%}" for f in freqs])
        print(f"  {sign} {metric}")
        print(f"      avg_coef: {avg_coef:+.6f}")
        print(f"      freqs: [{freq_str}]")
else:
    print("\n  (No core predictors found)")

# 9. Metrics with INCONSISTENT SIGN across methods
print("\n" + "=" * 70)
print("9. METRICS WITH INCONSISTENT SIGN ACROSS MERGERS")
print("=" * 70)

inconsistent_metrics = []
for metric in all_metrics:
    coefs = [all_results[m]['average_coefficients'][metric] for m in methods]
    nonzero_coefs = [c for c in coefs if abs(c) > 1e-10]
    if len(nonzero_coefs) >= 2:  # At least 2 nonzero to compare signs
        has_positive = any(c > 0 for c in nonzero_coefs)
        has_negative = any(c < 0 for c in nonzero_coefs)
        if has_positive and has_negative:
            freqs = [all_results[m].get('nonzero_frequency', {}).get(metric, 0.0) for m in methods]
            avg_freq = np.mean(freqs)
            inconsistent_metrics.append((metric, coefs, avg_freq))

inconsistent_metrics.sort(key=lambda x: x[2], reverse=True)

print(f"\nFound {len(inconsistent_metrics)} metrics with inconsistent sign:")
for metric, coefs, avg_freq in inconsistent_metrics:
    coef_str = ", ".join([f"{c:+.6f}" for c in coefs])
    print(f"  {metric}: [{coef_str}] (avg_freq: {avg_freq:.0%})")

# 10. Global sign agreement across all metrics
print("\n" + "=" * 70)
print("10. SIGN AGREEMENT ACROSS ALL METRICS")
print("=" * 70)

def sign_fn(x):
    if x > 1e-10:
        return 1
    if x < -1e-10:
        return -1
    return 0

for i, m1 in enumerate(methods):
    for m2 in methods[i+1:]:
        agree = 0
        total = 0

        for metric in all_metrics:
            c1 = all_results[m1]['average_coefficients'][metric]
            c2 = all_results[m2]['average_coefficients'][metric]

            s1 = sign_fn(c1)
            s2 = sign_fn(c2)

            if s1 == 0 or s2 == 0:
                continue

            total += 1
            if s1 == s2:
                agree += 1

        pct = 100 * agree / total if total > 0 else 0.0
        print(f"{m1} vs {m2}: {agree}/{total} metrics ({pct:.1f}% sign agreement)")

# 11. Summary statistics
print("\n" + "=" * 70)
print("11. SUMMARY")
print("=" * 70)

print(f"\nTotal metrics: {len(all_metrics)}")
print(f"Metrics with consistent sign: {len(consistent_sign_metrics)}")
print(f"Metrics consistently selected (>=80%): {len(consistent_freq_metrics)}")
print(f"Core predictors (both): {len(core_predictors)}")
print(f"Metrics with inconsistent sign: {len(inconsistent_metrics)}")

print("\nBest performing mergers by val_r:")
sorted_methods = sorted(methods, key=lambda m: all_results[m]['per_fold_stats']['val_r_mean'], reverse=True)
for i, method in enumerate(sorted_methods, 1):
    val_r = all_results[method]['per_fold_stats']['val_r_mean']
    val_r_std = all_results[method]['per_fold_stats']['val_r_std']
    print(f"  {i}. {method}: {val_r:.4f} ± {val_r_std:.4f}")

EOF
