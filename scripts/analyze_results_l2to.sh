cd /home/ubuntu/thesis/MM/Mergeability-Bench && python3 << 'EOF'
import json
import numpy as np

# Load results
results_dir = '/home/ubuntu/thesis/MM/Mergeability-Bench/results/metric_linear_optimization/l2to_cv_l1_lambda1.0'
methods = ['weight_avg', 'arithmetic', 'tsv', 'isotropic']

all_results = {}
for method in methods:
    with open(f'{results_dir}/{method}_l2to_results.json') as f:
        all_results[method] = json.load(f)

# 1. Compute metric overlap between methods
print("=" * 70)
print("1. METRIC OVERLAP BETWEEN METHODS")
print("=" * 70)

def get_top_metrics(coefficients, n=10):
    """Get top n metrics by absolute coefficient magnitude."""
    sorted_metrics = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    return set([m[0] for m in sorted_metrics[:n]])

for n_top in [5, 10, 28]:
    print(f"\nTop-{n_top} metrics overlap:")
    top_metrics = {m: get_top_metrics(all_results[m]['average_coefficients'], n_top) for m in methods}

    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            overlap = len(top_metrics[m1] & top_metrics[m2])
            pct = overlap / n_top * 100
            print(f"  {m1} vs {m2}: {overlap}/{n_top} ({pct:.0f}%)")

# 2. Validation Pearson correlation (aggregate across all held-out pairs)
print("\n" + "=" * 70)
print("2. AGGREGATE VALIDATION PEARSON CORRELATION")
print("=" * 70)

val_r_values = []
for method in methods:
    val_r = all_results[method]['aggregate_metrics']['val_r']
    val_r_values.append(val_r)
    print(f"{method}: val_r = {val_r:.4f}")

print(f"\nRange: [{min(val_r_values):.2f}, {max(val_r_values):.2f}]")

# 3. Per-fold mean and std for training (L2TO has 1 val sample per fold, so aggregate val_r is used)
print("\n" + "=" * 70)
print("3. PER-FOLD TRAINING STATS AND AGGREGATE VALIDATION")
print("=" * 70)

print(f"\n{'Method':<15} {'Train r mean':<15} {'Train r std':<15} {'Agg Val r':<15} {'Nonzero mean':<15}")
print("-" * 75)
for method in methods:
    stats = all_results[method]['per_fold_stats']
    agg_val_r = all_results[method]['aggregate_metrics']['val_r']
    print(f"{method:<15} {stats['train_r_mean']:<15.4f} {stats['train_r_std']:<15.4f} {agg_val_r:<15.4f} {stats['n_nonzero_mean']:<15.1f}")

# 4. Top-5 metrics by average coefficient magnitude
print("\n" + "=" * 70)
print("4. TOP-5 METRICS BY AVERAGE COEFFICIENT MAGNITUDE")
print("=" * 70)

for method in methods:
    coefs = all_results[method]['average_coefficients']
    stds = all_results[method]['coefficient_std']
    sorted_metrics = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"\n{method.upper()}:")
    for i, (metric, coef) in enumerate(sorted_metrics, 1):
        std = stds[metric]
        print(f"  {i}. {metric}: {coef:+.2f} (±{std:.2f})")

# 5. Identify stable metrics (consistent sign across methods, low relative std)
print("\n" + "=" * 70)
print("6. STABLE METRICS ANALYSIS")
print("=" * 70)

all_metrics = list(all_results['weight_avg']['average_coefficients'].keys())

print("\nMetrics with CONSISTENT SIGN across all methods:")
consistent_sign_metrics = []
for metric in all_metrics:
    coefs = [all_results[m]['average_coefficients'][metric] for m in methods]
    if all(c > 0 for c in coefs) or all(c < 0 for c in coefs):
        avg_coef = np.mean(coefs)
        sign = "+" if avg_coef > 0 else "-"
        consistent_sign_metrics.append((metric, avg_coef, sign, coefs))

consistent_sign_metrics.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\nFound {len(consistent_sign_metrics)} metrics with consistent sign:")
for metric, avg_coef, sign, coefs in consistent_sign_metrics:
    coef_str = ", ".join([f"{c:+.5f}" for c in coefs])
    print(f"  {sign} {metric}: [{coef_str}] (avg: {avg_coef:+.2f})")

print("\n\nMetrics with INCONSISTENT SIGN across methods:")
for metric in all_metrics:
    coefs = [all_results[m]['average_coefficients'][metric] for m in methods]
    if not (all(c > 0 for c in coefs) or all(c < 0 for c in coefs)):
        coef_str = ", ".join([f"{c:+.5f}" for c in coefs])
        print(f"  {metric}: [{coef_str}]")

# 7. Global sign agreement across all metrics (no ranking)
print("\n" + "=" * 70)
print("7. SIGN AGREEMENT ACROSS ALL METRICS")
print("=" * 70)

def sign_fn(x):
    if x > 0:
        return 1
    if x < 0:
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
        print(f"{m1} vs {m2}: {agree}/{total} metrics ({pct:.5f}% sign agreement)")

EOF
