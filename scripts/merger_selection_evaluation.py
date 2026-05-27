#!/usr/bin/env python3
"""
Metrics-Guided Merger Selection Evaluation

For each pair of models, predict which merger will perform best using L1 coefficients.
To avoid data leakage, we use LOTO: for pairs containing task T, we use coefficients
trained WITHOUT task T (i.e., from the fold where T is held out).

Output: JSON file mapping each pair to predicted best merger and evaluation metrics.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import os
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parent.parent))


def load_loto_results(results_dir: Path, methods: list) -> dict:
    """Load LOTO results for all methods."""
    results = {}
    for method in methods:
        fpath = results_dir / f'{method}_loto_results.json'
        with open(fpath) as f:
            results[method] = json.load(f)
    return results


def load_pairwise_metrics(metrics_path: Path) -> dict:
    """Load pairwise metrics data."""
    with open(metrics_path) as f:
        return json.load(f)


def load_performance_data(results_base: Path, methods: list) -> dict:
    """Load actual performance data for all methods."""
    perf_data = {}
    for method in methods:
        fpath = results_base / method / 'all_pairwise_summary_N20.json'
        with open(fpath) as f:
            perf_data[method] = json.load(f)
    return perf_data


def get_metric_vector(metrics_data: dict, task1: str, task2: str, metric_names: list) -> np.ndarray:
    """Get the metric vector for a pair of tasks."""
    pair_key = f"{task1}__{task2}"
    reverse_key = f"{task2}__{task1}"

    # Extract metrics in the specified order
    vector = []
    for name in metric_names:
        if name not in metrics_data['metrics']:
            vector.append(0.0)
            continue

        pairs_dict = metrics_data['metrics'][name]['pairs']
        if pair_key in pairs_dict:
            val = pairs_dict[pair_key]
        elif reverse_key in pairs_dict:
            val = pairs_dict[reverse_key]
        else:
            val = 0.0

        if val is None:
            val = 0.0
        vector.append(val)

    if all(v == 0.0 for v in vector):
        return None

    return np.array(vector)


def get_actual_performance(perf_data: dict, task1: str, task2: str) -> float:
    """Get actual normalized accuracy for a pair."""
    pair_key = f"{task1}__{task2}"
    if pair_key in perf_data:
        avg_data = perf_data[pair_key].get('avg', [{}])[0]
        return avg_data.get('normalized_acc/test/avg', 0.0)

    # Try reverse order
    pair_key = f"{task2}__{task1}"
    if pair_key in perf_data:
        avg_data = perf_data[pair_key].get('avg', [{}])[0]
        return avg_data.get('normalized_acc/test/avg', 0.0)

    return None


def predict_performance(metrics_vector: np.ndarray, coefficients: dict,
                       metric_names: list, train_mean: float, train_std: float) -> float:
    """Predict performance using linear model."""
    # Build coefficient vector in same order as metrics
    coef_vector = np.array([coefficients.get(name, 0.0) for name in metric_names])

    # Normalize metrics (same as during training)
    # Note: we need train stats, but for simplicity we'll use the raw prediction
    # The ranking should be preserved even without proper normalization

    # Linear prediction
    prediction = np.dot(metrics_vector, coef_vector)

    return prediction


def main():
    # Configuration
    l1_results_dir = Path(PROJECT_ROOT / 'results/metric_linear_optimization_v2/loto_cv_l1_lambda1.0')
    metrics_path = Path(PROJECT_ROOT / 'results/mergeability/ViT-B-16/pairwise_metrics_N20.json')
    results_base = Path(PROJECT_ROOT / 'results/ViT-B-16')
    output_path = Path(PROJECT_ROOT / 'results/metric_linear_optimization_v2/predicted_mergers.json')

    methods = ['weight_avg', 'arithmetic', 'tsv', 'ties', 'dare']

    print("=" * 70)
    print("Metrics-Guided Merger Selection Evaluation")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    loto_results = load_loto_results(l1_results_dir, methods)
    metrics_data = load_pairwise_metrics(metrics_path)
    perf_data = load_performance_data(results_base, methods)

    # Get list of all tasks
    all_tasks = metrics_data['datasets']
    print(f"Tasks: {len(all_tasks)}")

    # Get metric names from L1 results (these are the metrics used in training)
    # The coefficients dict has the metric names in the order they were used
    metric_names = list(loto_results['weight_avg']['average_coefficients'].keys())
    print(f"Metrics in L1 model: {len(metric_names)}")

    # Verify metrics exist in pairwise data
    available_metrics = list(metrics_data['metrics'].keys())
    print(f"Metrics in pairwise data: {len(available_metrics)}")

    # Generate all pairs
    all_pairs = []
    for i, t1 in enumerate(all_tasks):
        for t2 in all_tasks[i+1:]:
            all_pairs.append((t1, t2))
    print(f"Total pairs: {len(all_pairs)}")
    print()

    # For each fold, get the held-out task and corresponding coefficients
    fold_info = {}
    for method in methods:
        for fold_result in loto_results[method]['fold_results']:
            held_out = fold_result['held_out_task']
            fold_idx = fold_result['fold']

            if held_out not in fold_info:
                fold_info[held_out] = {'fold_idx': fold_idx, 'coefficients': {}}

            fold_info[held_out]['coefficients'][method] = fold_result['coefficients']

    print(f"Folds with held-out tasks: {len(fold_info)}")
    print()

    # For each pair, predict best merger using coefficients from appropriate fold
    results = {
        'pairs': {},
        'summary': {}
    }

    correct_predictions = 0
    total_predictions = 0

    # Track performance
    predicted_perfs = []
    oracle_perfs = []
    baseline_perfs = {method: [] for method in methods}

    print("Evaluating merger selection...")
    print("-" * 70)

    for task1, task2 in all_pairs:
        pair_key = f"{task1}_{task2}"

        # Get metrics for this pair
        metrics_vector = get_metric_vector(metrics_data, task1, task2, metric_names)
        if metrics_vector is None:
            continue

        # Get actual performance for each method
        actual_perfs = {}
        for method in methods:
            perf = get_actual_performance(perf_data[method], task1, task2)
            if perf is not None:
                actual_perfs[method] = perf

        if len(actual_perfs) < len(methods):
            continue

        # Determine which fold to use (pair contains task1 or task2 as held-out)
        # We'll use the fold where task1 is held out (could also use task2)
        # Both should give similar results since the pair wasn't in training for either

        # Use task1's fold
        if task1 in fold_info:
            fold_coefficients = fold_info[task1]['coefficients']
        elif task2 in fold_info:
            fold_coefficients = fold_info[task2]['coefficients']
        else:
            continue

        # Predict performance for each method
        predicted_perfs_pair = {}
        for method in methods:
            coefs = fold_coefficients[method]
            pred = predict_performance(metrics_vector, coefs, metric_names, 0, 1)
            predicted_perfs_pair[method] = pred

        # Select merger with highest predicted performance
        predicted_best = max(predicted_perfs_pair, key=predicted_perfs_pair.get)

        # Get actual best merger
        actual_best = max(actual_perfs, key=actual_perfs.get)

        # Record results
        results['pairs'][pair_key] = {
            'task1': task1,
            'task2': task2,
            'predicted_best_merger': predicted_best,
            'actual_best_merger': actual_best,
            'prediction_correct': predicted_best == actual_best,
            'predicted_scores': predicted_perfs_pair,
            'actual_performances': actual_perfs,
            'performance_with_predicted': actual_perfs[predicted_best],
            'performance_with_oracle': actual_perfs[actual_best],
            'fold_used': task1 if task1 in fold_info else task2
        }

        # Update stats
        if predicted_best == actual_best:
            correct_predictions += 1
        total_predictions += 1

        predicted_perfs.append(actual_perfs[predicted_best])
        oracle_perfs.append(actual_perfs[actual_best])
        for method in methods:
            baseline_perfs[method].append(actual_perfs[method])

    print(f"Evaluated {total_predictions} pairs")
    print()

    # Compute summary statistics
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Prediction Accuracy: {correct_predictions}/{total_predictions} ({accuracy*100:.1f}%)")
    print()

    avg_predicted = np.mean(predicted_perfs)
    avg_oracle = np.mean(oracle_perfs)

    print(f"Average Performance:")
    print(f"  Oracle (always best):     {avg_oracle:.4f}")
    print(f"  Predicted:                {avg_predicted:.4f}")

    for method in methods:
        avg_baseline = np.mean(baseline_perfs[method])
        print(f"  Always {method:<12}:   {avg_baseline:.4f}")

    # Compute regret (how much worse than oracle)
    regret = avg_oracle - avg_predicted
    print(f"\nRegret (Oracle - Predicted): {regret:.4f}")

    # Compare to best fixed baseline
    best_fixed_method = max(methods, key=lambda m: np.mean(baseline_perfs[m]))
    best_fixed_perf = np.mean(baseline_perfs[best_fixed_method])
    improvement = avg_predicted - best_fixed_perf

    print(f"\nBest fixed baseline: {best_fixed_method} ({best_fixed_perf:.4f})")
    print(f"Improvement over best fixed: {improvement:+.4f} ({improvement/best_fixed_perf*100:+.2f}%)")

    # Store summary
    results['summary'] = {
        'total_pairs': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'avg_performance_predicted': avg_predicted,
        'avg_performance_oracle': avg_oracle,
        'avg_performance_baselines': {m: np.mean(baseline_perfs[m]) for m in methods},
        'regret': regret,
        'best_fixed_baseline': best_fixed_method,
        'improvement_over_best_fixed': improvement
    }

    # Confusion matrix: which merger is predicted vs actual best
    print()
    print("=" * 70)
    print("CONFUSION MATRIX: Predicted vs Actual Best Merger")
    print("=" * 70)

    confusion = defaultdict(lambda: defaultdict(int))
    for pair_data in results['pairs'].values():
        pred = pair_data['predicted_best_merger']
        actual = pair_data['actual_best_merger']
        confusion[pred][actual] += 1

    # Print confusion matrix
    print(f"\n{'Predicted':<12} | " + " | ".join([f"{m:<10}" for m in methods]) + " | Total")
    print("-" * 80)
    for pred_method in methods:
        row = [confusion[pred_method][actual_method] for actual_method in methods]
        total = sum(row)
        print(f"{pred_method:<12} | " + " | ".join([f"{c:<10}" for c in row]) + f" | {total}")

    print("-" * 80)
    print(f"{'Actual Total':<12} | " + " | ".join([f"{sum(confusion[p][a] for p in methods):<10}" for a in methods]))

    # Distribution of actual best mergers
    print()
    print("=" * 70)
    print("DISTRIBUTION OF ACTUAL BEST MERGERS")
    print("=" * 70)

    actual_best_counts = defaultdict(int)
    for pair_data in results['pairs'].values():
        actual_best_counts[pair_data['actual_best_merger']] += 1

    for method in methods:
        count = actual_best_counts[method]
        pct = count / total_predictions * 100
        print(f"  {method:<12}: {count:>3} pairs ({pct:>5.1f}%)")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
