#!/usr/bin/env python3
"""
Reverse Greedy (Backward Elimination) for LOTO Cross-Validation.

This script performs backward elimination of metrics within each fold:
- Start with all 29 metrics, iteratively remove the metric whose removal hurts least
- Stop when removing any metric would decrease training correlation by more than threshold
- Validation data is only used for final evaluation (no leakage)
- Each fold may retain different metrics

Usage:
    python scripts/linear_optimization_loto_reverse_greedy.py
    python scripts/linear_optimization_loto_reverse_greedy.py --threshold 0.001
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from scipy.stats import pearsonr
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_merging.data_loader import load_json, extract_all_mergers_data


def normalize_metrics(metrics_array):
    """Normalize metrics to [-1, 1] range using min-max normalization."""
    min_vals = metrics_array.min(axis=0)
    max_vals = metrics_array.max(axis=0)

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    normalized = (metrics_array - min_vals) / ranges
    normalized = normalized * 2 - 1

    return normalized, min_vals, max_vals


def normalize_metrics_with_stats(metrics_array, min_vals, max_vals):
    """Normalize metrics using pre-computed min/max values (for validation data)."""
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    normalized = (metrics_array - min_vals) / ranges
    normalized = normalized * 2 - 1

    return normalized


def optimize_coefficients(metrics_train, performance_train,
                          n_iterations=1000, lr=0.01,
                          patience=50, convergence_threshold=1e-4):
    """
    Optimize linear coefficients on training data only.

    Returns:
        coefficients: Optimized coefficients
        train_r: Training Pearson r
    """
    n_metrics = metrics_train.shape[1]

    if n_metrics == 0:
        return np.array([]), 0.0

    # Initialize coefficients
    coefficients = torch.randn(n_metrics, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([coefficients], lr=lr)

    # Convert to tensors
    X_train = torch.FloatTensor(metrics_train)
    y_train = torch.FloatTensor(performance_train)

    best_train_loss = float('inf')
    patience_counter = 0

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Forward pass
        predictions = X_train @ coefficients

        # Loss: negative Pearson correlation
        pred_mean = predictions.mean()
        target_mean = y_train.mean()

        pred_centered = predictions - pred_mean
        target_centered = y_train - target_mean

        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())

        correlation = numerator / (denominator + 1e-8)
        loss = -correlation

        # Backward pass
        loss.backward()

        # Constraint: coefficients sum to 1
        with torch.no_grad():
            coefficients.data = coefficients.data / (coefficients.data.sum() + 1e-8)

        optimizer.step()

        # Early stopping
        if loss.item() < best_train_loss - convergence_threshold:
            best_train_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Final evaluation on training data
    with torch.no_grad():
        train_pred = (X_train @ coefficients).numpy()

    train_r, _ = pearsonr(train_pred, performance_train)

    return coefficients.detach().numpy(), train_r


def reverse_greedy_selection(metrics_train, performance_train, metric_names,
                              threshold=0.001, verbose=False):
    """
    Reverse greedy (backward elimination) selection of metrics based on training correlation.

    Args:
        metrics_train: Training metrics array (n_samples, n_metrics)
        performance_train: Training target values
        metric_names: List of metric names
        threshold: Maximum allowed decrease in correlation when removing a metric
        verbose: Print progress

    Returns:
        retained_indices: List of retained metric indices
        retained_names: List of retained metric names
        removed_indices: List of removed metric indices in order
        removed_names: List of removed metric names in order
        final_coefficients: Coefficients for retained metrics
        final_train_r: Final training correlation
        elimination_history: List of (metric_name, train_r, decrease) for each round
    """
    n_metrics = len(metric_names)
    retained_indices = list(range(n_metrics))
    removed_indices = []
    removed_names = []
    elimination_history = []

    # Get initial training correlation with all metrics
    initial_coefficients, current_train_r = optimize_coefficients(metrics_train, performance_train)

    if verbose:
        print(f"    Initial train_r with all {n_metrics} metrics: {current_train_r:.4f}")

    round_num = 0
    while len(retained_indices) > 1:  # Keep at least 1 metric
        round_num += 1
        best_metric_to_remove = None
        best_train_r_after_removal = -float('inf')
        smallest_decrease = float('inf')

        # Try removing each retained metric
        for candidate_idx in retained_indices:
            # Create candidate set without this metric
            candidate_indices = [i for i in retained_indices if i != candidate_idx]
            candidate_metrics = metrics_train[:, candidate_indices]

            # Optimize and evaluate
            _, train_r = optimize_coefficients(candidate_metrics, performance_train)

            decrease = current_train_r - train_r

            # We want to remove the metric with the smallest decrease (or largest increase)
            if decrease < smallest_decrease:
                smallest_decrease = decrease
                best_train_r_after_removal = train_r
                best_metric_to_remove = candidate_idx

        # Check if the smallest decrease exceeds threshold
        if smallest_decrease > threshold:
            if verbose:
                print(f"    Round {round_num}: Stopping - removing any metric would decrease by > {threshold}")
                print(f"    Best candidate would decrease by {smallest_decrease:.4f}")
            break

        # Remove the metric with smallest impact
        retained_indices.remove(best_metric_to_remove)
        removed_indices.append(best_metric_to_remove)
        removed_names.append(metric_names[best_metric_to_remove])

        elimination_history.append({
            'round': round_num,
            'removed_metric': metric_names[best_metric_to_remove],
            'train_r_after': float(best_train_r_after_removal),
            'decrease': float(smallest_decrease)
        })

        if verbose:
            print(f"    Round {round_num}: Removed '{metric_names[best_metric_to_remove]}' "
                  f"(train_r={best_train_r_after_removal:.4f}, Δ={-smallest_decrease:+.4f})")

        current_train_r = best_train_r_after_removal

    # Get final coefficients for retained metrics
    if retained_indices:
        retained_metrics = metrics_train[:, retained_indices]
        final_coefficients, final_train_r = optimize_coefficients(retained_metrics, performance_train)
    else:
        final_coefficients = np.array([])
        final_train_r = 0.0

    retained_names = [metric_names[i] for i in retained_indices]

    return (retained_indices, retained_names, removed_indices, removed_names,
            final_coefficients, final_train_r, elimination_history)


def run_loto_cv_reverse_greedy(metrics_array, performance_array, pair_names, all_tasks,
                                metric_names, threshold=0.001):
    """
    Run Leave-One-Task-Out cross-validation with reverse greedy metric elimination.

    Returns:
        results: Dictionary with fold results and aggregate metrics
    """
    n_tasks = len(all_tasks)
    n_metrics = len(metric_names)

    # Storage for results
    fold_results = []
    all_train_preds = []
    all_train_targets = []
    all_val_preds = []
    all_val_targets = []

    # Track coefficients across folds (with 0 for removed)
    fold_full_coefficients = []

    # Track retention frequency
    retention_counts = {name: 0 for name in metric_names}

    print(f"Running Reverse Greedy LOTO CV with {n_tasks} folds (threshold={threshold})...")
    print()

    for fold_idx, held_out_task in enumerate(all_tasks):
        print(f"Fold {fold_idx+1}/{n_tasks}: Held-out task = {held_out_task}")

        # Determine train and validation tasks
        train_tasks = [t for t in all_tasks if t != held_out_task]

        # Split pairs based on task membership
        train_indices = []
        val_indices = []

        for i, pair_name in enumerate(pair_names):
            task1, task2 = pair_name.split('__')

            if task1 in train_tasks and task2 in train_tasks:
                train_indices.append(i)
            elif task1 == held_out_task or task2 == held_out_task:
                val_indices.append(i)

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        if len(val_indices) == 0:
            print(f"  WARNING: No validation pairs for {held_out_task}, skipping")
            continue

        print(f"  Train pairs: {len(train_indices)}, Val pairs: {len(val_indices)}")

        # Extract raw data for this fold
        metrics_train_raw = metrics_array[train_indices]
        performance_train = performance_array[train_indices]
        metrics_val_raw = metrics_array[val_indices]
        performance_val = performance_array[val_indices]

        # Normalize using ONLY training data statistics (no leakage)
        metrics_train, min_vals, max_vals = normalize_metrics(metrics_train_raw)
        metrics_val = normalize_metrics_with_stats(metrics_val_raw, min_vals, max_vals)

        # Reverse greedy elimination on TRAINING data only
        (retained_indices, retained_names, removed_indices, removed_names,
         retained_coefficients, train_r, elimination_history) = \
            reverse_greedy_selection(metrics_train, performance_train, metric_names,
                                     threshold=threshold, verbose=True)

        print(f"  Retained {len(retained_names)} metrics (removed {len(removed_names)}), train_r={train_r:.4f}")

        # Evaluate on validation data (only once, no leakage)
        if retained_indices:
            retained_metrics_train = metrics_train[:, retained_indices]
            retained_metrics_val = metrics_val[:, retained_indices]
            train_preds = retained_metrics_train @ retained_coefficients
            val_preds = retained_metrics_val @ retained_coefficients
            val_r, _ = pearsonr(val_preds, performance_val)
        else:
            train_preds = np.zeros(len(performance_train))
            val_preds = np.zeros(len(performance_val))
            val_r = 0.0

        print(f"  Validation r={val_r:.4f}")

        # Store predictions for aggregate evaluation
        all_train_preds.append(train_preds)
        all_train_targets.append(performance_train)
        all_val_preds.append(val_preds)
        all_val_targets.append(performance_val)

        # Create full coefficient vector (0 for removed metrics)
        full_coefficients = np.zeros(n_metrics)
        for idx, coef in zip(retained_indices, retained_coefficients):
            full_coefficients[idx] = coef
        fold_full_coefficients.append(full_coefficients)

        # Update retention counts
        for name in retained_names:
            retention_counts[name] += 1

        # Store fold results
        fold_results.append({
            'fold': fold_idx,
            'held_out_task': held_out_task,
            'n_train_pairs': len(train_indices),
            'n_val_pairs': len(val_indices),
            'train_r': float(train_r),
            'val_r': float(val_r),
            'n_retained_metrics': len(retained_names),
            'n_removed_metrics': len(removed_names),
            'retained_metrics': retained_names,
            'removed_metrics': removed_names,
            'retained_coefficients': {name: float(coef) for name, coef in zip(retained_names, retained_coefficients)},
            'elimination_history': elimination_history
        })

        print()

    # Aggregate results
    print("="*70)
    print("Aggregate Results")
    print("="*70)

    # Concatenate all predictions
    all_train_preds = np.concatenate(all_train_preds)
    all_train_targets = np.concatenate(all_train_targets)
    all_val_preds = np.concatenate(all_val_preds)
    all_val_targets = np.concatenate(all_val_targets)

    # Compute aggregate correlations
    aggregate_train_r, aggregate_train_p = pearsonr(all_train_preds, all_train_targets)
    aggregate_val_r, aggregate_val_p = pearsonr(all_val_preds, all_val_targets)

    print(f"Aggregate Training: r={aggregate_train_r:.4f}, p={aggregate_train_p:.2e}")
    print(f"Aggregate Validation: r={aggregate_val_r:.4f}, p={aggregate_val_p:.2e}")

    # Per-fold statistics
    fold_train_r = [f['train_r'] for f in fold_results]
    fold_val_r = [f['val_r'] for f in fold_results]
    fold_n_retained = [f['n_retained_metrics'] for f in fold_results]

    print(f"Per-fold: Train r={np.mean(fold_train_r):.4f}±{np.std(fold_train_r):.4f}")
    print(f"Per-fold: Val r={np.mean(fold_val_r):.4f}±{np.std(fold_val_r):.4f}")
    print(f"Per-fold: N retained={np.mean(fold_n_retained):.1f}±{np.std(fold_n_retained):.1f}")

    # Average coefficients across folds
    avg_coefficients = np.mean(fold_full_coefficients, axis=0)
    std_coefficients = np.std(fold_full_coefficients, axis=0)

    # Retention frequency (proportion of folds where each metric was retained)
    retention_frequency = {name: count / n_tasks for name, count in retention_counts.items()}

    results = {
        'aggregate_metrics': {
            'train_r': float(aggregate_train_r),
            'train_p': float(aggregate_train_p),
            'val_r': float(aggregate_val_r),
            'val_p': float(aggregate_val_p)
        },
        'per_fold_stats': {
            'train_r_mean': float(np.mean(fold_train_r)),
            'train_r_std': float(np.std(fold_train_r)),
            'val_r_mean': float(np.mean(fold_val_r)),
            'val_r_std': float(np.std(fold_val_r)),
            'n_retained_mean': float(np.mean(fold_n_retained)),
            'n_retained_std': float(np.std(fold_n_retained))
        },
        'average_coefficients': {name: float(coef) for name, coef in zip(metric_names, avg_coefficients)},
        'coefficient_std': {name: float(std) for name, std in zip(metric_names, std_coefficients)},
        'retention_frequency': retention_frequency,
        'fold_results': fold_results,
        'reverse_greedy_params': {
            'threshold': threshold
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Reverse Greedy (Backward Elimination) LOTO Cross-Validation')
    parser.add_argument('--threshold', type=float, default=0.001,
                        help='Maximum allowed decrease in training correlation when removing a metric')
    parser.add_argument('--model', type=str, default='ViT-B-16', choices=['ViT-B-16', 'ViT-B-32'],
                        help='Model architecture to use')
    parser.add_argument('--exclude_metrics', type=str, nargs='+', default=[],
                        help='List of metric names to exclude')
    args = parser.parse_args()

    exclude_metrics = set(args.exclude_metrics)
    model = args.model

    # Configuration - paths based on model
    base_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results')
    metrics_path = base_path / 'mergeability' / model / 'pairwise_metrics_N20.json'
    results_base_path = base_path / model

    # Output directory - always use model-specific subfolder
    output_dir = base_path / 'metric_linear_optimization_v2' / model.lower() / 'loto_cv_reverse_greedy_selection'
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_methods = ['weight_avg', 'arithmetic', 'tsv', 'ties', 'dare']

    print("="*70)
    print(f"Reverse Greedy (Backward Elimination) with LOTO CV (threshold={args.threshold})")
    print(f"Model: {model}")
    print("="*70)
    print()

    # Load metrics data
    print("Loading data...")
    metrics_data = load_json(metrics_path)

    # Load performance data for all methods
    performance_data_dict = {}
    for method in merge_methods:
        perf_path = results_base_path / method / 'all_pairwise_summary_N20.json'
        if not perf_path.exists():
            print(f"Warning: {perf_path} not found, skipping {method}")
            continue
        performance_data_dict[method] = load_json(perf_path)

    print(f"Loaded data for methods: {list(performance_data_dict.keys())}")
    print()

    # Extract pairwise data
    print("Extracting pairwise data...")
    metrics_array, performance_matrix, pair_names, metric_names, merge_methods = \
        extract_all_mergers_data(metrics_data, performance_data_dict)

    # Always exclude redundant metrics (they are averages of _top_k and _bottom_k variants)
    redundant_metrics = {'right_subspace_overlap'}
    exclude_metrics = exclude_metrics | redundant_metrics

    # Filter out excluded metrics
    if exclude_metrics:
        keep_indices = [i for i, name in enumerate(metric_names) if name not in exclude_metrics]
        excluded_count = len(metric_names) - len(keep_indices)
        metric_names = [metric_names[i] for i in keep_indices]
        metrics_array = metrics_array[:, keep_indices]
        print(f"Excluded {excluded_count} metrics: {sorted(exclude_metrics)}")

    print(f"Number of pairs: {len(pair_names)}")
    print(f"Number of metrics: {len(metric_names)}")
    print(f"Number of merge methods: {len(merge_methods)}")
    print()

    # Get list of tasks
    all_tasks = metrics_data['datasets']
    print(f"Total tasks: {len(all_tasks)}")
    print()

    # Run LOTO for each merge method
    all_results = {}

    for method_idx, method in enumerate(merge_methods):
        print("="*70)
        print(f"Reverse Greedy LOTO Cross-Validation for: {method}")
        print("="*70)
        print()

        # Extract performance for this method
        performance = performance_matrix[:, method_idx]

        # Run Reverse Greedy LOTO CV
        results = run_loto_cv_reverse_greedy(
            metrics_array,
            performance,
            pair_names,
            all_tasks,
            metric_names,
            threshold=args.threshold
        )

        all_results[method] = results

        # Save individual method results
        method_output_file = output_dir / f'{method}_loto_results.json'
        with open(method_output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to: {method_output_file}")
        print()

    # Save combined results
    combined_results = {
        'reverse_greedy_threshold': args.threshold,
        'methods': all_results
    }
    combined_output_file = output_dir / 'all_methods_loto_results.json'
    with open(combined_output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    # Print summary
    print("="*70)
    print("SUMMARY: Reverse Greedy (Backward Elimination) LOTO Results")
    print("="*70)
    print()
    print(f"{'Method':<15} {'Train r':<12} {'Val r':<12} {'Val r std':<12} {'N retained':<12}")
    print("-"*70)
    for method in merge_methods:
        train_r = all_results[method]['per_fold_stats']['train_r_mean']
        val_r = all_results[method]['per_fold_stats']['val_r_mean']
        val_r_std = all_results[method]['per_fold_stats']['val_r_std']
        n_retained = all_results[method]['per_fold_stats']['n_retained_mean']
        print(f"{method:<15} {train_r:<12.4f} {val_r:<12.4f} {val_r_std:<12.4f} {n_retained:<12.1f}")
    print("="*70)

    # Print most frequently retained metrics
    print()
    print("="*70)
    print("Most Frequently Retained Metrics (across all folds)")
    print("="*70)

    for method in merge_methods:
        freq = all_results[method]['retention_frequency']
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{method.upper()}:")
        print("-" * 60)
        for metric, f in sorted_freq[:10]:
            if f > 0:
                print(f"  {metric:<45} {f:>6.0%}")

    # Print most frequently removed metrics
    print()
    print("="*70)
    print("Most Frequently Removed Metrics (across all folds)")
    print("="*70)

    for method in merge_methods:
        freq = all_results[method]['retention_frequency']
        sorted_freq = sorted(freq.items(), key=lambda x: x[1])  # Ascending = most removed

        print(f"\n{method.upper()}:")
        print("-" * 60)
        for metric, f in sorted_freq[:10]:
            removal_rate = 1 - f
            if removal_rate > 0:
                print(f"  {metric:<45} {removal_rate:>6.0%}")

    print()
    print("="*70)
    print("Reverse Greedy (Backward Elimination) Complete!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
