#!/usr/bin/env python3
"""
Leave-One-Task-Out Cross-Validation for Linear Optimization.

This script performs LOTO CV for linear mergeability prediction:
- For each of 20 tasks, train on 19 tasks and validate on 1 held-out task
- Aggregate results across all folds
- Provides robust coefficient estimates and performance metrics

Usage:
    python scripts/linear_optimization_loto.py
    python scripts/linear_optimization_loto.py --exclude_metrics effective_rank stable_rank
    python scripts/linear_optimization_loto.py --exclude_group eff_rank --output_suffix no_eff_rank
    python scripts/linear_optimization_loto.py --exclude_group grad_based --output_suffix no_grad_based
    python scripts/linear_optimization_loto.py --zero_mean
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_merging.data_loader import load_json, extract_all_mergers_data

# Define metric groups for ablation
METRIC_GROUPS = {
    'eff_rank': [
        'effective_rank',
        'effective_rank_mergeability_score',
        'layerwise_effective_rank',
        'layerwise_effective_rank_mergeability_score',
        'stable_rank',
        'spectral_gap',
        'singular_value_ratio',
    ],
    'grad_based': [
        'encoder_gradient_cosine_similarity',
        'encoder_gradient_l2_distance',
        'encoder_gradient_dot_product',
        'input_gradient_cosine_similarity',
        'input_gradient_l2_distance',
        'input_gradient_dot_product',
    ],
    'activation': [
        'activation_l2_distance',
        'activation_cosine_similarity',
        'activation_magnitude_ratio',
        'activation_dot_product',
    ],
    'subspace': [
        'right_subspace_overlap',
        'right_subspace_overlap_top_k',
        'right_subspace_overlap_bottom_k',
        'subspace_overlap',
        'singular_value_overlap',
        'interaction_matrix_overlap_top_k',
        'interaction_matrix_overlap_bottom_k',
    ],
    'task_vector': [
        'task_vector_cosine_similarity',
        'task_vector_l2_distance',
        'task_vector_dot_product',
        'task_vector_magnitude_ratio',
        'weight_space_angle',
    ],
}


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


def linear_optimization_single_fold(metrics_train, performance_train,
                                     metrics_val, performance_val,
                                     n_iterations=1000, lr=0.01,
                                     patience=50, convergence_threshold=1e-4):
    """
    Optimize linear coefficients for a single fold.

    Returns:
        coefficients: Optimized coefficients
        train_r: Training Pearson r
        val_r: Validation Pearson r
        n_iters: Number of iterations run
    """
    n_metrics = metrics_train.shape[1]

    # Initialize coefficients
    coefficients = torch.randn(n_metrics, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([coefficients], lr=lr)

    # Convert to tensors
    X_train = torch.FloatTensor(metrics_train)
    y_train = torch.FloatTensor(performance_train)
    X_val = torch.FloatTensor(metrics_val)
    y_val = torch.FloatTensor(performance_val)

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
        loss = -correlation  # Maximize correlation = minimize negative correlation

        # Backward pass
        loss.backward()

        # Constraint: coefficients sum to 1
        with torch.no_grad():
            coefficients.data = coefficients.data / (coefficients.data.sum() + 1e-8)

        optimizer.step()

        # Validation
        with torch.no_grad():
            val_predictions = X_val @ coefficients
            val_pred_mean = val_predictions.mean()
            val_target_mean = y_val.mean()
            val_pred_centered = val_predictions - val_pred_mean
            val_target_centered = y_val - val_target_mean
            val_numerator = (val_pred_centered * val_target_centered).sum()
            val_denominator = torch.sqrt((val_pred_centered ** 2).sum() * (val_target_centered ** 2).sum())
            val_correlation = val_numerator / (val_denominator + 1e-8)
            val_loss = -val_correlation.item()

        # Early stopping
        if loss < best_train_loss - convergence_threshold:
            best_train_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Final evaluation
    with torch.no_grad():
        train_pred = (X_train @ coefficients).numpy()
        val_pred = (X_val @ coefficients).numpy()

    train_r, _ = pearsonr(train_pred, performance_train)
    val_r, _ = pearsonr(val_pred, performance_val)

    return coefficients.detach().numpy(), train_r, val_r, iteration + 1


def run_loto_cv(metrics_array, performance_array, pair_names, all_tasks,
                metric_names, n_iterations=1000, lr=0.01,
                patience=50, convergence_threshold=1e-4,
                zero_mean=False):
    """
    Run Leave-One-Task-Out cross-validation.

    Args:
        zero_mean: If True, subtract the training mean from target variable
                   before optimization. This makes the model predict deviation
                   from the mean rather than absolute performance.

    Returns:
        results: Dictionary with fold results and aggregate metrics
    """
    n_tasks = len(all_tasks)

    # Storage for results
    fold_results = []
    all_train_preds = []
    all_train_targets = []
    all_val_preds = []
    all_val_targets = []
    fold_coefficients = []

    print(f"Running LOTO CV with {n_tasks} folds...")
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

            # Training: both tasks in train_tasks
            if task1 in train_tasks and task2 in train_tasks:
                train_indices.append(i)
            # Validation: at least one task is held-out
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

        # Zero-mean normalization of target variable (if enabled)
        # Use training mean for both train and val to avoid leakage
        train_target_mean = 0.0
        if zero_mean:
            train_target_mean = np.mean(performance_train)
            performance_train = performance_train - train_target_mean
            performance_val = performance_val - train_target_mean

        # Normalize using ONLY training data statistics (no leakage)
        metrics_train, min_vals, max_vals = normalize_metrics(metrics_train_raw)
        metrics_val = normalize_metrics_with_stats(metrics_val_raw, min_vals, max_vals)

        # Optimize coefficients for this fold
        coefficients, train_r, val_r, n_iters = linear_optimization_single_fold(
            metrics_train, performance_train,
            metrics_val, performance_val,
            n_iterations=n_iterations,
            lr=lr,
            patience=patience,
            convergence_threshold=convergence_threshold
        )

        print(f"  Converged in {n_iters} iterations: train_r={train_r:.4f}, val_r={val_r:.4f}")

        # Store predictions for aggregate evaluation
        train_preds = metrics_train @ coefficients
        val_preds = metrics_val @ coefficients

        all_train_preds.append(train_preds)
        all_train_targets.append(performance_train)
        all_val_preds.append(val_preds)
        all_val_targets.append(performance_val)
        fold_coefficients.append(coefficients)

        # Store fold results
        fold_result = {
            'fold': fold_idx,
            'held_out_task': held_out_task,
            'n_train_pairs': len(train_indices),
            'n_val_pairs': len(val_indices),
            'train_r': float(train_r),
            'val_r': float(val_r),
            'n_iterations': int(n_iters),
            'coefficients': {name: float(coef) for name, coef in zip(metric_names, coefficients)}
        }
        if zero_mean:
            fold_result['train_target_mean'] = float(train_target_mean)
        fold_results.append(fold_result)

    # Aggregate results
    print()
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

    print(f"Per-fold: Train r={np.mean(fold_train_r):.4f}±{np.std(fold_train_r):.4f}")
    print(f"Per-fold: Val r={np.mean(fold_val_r):.4f}±{np.std(fold_val_r):.4f}")

    # Average coefficients across folds
    avg_coefficients = np.mean(fold_coefficients, axis=0)
    std_coefficients = np.std(fold_coefficients, axis=0)

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
            'val_r_std': float(np.std(fold_val_r))
        },
        'average_coefficients': {name: float(coef) for name, coef in zip(metric_names, avg_coefficients)},
        'coefficient_std': {name: float(std) for name, std in zip(metric_names, std_coefficients)},
        'fold_results': fold_results,
        'optimization_params': {
            'n_iterations': n_iterations,
            'learning_rate': lr,
            'patience': patience,
            'convergence_threshold': convergence_threshold
        }
    }

    return results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='LOTO Cross-Validation for Linear Optimization')
    parser.add_argument('--exclude_metrics', type=str, nargs='+', default=[],
                        help='List of metric names to exclude')
    parser.add_argument('--exclude_group', type=str, choices=list(METRIC_GROUPS.keys()),
                        help='Exclude a predefined group of metrics (eff_rank, grad_based, activation, subspace, task_vector)')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='Suffix to add to output directory name')
    parser.add_argument('--zero_mean', action='store_true',
                        help='Zero-mean normalize the target variable per method before optimization. '
                             'This subtracts the training mean from both train and val targets, '
                             'making the model predict deviation from mean rather than absolute performance.')
    args = parser.parse_args()

    # Determine which metrics to exclude
    exclude_metrics = set(args.exclude_metrics)
    if args.exclude_group:
        exclude_metrics.update(METRIC_GROUPS[args.exclude_group])

    # Configuration
    metrics_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/mergeability/ViT-B-16/pairwise_metrics_N20.json')
    results_base_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/ViT-B-16')

    # Output directory with optional suffix
    if args.output_suffix:
        output_dir = Path(f'/home/ubuntu/thesis/MM/Mergeability-Bench/results/metric_linear_optimization_v2/loto_cv_{args.output_suffix}')
    elif args.zero_mean:
        output_dir = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/metric_linear_optimization_v2/loto_cv_no_leakage_zero_mean')
    else:
        output_dir = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/metric_linear_optimization_v2/loto_cv_no_leakage')
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_methods = ['weight_avg', 'arithmetic', 'tsv', 'ties', 'dare']

    print("="*70)
    print("Linear Optimization with LOTO Cross-Validation")
    print("="*70)
    print()

    if exclude_metrics:
        print(f"EXCLUDING METRICS: {sorted(exclude_metrics)}")
        print()

    if args.zero_mean:
        print("ZERO-MEAN NORMALIZATION: Enabled")
        print("  Target variable will be centered using training mean per fold")
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

    # Filter out excluded metrics
    if exclude_metrics:
        keep_indices = [i for i, name in enumerate(metric_names) if name not in exclude_metrics]
        excluded_count = len(metric_names) - len(keep_indices)
        metric_names = [metric_names[i] for i in keep_indices]
        metrics_array = metrics_array[:, keep_indices]
        print(f"Excluded {excluded_count} metrics, {len(metric_names)} remaining")
        print()

    print(f"Number of pairs: {len(pair_names)}")
    print(f"Number of metrics: {len(metric_names)}")
    print(f"Number of merge methods: {len(merge_methods)}")
    print()

    # NOTE: Normalization is now done per-fold inside run_loto_cv to avoid leakage
    print("Normalization will be done per-fold (train stats only, no leakage)")
    print()

    # Get list of tasks
    all_tasks = metrics_data['datasets']
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Tasks: {all_tasks}")
    print()

    # Run LOTO for each merge method
    all_results = {}

    for method_idx, method in enumerate(merge_methods):
        print("="*70)
        print(f"LOTO Cross-Validation for: {method}")
        print("="*70)
        print()

        # Extract performance for this method
        performance = performance_matrix[:, method_idx]

        # Run LOTO CV (uses raw metrics, normalization done per-fold)
        results = run_loto_cv(
            metrics_array,
            performance,
            pair_names,
            all_tasks,
            metric_names,
            n_iterations=1000,
            lr=0.01,
            patience=50,
            convergence_threshold=1e-4,
            zero_mean=args.zero_mean
        )

        all_results[method] = results

        # Save individual method results
        method_output_file = output_dir / f'{method}_loto_results.json'
        with open(method_output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to: {method_output_file}")
        print()

    # Save combined results with metadata
    combined_results = {
        'excluded_metrics': sorted(exclude_metrics) if exclude_metrics else [],
        'zero_mean_normalization': args.zero_mean,
        'n_metrics_used': len(metric_names),
        'metrics_used': metric_names,
        'methods': all_results
    }
    combined_output_file = output_dir / 'all_methods_loto_results.json'
    with open(combined_output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print("="*70)
    print("SUMMARY: LOTO Cross-Validation Results")
    if exclude_metrics:
        print(f"(Excluded: {args.exclude_group or 'custom'} - {len(exclude_metrics)} metrics)")
    if args.zero_mean:
        print("(Zero-mean normalization enabled)")
    print("="*70)
    print()
    print(f"{'Method':<15} {'Train r':<12} {'Val r':<12} {'Val r std':<12}")
    print("-"*70)
    for method in merge_methods:
        train_r = all_results[method]['aggregate_metrics']['train_r']
        val_r = all_results[method]['aggregate_metrics']['val_r']
        val_r_std = all_results[method]['per_fold_stats']['val_r_std']
        print(f"{method:<15} {train_r:<12.4f} {val_r:<12.4f} {val_r_std:<12.4f}")
    print("="*70)

    print()
    print(f"All results saved to: {output_dir}")

    # Create coefficient comparison
    print()
    print("="*70)
    print("Average Coefficients Across Folds (Top 10 per method)")
    print("="*70)

    for method in merge_methods:
        avg_coefs = all_results[method]['average_coefficients']
        std_coefs = all_results[method]['coefficient_std']

        # Sort by absolute value
        sorted_items = sorted(avg_coefs.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"\n{method.upper()}:")
        print("-" * 60)
        for i, (metric, coef) in enumerate(sorted_items[:10], 1):
            std = std_coefs[metric]
            print(f"{i:2d}. {metric:45s} {coef:+7.4f} (±{std:.4f})")

    print()
    print("="*70)
    print("LOTO Cross-Validation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
