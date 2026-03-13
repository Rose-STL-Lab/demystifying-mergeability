#!/usr/bin/env python3
"""
Leave-One-Task-Out Cross-Validation for Linear Optimization using MSE objective.

This script performs LOTO CV for linear mergeability prediction:
- Uses Mean Squared Error (MSE) as the optimization objective instead of Pearson correlation
- For each of 20 tasks, train on 19 tasks and validate on 1 held-out task
- Aggregate results across all folds
- Provides robust coefficient estimates and performance metrics
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_merging.data_loader import load_json, extract_all_mergers_data


def compute_mse(y_true, y_pred):
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def normalize_metrics(metrics_array):
    """Normalize metrics to [-1, 1] range using min-max normalization."""
    min_vals = metrics_array.min(axis=0)
    max_vals = metrics_array.max(axis=0)

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    normalized = (metrics_array - min_vals) / ranges
    normalized = normalized * 2 - 1

    return normalized, min_vals, max_vals


def linear_optimization_single_fold_mse(metrics_train, performance_train,
                                        metrics_val, performance_val,
                                        n_iterations=1000, lr=0.01,
                                        patience=50, convergence_threshold=1e-4):
    """
    Optimize linear coefficients for a single fold using MSE loss.

    Returns:
        coefficients: Optimized coefficients
        train_metrics: Dict with train MSE, R2, and Pearson r
        val_metrics: Dict with val MSE, R2, and Pearson r
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

    best_val_loss = float('inf')
    patience_counter = 0

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Forward pass
        predictions = X_train @ coefficients

        # Loss: Mean Squared Error
        loss = torch.mean((predictions - y_train) ** 2)

        # Backward pass
        loss.backward()

        # Constraint: coefficients sum to 1
        with torch.no_grad():
            coefficients.data = coefficients.data / (coefficients.data.sum() + 1e-8)

        optimizer.step()

        # Validation
        with torch.no_grad():
            val_predictions = X_val @ coefficients
            val_loss = torch.mean((val_predictions - y_val) ** 2).item()

        # Early stopping
        if val_loss < best_val_loss - convergence_threshold:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Final evaluation
    with torch.no_grad():
        train_pred = (X_train @ coefficients).numpy()
        val_pred = (X_val @ coefficients).numpy()

    # Compute metrics
    train_mse = compute_mse(performance_train, train_pred)
    train_r2 = compute_r2(performance_train, train_pred)
    train_r, train_p = pearsonr(train_pred, performance_train)

    val_mse = compute_mse(performance_val, val_pred)
    val_r2 = compute_r2(performance_val, val_pred)
    val_r, val_p = pearsonr(val_pred, performance_val)

    train_metrics = {
        'mse': train_mse,
        'r2': train_r2,
        'pearson_r': train_r,
        'pearson_p': train_p
    }

    val_metrics = {
        'mse': val_mse,
        'r2': val_r2,
        'pearson_r': val_r,
        'pearson_p': val_p
    }

    return coefficients.detach().numpy(), train_metrics, val_metrics, iteration + 1


def run_loto_cv_mse(metrics_array, performance_array, pair_names, all_tasks,
                    metric_names, n_iterations=1000, lr=0.01,
                    patience=50, convergence_threshold=1e-4):
    """
    Run Leave-One-Task-Out cross-validation with MSE objective.

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

    print(f"Running LOTO CV (MSE objective) with {n_tasks} folds...")
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

        # Extract data for this fold
        metrics_train = metrics_array[train_indices]
        performance_train = performance_array[train_indices]
        metrics_val = metrics_array[val_indices]
        performance_val = performance_array[val_indices]

        # Optimize coefficients for this fold
        coefficients, train_metrics, val_metrics, n_iters = linear_optimization_single_fold_mse(
            metrics_train, performance_train,
            metrics_val, performance_val,
            n_iterations=n_iterations,
            lr=lr,
            patience=patience,
            convergence_threshold=convergence_threshold
        )

        print(f"  Converged in {n_iters} iterations:")
        print(f"    Train - MSE: {train_metrics['mse']:.6f}, R²: {train_metrics['r2']:.4f}, r: {train_metrics['pearson_r']:.4f}")
        print(f"    Val   - MSE: {val_metrics['mse']:.6f}, R²: {val_metrics['r2']:.4f}, r: {val_metrics['pearson_r']:.4f}")

        # Store predictions for aggregate evaluation
        train_preds = metrics_train @ coefficients
        val_preds = metrics_val @ coefficients

        all_train_preds.append(train_preds)
        all_train_targets.append(performance_train)
        all_val_preds.append(val_preds)
        all_val_targets.append(performance_val)
        fold_coefficients.append(coefficients)

        # Store fold results
        fold_results.append({
            'fold': fold_idx,
            'held_out_task': held_out_task,
            'n_train_pairs': len(train_indices),
            'n_val_pairs': len(val_indices),
            'train_mse': float(train_metrics['mse']),
            'train_r2': float(train_metrics['r2']),
            'train_pearson_r': float(train_metrics['pearson_r']),
            'val_mse': float(val_metrics['mse']),
            'val_r2': float(val_metrics['r2']),
            'val_pearson_r': float(val_metrics['pearson_r']),
            'n_iterations': int(n_iters),
            'coefficients': {name: float(coef) for name, coef in zip(metric_names, coefficients)}
        })

    # Aggregate results
    print()
    print("="*70)
    print("Aggregate Results (MSE Objective)")
    print("="*70)

    # Concatenate all predictions
    all_train_preds = np.concatenate(all_train_preds)
    all_train_targets = np.concatenate(all_train_targets)
    all_val_preds = np.concatenate(all_val_preds)
    all_val_targets = np.concatenate(all_val_targets)

    # Compute aggregate metrics
    aggregate_train_mse = compute_mse(all_train_targets, all_train_preds)
    aggregate_train_r2 = compute_r2(all_train_targets, all_train_preds)
    aggregate_train_r, aggregate_train_p = pearsonr(all_train_preds, all_train_targets)

    aggregate_val_mse = compute_mse(all_val_targets, all_val_preds)
    aggregate_val_r2 = compute_r2(all_val_targets, all_val_preds)
    aggregate_val_r, aggregate_val_p = pearsonr(all_val_preds, all_val_targets)

    print(f"Aggregate Training:")
    print(f"  MSE: {aggregate_train_mse:.6f}, R²: {aggregate_train_r2:.4f}, r: {aggregate_train_r:.4f}, p: {aggregate_train_p:.2e}")
    print(f"Aggregate Validation:")
    print(f"  MSE: {aggregate_val_mse:.6f}, R²: {aggregate_val_r2:.4f}, r: {aggregate_val_r:.4f}, p: {aggregate_val_p:.2e}")

    # Per-fold statistics
    fold_train_mse = [f['train_mse'] for f in fold_results]
    fold_val_mse = [f['val_mse'] for f in fold_results]
    fold_train_r = [f['train_pearson_r'] for f in fold_results]
    fold_val_r = [f['val_pearson_r'] for f in fold_results]

    print(f"\nPer-fold Statistics:")
    print(f"  Train MSE: {np.mean(fold_train_mse):.6f}±{np.std(fold_train_mse):.6f}")
    print(f"  Val MSE:   {np.mean(fold_val_mse):.6f}±{np.std(fold_val_mse):.6f}")
    print(f"  Train r:   {np.mean(fold_train_r):.4f}±{np.std(fold_train_r):.4f}")
    print(f"  Val r:     {np.mean(fold_val_r):.4f}±{np.std(fold_val_r):.4f}")

    # Average coefficients across folds
    avg_coefficients = np.mean(fold_coefficients, axis=0)
    std_coefficients = np.std(fold_coefficients, axis=0)

    results = {
        'aggregate_metrics': {
            'train_mse': float(aggregate_train_mse),
            'train_r2': float(aggregate_train_r2),
            'train_pearson_r': float(aggregate_train_r),
            'train_pearson_p': float(aggregate_train_p),
            'val_mse': float(aggregate_val_mse),
            'val_r2': float(aggregate_val_r2),
            'val_pearson_r': float(aggregate_val_r),
            'val_pearson_p': float(aggregate_val_p)
        },
        'per_fold_stats': {
            'train_mse_mean': float(np.mean(fold_train_mse)),
            'train_mse_std': float(np.std(fold_train_mse)),
            'val_mse_mean': float(np.mean(fold_val_mse)),
            'val_mse_std': float(np.std(fold_val_mse)),
            'train_r_mean': float(np.mean(fold_train_r)),
            'train_r_std': float(np.std(fold_train_r)),
            'val_r_mean': float(np.mean(fold_val_r)),
            'val_r_std': float(np.std(fold_val_r))
        },
        'average_coefficients': {name: float(coef) for name, coef in zip(metric_names, avg_coefficients)},
        'coefficient_std': {name: float(std) for name, std in zip(metric_names, std_coefficients)},
        'fold_results': fold_results,
        'optimization_params': {
            'objective': 'MSE',
            'n_iterations': n_iterations,
            'learning_rate': lr,
            'patience': patience,
            'convergence_threshold': convergence_threshold
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='LOTO Cross-Validation with MSE Objective')
    parser.add_argument('--model', type=str, default='ViT-B-16', choices=['ViT-B-16', 'ViT-B-32'],
                        help='Model architecture to use')
    args = parser.parse_args()

    model = args.model

    # Configuration - paths based on model
    base_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results')
    metrics_path = base_path / 'mergeability' / model / 'pairwise_metrics_N20.json'
    results_base_path = base_path / model

    # Output directory - always use model-specific subfolder
    output_dir = base_path / 'metric_linear_optimization_v2' / model.lower() / 'loto_cv_mse'
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_methods = ['weight_avg', 'arithmetic', 'tsv', 'ties', 'dare']

    print("="*70)
    print("Linear Optimization with LOTO Cross-Validation (MSE Objective)")
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
    keep_indices = [i for i, name in enumerate(metric_names) if name not in redundant_metrics]
    if len(keep_indices) < len(metric_names):
        excluded_count = len(metric_names) - len(keep_indices)
        metric_names = [metric_names[i] for i in keep_indices]
        metrics_array = metrics_array[:, keep_indices]
        print(f"Excluded {excluded_count} redundant metrics: {sorted(redundant_metrics)}")

    print(f"Number of pairs: {len(pair_names)}")
    print(f"Number of metrics: {len(metric_names)}")
    print(f"Number of merge methods: {len(merge_methods)}")
    print()

    # Normalize metrics
    print("Normalizing metrics...")
    metrics_normalized, _, _ = normalize_metrics(metrics_array)
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
        print(f"LOTO Cross-Validation (MSE) for: {method}")
        print("="*70)
        print()

        # Extract performance for this method
        performance = performance_matrix[:, method_idx]

        # Run LOTO CV with MSE
        results = run_loto_cv_mse(
            metrics_normalized,
            performance,
            pair_names,
            all_tasks,
            metric_names,
            n_iterations=1000,
            lr=0.01,
            patience=50,
            convergence_threshold=1e-6  # Smaller threshold for MSE
        )

        all_results[method] = results

        # Save individual method results
        method_output_file = output_dir / f'{method}_loto_mse_results.json'
        with open(method_output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to: {method_output_file}")
        print()

    # Save combined results (matching L1 LOTO format)
    combined_results = {
        'objective': 'MSE',
        'methods': all_results
    }
    combined_output_file = output_dir / 'all_methods_loto_results.json'
    with open(combined_output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print("="*70)
    print("SUMMARY: LOTO Cross-Validation Results (MSE Objective)")
    print("="*70)
    print()
    print(f"{'Method':<15} {'Train MSE':<12} {'Val MSE':<12} {'Train r':<12} {'Val r':<12}")
    print("-"*70)
    for method in merge_methods:
        train_mse = all_results[method]['aggregate_metrics']['train_mse']
        val_mse = all_results[method]['aggregate_metrics']['val_mse']
        train_r = all_results[method]['aggregate_metrics']['train_pearson_r']
        val_r = all_results[method]['aggregate_metrics']['val_pearson_r']
        print(f"{method:<15} {train_mse:<12.6f} {val_mse:<12.6f} {train_r:<12.4f} {val_r:<12.4f}")
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
    print("LOTO Cross-Validation Complete (MSE Objective)!")
    print("="*70)


if __name__ == "__main__":
    main()
