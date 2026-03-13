#!/usr/bin/env python3
"""
Leave-One-Task-Out Cross-Validation for L1-Regularized Linear Optimization with MSE objective.

This script performs LOTO CV for linear mergeability prediction:
- Uses Mean Squared Error (MSE) as the optimization objective
- Adds L1 penalty for sparse coefficients
- For each of 20 tasks, train on 19 tasks and validate on 1 held-out task
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


def normalize_metrics_with_stats(metrics_array, min_vals, max_vals):
    """Normalize metrics using pre-computed min/max values (for validation data)."""
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    normalized = (metrics_array - min_vals) / ranges
    normalized = normalized * 2 - 1

    return normalized


def linear_optimization_single_fold_l1_mse(metrics_train, performance_train,
                                            metrics_val, performance_val,
                                            lambda_l1=1.0,
                                            n_iterations=2000, lr=0.01,
                                            patience=100, convergence_threshold=1e-5):
    """
    Optimize linear coefficients for a single fold using MSE loss + L1 penalty.

    Returns:
        coefficients: Optimized coefficients (sparse)
        train_metrics: Dict with train MSE, R2, and Pearson r
        val_metrics: Dict with val MSE, R2, and Pearson r
        n_iters: Number of iterations run
        n_nonzero: Number of non-zero coefficients
    """
    n_metrics = metrics_train.shape[1]

    # Initialize coefficients (smaller initialization for L1)
    coefficients = torch.randn(n_metrics, dtype=torch.float32) * 0.1
    coefficients.requires_grad_(True)
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

        # Loss: MSE + L1 penalty
        mse_loss = torch.mean((predictions - y_train) ** 2)
        l1_penalty = torch.abs(coefficients).sum()
        loss = mse_loss + lambda_l1 * l1_penalty

        # Backward pass
        loss.backward()
        optimizer.step()

        # Early stopping based on training loss
        if loss.item() < best_train_loss - convergence_threshold:
            best_train_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Apply soft thresholding to get true zeros
    final_coefficients = coefficients.detach().numpy()
    threshold = 1e-3
    final_coefficients[np.abs(final_coefficients) < threshold] = 0

    # Final evaluation
    with torch.no_grad():
        coef_tensor = torch.FloatTensor(final_coefficients)
        train_pred = (X_train @ coef_tensor).numpy()
        val_pred = (X_val @ coef_tensor).numpy()

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

    n_nonzero = np.sum(final_coefficients != 0)

    return final_coefficients, train_metrics, val_metrics, iteration + 1, n_nonzero


def run_loto_cv_l1_mse(metrics_array, performance_array, pair_names, all_tasks,
                        metric_names, lambda_l1=1.0, n_iterations=2000, lr=0.01,
                        patience=100, convergence_threshold=1e-5):
    """
    Run Leave-One-Task-Out cross-validation with L1+MSE objective.

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
    fold_coefficients = []
    fold_nonzero_counts = []

    print(f"Running LOTO CV (L1+MSE, lambda={lambda_l1}) with {n_tasks} folds...")
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

        # Optimize coefficients for this fold
        coefficients, train_metrics, val_metrics, n_iters, n_nonzero = linear_optimization_single_fold_l1_mse(
            metrics_train, performance_train,
            metrics_val, performance_val,
            lambda_l1=lambda_l1,
            n_iterations=n_iterations,
            lr=lr,
            patience=patience,
            convergence_threshold=convergence_threshold
        )

        print(f"  Converged in {n_iters} iters: train_r={train_metrics['pearson_r']:.4f}, val_r={val_metrics['pearson_r']:.4f}, nonzero={n_nonzero}/{n_metrics}")

        # Store predictions for aggregate evaluation
        train_preds = metrics_train @ coefficients
        val_preds = metrics_val @ coefficients

        all_train_preds.append(train_preds)
        all_train_targets.append(performance_train)
        all_val_preds.append(val_preds)
        all_val_targets.append(performance_val)
        fold_coefficients.append(coefficients)
        fold_nonzero_counts.append(n_nonzero)

        # Store fold results
        fold_results.append({
            'fold': fold_idx,
            'held_out_task': held_out_task,
            'n_train_pairs': len(train_indices),
            'n_val_pairs': len(val_indices),
            'train_mse': float(train_metrics['mse']),
            'train_r2': float(train_metrics['r2']),
            'train_r': float(train_metrics['pearson_r']),
            'val_mse': float(val_metrics['mse']),
            'val_r2': float(val_metrics['r2']),
            'val_r': float(val_metrics['pearson_r']),
            'n_iterations': int(n_iters),
            'n_nonzero_coefficients': int(n_nonzero),
            'coefficients': {name: float(coef) for name, coef in zip(metric_names, coefficients)}
        })

    # Aggregate results
    print()
    print("="*70)
    print("Aggregate Results (L1+MSE)")
    print("="*70)

    # Concatenate all predictions
    all_train_preds = np.concatenate(all_train_preds)
    all_train_targets = np.concatenate(all_train_targets)
    all_val_preds = np.concatenate(all_val_preds)
    all_val_targets = np.concatenate(all_val_targets)

    # Compute aggregate metrics
    aggregate_train_r, aggregate_train_p = pearsonr(all_train_preds, all_train_targets)
    aggregate_val_r, aggregate_val_p = pearsonr(all_val_preds, all_val_targets)

    print(f"Aggregate Training: r={aggregate_train_r:.4f}, p={aggregate_train_p:.2e}")
    print(f"Aggregate Validation: r={aggregate_val_r:.4f}, p={aggregate_val_p:.2e}")

    # Per-fold statistics
    fold_train_r = [f['train_r'] for f in fold_results]
    fold_val_r = [f['val_r'] for f in fold_results]

    print(f"Per-fold: Train r={np.mean(fold_train_r):.4f}±{np.std(fold_train_r):.4f}")
    print(f"Per-fold: Val r={np.mean(fold_val_r):.4f}±{np.std(fold_val_r):.4f}")
    print(f"Per-fold: Nonzero coeffs={np.mean(fold_nonzero_counts):.1f}±{np.std(fold_nonzero_counts):.1f}")

    # Average coefficients across folds
    avg_coefficients = np.mean(fold_coefficients, axis=0)
    std_coefficients = np.std(fold_coefficients, axis=0)
    nonzero_frequency = np.mean([c != 0 for c in fold_coefficients], axis=0)

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
            'n_nonzero_mean': float(np.mean(fold_nonzero_counts)),
            'n_nonzero_std': float(np.std(fold_nonzero_counts))
        },
        'average_coefficients': {name: float(coef) for name, coef in zip(metric_names, avg_coefficients)},
        'coefficient_std': {name: float(std) for name, std in zip(metric_names, std_coefficients)},
        'nonzero_frequency': {name: float(freq) for name, freq in zip(metric_names, nonzero_frequency)},
        'fold_results': fold_results,
        'optimization_params': {
            'objective': 'L1+MSE',
            'lambda_l1': lambda_l1,
            'n_iterations': n_iterations,
            'learning_rate': lr,
            'patience': patience,
            'convergence_threshold': convergence_threshold
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='L1+MSE LOTO Cross-Validation')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='L1 regularization strength')
    parser.add_argument('--model', type=str, default='ViT-B-16', choices=['ViT-B-16', 'ViT-B-32'],
                        help='Model architecture to use')
    args = parser.parse_args()

    lambda_l1 = args.lambda_l1
    model = args.model

    # Configuration - paths based on model
    base_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results')
    metrics_path = base_path / 'mergeability' / model / 'pairwise_metrics_N20.json'
    results_base_path = base_path / model

    # Output directory - include model name for non-default models
    if model == 'ViT-B-16':
        output_dir = base_path / 'metric_linear_optimization_v2' / f'l1_loto_cv_mse_lambda{lambda_l1}'
    else:
        output_dir = base_path / 'metric_linear_optimization_v2' / model.lower() / f'l1_loto_cv_mse_lambda{lambda_l1}'
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_methods = ['weight_avg', 'arithmetic', 'tsv', 'ties', 'dare']

    print("="*70)
    print(f"L1+MSE Linear Optimization with LOTO CV (lambda={lambda_l1})")
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

    print(f"Number of pairs: {len(pair_names)}")
    print(f"Number of metrics: {len(metric_names)}")
    print(f"Number of merge methods: {len(merge_methods)}")
    print()

    print("Normalization will be done per-fold (train stats only, no leakage)")
    print()

    # Get list of tasks
    all_tasks = metrics_data['datasets']
    print(f"Total tasks: {len(all_tasks)}")
    print()

    # Run LOTO for each merge method
    all_results = {}

    for method_idx, method in enumerate(merge_methods):
        print("="*70)
        print(f"LOTO Cross-Validation (L1+MSE) for: {method}")
        print("="*70)
        print()

        # Extract performance for this method
        performance = performance_matrix[:, method_idx]

        # Run LOTO CV with L1+MSE
        results = run_loto_cv_l1_mse(
            metrics_array,
            performance,
            pair_names,
            all_tasks,
            metric_names,
            lambda_l1=lambda_l1,
            n_iterations=2000,
            lr=0.01,
            patience=100,
            convergence_threshold=1e-5
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
        'objective': 'L1+MSE',
        'lambda_l1': lambda_l1,
        'methods': all_results
    }
    combined_output_file = output_dir / 'all_methods_loto_results.json'
    with open(combined_output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print("="*70)
    print("SUMMARY: L1+MSE LOTO Cross-Validation Results")
    print("="*70)
    print()
    print(f"{'Method':<15} {'Train r':<12} {'Val r':<12} {'Val r std':<12} {'Nonzero':<12}")
    print("-"*70)
    for method in merge_methods:
        train_r = all_results[method]['per_fold_stats']['train_r_mean']
        val_r = all_results[method]['per_fold_stats']['val_r_mean']
        val_r_std = all_results[method]['per_fold_stats']['val_r_std']
        n_nonzero = all_results[method]['per_fold_stats']['n_nonzero_mean']
        print(f"{method:<15} {train_r:<12.4f} {val_r:<12.4f} {val_r_std:<12.4f} {n_nonzero:<12.1f}")
    print("="*70)

    print()
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
