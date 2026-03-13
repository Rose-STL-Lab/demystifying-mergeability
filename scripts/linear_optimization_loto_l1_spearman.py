#!/usr/bin/env python3
"""
Leave-One-Task-Out Cross-Validation for L1-Regularized Linear Optimization with Spearman correlation.

This script performs LOTO CV for linear mergeability prediction:
- Uses Spearman correlation (via differentiable soft ranking) as optimization objective
- L1 penalty encourages sparse coefficients
- For each of 20 tasks, train on 19 tasks and validate on 1 held-out task
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


def soft_rank(x, temperature=1.0):
    """
    Compute differentiable soft ranks using pairwise comparisons.

    For each element x_i, its soft rank is approximately:
    rank_i = 1 + Σ_j sigmoid((x_j - x_i) / temperature)

    Lower temperature = sharper ranking (closer to true ranks)
    Higher temperature = smoother gradients
    """
    n = x.shape[0]
    # Pairwise differences: diff[i,j] = x[j] - x[i]
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # shape: (n, n)
    # Soft comparison: how many elements are smaller than x_i
    soft_comparisons = torch.sigmoid(diff / temperature)
    # Sum over j to get soft rank (add 1 for 1-based ranking)
    ranks = soft_comparisons.sum(dim=0) + 1
    return ranks


def differentiable_spearman(pred, target, temperature=1.0):
    """
    Compute differentiable Spearman correlation using soft ranking.

    Spearman correlation = Pearson correlation of ranks
    """
    # Compute soft ranks
    pred_ranks = soft_rank(pred, temperature)
    target_ranks = soft_rank(target, temperature)

    # Compute Pearson correlation of ranks
    pred_centered = pred_ranks - pred_ranks.mean()
    target_centered = target_ranks - target_ranks.mean()

    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())

    correlation = numerator / (denominator + 1e-8)
    return correlation


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


def linear_optimization_single_fold_l1_spearman(metrics_train, performance_train,
                                                 metrics_val, performance_val,
                                                 lambda_l1=1.0, temperature=1.0,
                                                 n_iterations=2000, lr=0.01,
                                                 patience=100, convergence_threshold=1e-4):
    """
    Optimize linear coefficients for a single fold with L1 regularization and Spearman objective.

    Returns:
        coefficients: Optimized coefficients (sparse)
        train_r: Training Pearson r (for comparison)
        train_rho: Training Spearman rho
        val_r: Validation Pearson r
        val_rho: Validation Spearman rho
        n_iters: Number of iterations run
        n_nonzero: Number of non-zero coefficients
    """
    n_metrics = metrics_train.shape[1]

    # Initialize coefficients
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

        # Compute differentiable Spearman correlation
        spearman_corr = differentiable_spearman(predictions, y_train, temperature)

        # Loss: negative Spearman + L1 penalty
        l1_penalty = torch.abs(coefficients).sum()
        loss = -spearman_corr + lambda_l1 * l1_penalty

        # Backward pass
        loss.backward()
        optimizer.step()

        # Early stopping
        if loss.item() < best_train_loss - convergence_threshold:
            best_train_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Apply soft thresholding
    final_coefficients = coefficients.detach().numpy()
    threshold = 1e-3
    final_coefficients[np.abs(final_coefficients) < threshold] = 0

    # Final evaluation (using scipy for exact correlations)
    with torch.no_grad():
        coef_tensor = torch.FloatTensor(final_coefficients)
        train_pred = (X_train @ coef_tensor).numpy()
        val_pred = (X_val @ coef_tensor).numpy()

    train_r, _ = pearsonr(train_pred, performance_train)
    train_rho, _ = spearmanr(train_pred, performance_train)
    val_r, _ = pearsonr(val_pred, performance_val)
    val_rho, _ = spearmanr(val_pred, performance_val)

    n_nonzero = np.sum(final_coefficients != 0)

    return final_coefficients, train_r, train_rho, val_r, val_rho, iteration + 1, n_nonzero


def run_loto_cv_l1_spearman(metrics_array, performance_array, pair_names, all_tasks,
                             metric_names, lambda_l1=1.0, temperature=1.0,
                             n_iterations=2000, lr=0.01,
                             patience=100, convergence_threshold=1e-4):
    """
    Run Leave-One-Task-Out cross-validation with L1 + Spearman objective.
    """
    n_tasks = len(all_tasks)
    n_metrics = len(metric_names)

    fold_results = []
    all_val_preds = []
    all_val_targets = []
    fold_coefficients = []
    fold_nonzero_counts = []

    print(f"Running LOTO CV (L1+Spearman, lambda={lambda_l1}, temp={temperature}) with {n_tasks} folds...")
    print()

    for fold_idx, held_out_task in enumerate(all_tasks):
        print(f"Fold {fold_idx+1}/{n_tasks}: Held-out task = {held_out_task}")

        train_tasks = [t for t in all_tasks if t != held_out_task]

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

        # Extract raw data
        metrics_train_raw = metrics_array[train_indices]
        performance_train = performance_array[train_indices]
        metrics_val_raw = metrics_array[val_indices]
        performance_val = performance_array[val_indices]

        # Normalize per-fold (no leakage)
        metrics_train, min_vals, max_vals = normalize_metrics(metrics_train_raw)
        metrics_val = normalize_metrics_with_stats(metrics_val_raw, min_vals, max_vals)

        # Optimize
        coefficients, train_r, train_rho, val_r, val_rho, n_iters, n_nonzero = \
            linear_optimization_single_fold_l1_spearman(
                metrics_train, performance_train,
                metrics_val, performance_val,
                lambda_l1=lambda_l1,
                temperature=temperature,
                n_iterations=n_iterations,
                lr=lr,
                patience=patience,
                convergence_threshold=convergence_threshold
            )

        print(f"  Converged in {n_iters} iters: train_rho={train_rho:.4f}, val_rho={val_rho:.4f}, nonzero={n_nonzero}/{n_metrics}")

        val_preds = metrics_val @ coefficients
        all_val_preds.append(val_preds)
        all_val_targets.append(performance_val)
        fold_coefficients.append(coefficients)
        fold_nonzero_counts.append(n_nonzero)

        fold_results.append({
            'fold': fold_idx,
            'held_out_task': held_out_task,
            'n_train_pairs': len(train_indices),
            'n_val_pairs': len(val_indices),
            'train_r': float(train_r),
            'train_rho': float(train_rho),
            'val_r': float(val_r),
            'val_rho': float(val_rho),
            'n_iterations': int(n_iters),
            'n_nonzero_coefficients': int(n_nonzero),
            'coefficients': {name: float(coef) for name, coef in zip(metric_names, coefficients)}
        })

    # Aggregate results
    print()
    print("="*70)
    print("Aggregate Results (L1+Spearman)")
    print("="*70)

    all_val_preds = np.concatenate(all_val_preds)
    all_val_targets = np.concatenate(all_val_targets)

    aggregate_val_r, aggregate_val_p = pearsonr(all_val_preds, all_val_targets)
    aggregate_val_rho, aggregate_val_rho_p = spearmanr(all_val_preds, all_val_targets)

    print(f"Aggregate Validation: Pearson r={aggregate_val_r:.4f}, Spearman ρ={aggregate_val_rho:.4f}")

    fold_val_r = [f['val_r'] for f in fold_results]
    fold_val_rho = [f['val_rho'] for f in fold_results]

    print(f"Per-fold: Val Pearson r={np.mean(fold_val_r):.4f}±{np.std(fold_val_r):.4f}")
    print(f"Per-fold: Val Spearman ρ={np.mean(fold_val_rho):.4f}±{np.std(fold_val_rho):.4f}")
    print(f"Per-fold: Nonzero coeffs={np.mean(fold_nonzero_counts):.1f}±{np.std(fold_nonzero_counts):.1f}")

    avg_coefficients = np.mean(fold_coefficients, axis=0)
    std_coefficients = np.std(fold_coefficients, axis=0)
    nonzero_frequency = np.mean([c != 0 for c in fold_coefficients], axis=0)

    results = {
        'aggregate_metrics': {
            'val_r': float(aggregate_val_r),
            'val_r_p': float(aggregate_val_p),
            'val_rho': float(aggregate_val_rho),
            'val_rho_p': float(aggregate_val_rho_p)
        },
        'per_fold_stats': {
            'train_r_mean': float(np.mean([f['train_r'] for f in fold_results])),
            'train_r_std': float(np.std([f['train_r'] for f in fold_results])),
            'train_rho_mean': float(np.mean([f['train_rho'] for f in fold_results])),
            'train_rho_std': float(np.std([f['train_rho'] for f in fold_results])),
            'val_r_mean': float(np.mean(fold_val_r)),
            'val_r_std': float(np.std(fold_val_r)),
            'val_rho_mean': float(np.mean(fold_val_rho)),
            'val_rho_std': float(np.std(fold_val_rho)),
            'n_nonzero_mean': float(np.mean(fold_nonzero_counts)),
            'n_nonzero_std': float(np.std(fold_nonzero_counts))
        },
        'average_coefficients': {name: float(coef) for name, coef in zip(metric_names, avg_coefficients)},
        'coefficient_std': {name: float(std) for name, std in zip(metric_names, std_coefficients)},
        'nonzero_frequency': {name: float(freq) for name, freq in zip(metric_names, nonzero_frequency)},
        'fold_results': fold_results,
        'optimization_params': {
            'objective': 'L1+Spearman',
            'lambda_l1': lambda_l1,
            'temperature': temperature,
            'n_iterations': n_iterations,
            'learning_rate': lr,
            'patience': patience,
            'convergence_threshold': convergence_threshold
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='L1+Spearman LOTO Cross-Validation')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='L1 regularization strength')
    parser.add_argument('--temperature', type=float, default=1.0, help='Soft ranking temperature')
    args = parser.parse_args()

    lambda_l1 = args.lambda_l1
    temperature = args.temperature

    metrics_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/mergeability/ViT-B-16/pairwise_metrics_N20.json')
    results_base_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/ViT-B-16')
    output_dir = Path(f'/home/ubuntu/thesis/MM/Mergeability-Bench/results/metric_linear_optimization_v2/loto_cv_l1_spearman_lambda{lambda_l1}')
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_methods = ['weight_avg', 'arithmetic', 'tsv', 'ties', 'dare']

    print("="*70)
    print(f"L1+Spearman Linear Optimization with LOTO CV (lambda={lambda_l1})")
    print("="*70)
    print()

    print("Loading data...")
    metrics_data = load_json(metrics_path)

    performance_data_dict = {}
    for method in merge_methods:
        perf_path = results_base_path / method / 'all_pairwise_summary_N20.json'
        if not perf_path.exists():
            print(f"Warning: {perf_path} not found, skipping {method}")
            continue
        performance_data_dict[method] = load_json(perf_path)

    print(f"Loaded data for methods: {list(performance_data_dict.keys())}")
    print()

    print("Extracting pairwise data...")
    metrics_array, performance_matrix, pair_names, metric_names, merge_methods = \
        extract_all_mergers_data(metrics_data, performance_data_dict)

    print(f"Number of pairs: {len(pair_names)}")
    print(f"Number of metrics: {len(metric_names)}")
    print()

    all_tasks = metrics_data['datasets']
    print(f"Total tasks: {len(all_tasks)}")
    print()

    all_results = {}

    for method_idx, method in enumerate(merge_methods):
        print("="*70)
        print(f"LOTO Cross-Validation (L1+Spearman) for: {method}")
        print("="*70)
        print()

        performance = performance_matrix[:, method_idx]

        results = run_loto_cv_l1_spearman(
            metrics_array,
            performance,
            pair_names,
            all_tasks,
            metric_names,
            lambda_l1=lambda_l1,
            temperature=temperature,
            n_iterations=2000,
            lr=0.01,
            patience=100,
            convergence_threshold=1e-4
        )

        all_results[method] = results

        method_output_file = output_dir / f'{method}_loto_results.json'
        with open(method_output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to: {method_output_file}")
        print()

    # Save combined results
    combined_results = {
        'objective': 'L1+Spearman',
        'lambda_l1': lambda_l1,
        'temperature': temperature,
        'methods': all_results
    }
    combined_output_file = output_dir / 'all_methods_loto_results.json'
    with open(combined_output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print("="*70)
    print("SUMMARY: L1+Spearman LOTO Cross-Validation Results")
    print("="*70)
    print()
    print(f"{'Method':<15} {'Train ρ':<12} {'Val ρ':<12} {'Val r':<12} {'Val ρ std':<12} {'Nonzero':<12}")
    print("-"*80)
    for method in merge_methods:
        s = all_results[method]['per_fold_stats']
        print(f"{method:<15} {s['train_rho_mean']:<12.4f} {s['val_rho_mean']:<12.4f} {s['val_r_mean']:<12.4f} {s['val_rho_std']:<12.4f} {s['n_nonzero_mean']:<12.1f}")
    print("="*80)

    print()
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
