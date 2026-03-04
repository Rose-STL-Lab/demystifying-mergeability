#!/usr/bin/env python3
"""
Subset Mergeability Ranking

Tests whether pairwise metrics generalize to multi-way merging by:
1. Sampling 200 random subsets of 5 tasks
2. Computing subset mergeability as average pairwise predicted mergeability
3. Ranking subsets by this score for each merger method

Usage:
    python subset_mergeability_ranking.py
"""

import json
import random
from pathlib import Path
from itertools import combinations
import argparse


def load_pairwise_metrics(metrics_path: Path) -> dict:
    """Load pairwise metrics data."""
    with open(metrics_path) as f:
        return json.load(f)


def load_loto_results(loto_path: Path) -> dict:
    """Load L1 LOTO results with average coefficients."""
    with open(loto_path) as f:
        return json.load(f)


def get_pairwise_metric_value(metrics_data: dict, metric_name: str, task1: str, task2: str) -> float:
    """Get the metric value for a specific pair of tasks by their names."""
    pairs_dict = metrics_data['metrics'][metric_name]['pairs']
    pair_key = f"{task1}__{task2}"
    reverse_key = f"{task2}__{task1}"

    val = None
    if pair_key in pairs_dict:
        val = pairs_dict[pair_key]
    elif reverse_key in pairs_dict:
        val = pairs_dict[reverse_key]

    # Handle None/NaN values
    if val is None:
        return 0.0
    return val


def compute_pairwise_predicted_mergeability(
    metrics_data: dict,
    coefficients: dict,
    task1: str,
    task2: str
) -> float:
    """Compute predicted mergeability for a pair using linear coefficients."""
    score = 0.0
    for metric_name, coef in coefficients.items():
        if metric_name not in metrics_data['metrics']:
            continue
        metric_val = get_pairwise_metric_value(metrics_data, metric_name, task1, task2)
        score += coef * metric_val
    return score


def compute_subset_mergeability(
    metrics_data: dict,
    coefficients: dict,
    tasks: list
) -> float:
    """
    Compute subset mergeability as average pairwise predicted mergeability.
    For 5 tasks, this averages over C(5,2) = 10 pairs.
    """
    pairs = list(combinations(tasks, 2))
    pairwise_scores = []
    for task1, task2 in pairs:
        score = compute_pairwise_predicted_mergeability(
            metrics_data, coefficients, task1, task2
        )
        pairwise_scores.append(score)
    return sum(pairwise_scores) / len(pairwise_scores)


def sample_subsets(n_tasks: int, subset_size: int, n_subsets: int, seed: int = 42) -> list:
    """Sample random subsets of tasks."""
    random.seed(seed)
    all_indices = list(range(n_tasks))
    subsets = []
    seen = set()

    while len(subsets) < n_subsets:
        subset = tuple(sorted(random.sample(all_indices, subset_size)))
        if subset not in seen:
            seen.add(subset)
            subsets.append(list(subset))

    return subsets


def main():
    parser = argparse.ArgumentParser(description='Compute subset mergeability rankings')
    parser.add_argument('--n_subsets', type=int, default=200, help='Number of subsets to sample')
    parser.add_argument('--subset_size', type=int, default=5, help='Number of tasks per subset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Paths
    base_path = Path(__file__).parent.parent
    metrics_path = base_path / 'results' / 'mergeability' / 'ViT-B-16' / 'pairwise_metrics_N20.json'
    loto_path = base_path / 'results' / 'metric_linear_optimization_v2' / 'loto_cv_l1_lambda1.0' / 'all_methods_loto_results.json'
    output_dir = base_path / 'results' / 'mergeability' / 'ViT-B-16' / 'subset_rankings'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading pairwise metrics from {metrics_path}")
    metrics_data = load_pairwise_metrics(metrics_path)

    print(f"Loading LOTO results from {loto_path}")
    loto_results = load_loto_results(loto_path)

    datasets = metrics_data['datasets']
    n_tasks = len(datasets)
    print(f"Found {n_tasks} tasks: {datasets}")

    # Sample subsets
    print(f"\nSampling {args.n_subsets} subsets of size {args.subset_size} (seed={args.seed})")
    subsets = sample_subsets(n_tasks, args.subset_size, args.n_subsets, args.seed)

    # Convert indices to task names for output
    subsets_with_names = [
        {'indices': subset, 'tasks': [datasets[i] for i in subset]}
        for subset in subsets
    ]

    # Methods to process
    methods = ['tsv', 'ties', 'weight_avg', 'arithmetic', 'dare']

    for method in methods:
        if method not in loto_results['methods']:
            print(f"Warning: Method '{method}' not found in LOTO results, skipping")
            continue

        print(f"\nProcessing method: {method}")
        coefficients = loto_results['methods'][method]['average_coefficients']

        # Compute subset mergeability for each subset
        rankings = []
        for i, subset_info in enumerate(subsets_with_names):
            indices = subset_info['indices']
            tasks = subset_info['tasks']

            mergeability_score = compute_subset_mergeability(
                metrics_data, coefficients, tasks
            )

            rankings.append({
                'subset_id': i,
                'tasks': tasks,
                'task_indices': indices,
                'subset_mergeability': mergeability_score
            })

        # Sort by mergeability (highest first)
        rankings.sort(key=lambda x: x['subset_mergeability'], reverse=True)

        # Add rank
        for rank, entry in enumerate(rankings, 1):
            entry['rank'] = rank

        # Compute summary stats
        scores = [r['subset_mergeability'] for r in rankings]
        summary = {
            'method': method,
            'n_subsets': len(rankings),
            'subset_size': args.subset_size,
            'seed': args.seed,
            'score_mean': sum(scores) / len(scores),
            'score_std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
            'score_min': min(scores),
            'score_max': max(scores),
            'top_5_subsets': rankings[:5],
            'bottom_5_subsets': rankings[-5:]
        }

        # Output
        output = {
            'summary': summary,
            'rankings': rankings
        }

        output_path = output_dir / f'{method}_subset_rankings.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"  Saved rankings to {output_path}")
        print(f"  Score range: [{summary['score_min']:.6f}, {summary['score_max']:.6f}]")
        print(f"  Top subset: {rankings[0]['tasks']}")
        print(f"  Bottom subset: {rankings[-1]['tasks']}")

    print(f"\nDone! Rankings saved to {output_dir}")


if __name__ == '__main__':
    main()
