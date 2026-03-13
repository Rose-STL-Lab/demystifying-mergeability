#!/usr/bin/env python3
"""
Compare Predicted vs Actual Subset Rankings

Loads actual evaluation results, ranks them by performance, and compares
with predicted rankings using Spearman correlation.

Usage:
    python scripts/compare_subset_rankings.py --merger tsv
    python scripts/compare_subset_rankings.py --merger all
"""

import argparse
import json
from pathlib import Path
from scipy import stats
import numpy as np


def load_predicted_rankings(rankings_path: Path) -> dict:
    """Load predicted subset rankings."""
    with open(rankings_path) as f:
        return json.load(f)


def load_actual_results(results_dir: Path) -> dict:
    """Load actual evaluation results from individual subset files."""
    results = {}
    for f in results_dir.glob('subset_*.json'):
        if f.name == 'summary.json':
            continue
        with open(f) as fp:
            data = json.load(fp)
        # Extract subset name from filename: subset_{id}_{name}.json
        # or from the result data
        subset_name = f.stem.split('_', 2)[-1] if '_' in f.stem else f.stem
        results[subset_name] = data
    return results


def extract_avg_accuracy(result: dict) -> float:
    """Extract average accuracy from evaluation result."""
    if 'error' in result:
        return None
    if 'avg' in result:
        avg_data = result['avg']
        if isinstance(avg_data, list) and len(avg_data) > 0:
            return avg_data[0].get('test/avg_accuracy', None)
        elif isinstance(avg_data, dict):
            return avg_data.get('test/avg_accuracy', None)
    return None


def compare_rankings(merger: str, base_path: Path) -> dict:
    """Compare predicted vs actual rankings for a merger."""
    predicted_path = base_path / 'predicted_rankings' / f'{merger}_subset_rankings.json'
    actual_dir = base_path / 'actual_rankings' / merger

    if not predicted_path.exists():
        print(f"Predicted rankings not found: {predicted_path}")
        return None

    if not actual_dir.exists():
        print(f"Actual results not found: {actual_dir}")
        return None

    # Load data
    predicted_data = load_predicted_rankings(predicted_path)
    actual_results = load_actual_results(actual_dir)

    if len(actual_results) == 0:
        print(f"No actual results found in {actual_dir}")
        return None

    # Build comparison data
    comparison = []
    for pred_entry in predicted_data['rankings']:
        subset_name = "__".join(pred_entry['tasks'])
        subset_id = pred_entry['subset_id']

        if subset_name not in actual_results:
            continue

        actual_acc = extract_avg_accuracy(actual_results[subset_name])
        if actual_acc is None:
            continue

        comparison.append({
            'subset_id': subset_id,
            'subset_name': subset_name,
            'tasks': pred_entry['tasks'],
            'predicted_score': pred_entry['subset_mergeability'],
            'predicted_rank': pred_entry['rank'],
            'actual_accuracy': actual_acc
        })

    if len(comparison) < 2:
        print(f"Not enough valid results for comparison ({len(comparison)} subsets)")
        return None

    # Sort by actual accuracy to get actual ranks
    comparison_sorted = sorted(comparison, key=lambda x: x['actual_accuracy'], reverse=True)
    for i, entry in enumerate(comparison_sorted, 1):
        entry['actual_rank'] = i

    # Re-sort by subset_id for consistent ordering
    comparison = sorted(comparison, key=lambda x: x['subset_id'])

    # Extract ranks for correlation
    predicted_ranks = [e['predicted_rank'] for e in comparison]
    actual_ranks = [e['actual_rank'] for e in comparison]

    # Compute Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(predicted_ranks, actual_ranks)

    # Also compute correlation between predicted score and actual accuracy
    predicted_scores = [e['predicted_score'] for e in comparison]
    actual_accs = [e['actual_accuracy'] for e in comparison]
    score_corr, score_p = stats.pearsonr(predicted_scores, actual_accs)

    # Get top/bottom comparisons
    comparison_by_pred = sorted(comparison, key=lambda x: x['predicted_rank'])
    comparison_by_actual = sorted(comparison, key=lambda x: x['actual_rank'])

    result = {
        'merger': merger,
        'n_subsets_evaluated': len(comparison),
        'n_subsets_total': len(predicted_data['rankings']),
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'score_accuracy_correlation': score_corr,
        'score_accuracy_p_value': score_p,
        'predicted_top_5': [
            {
                'tasks': e['tasks'],
                'predicted_rank': e['predicted_rank'],
                'actual_rank': e['actual_rank'],
                'actual_accuracy': e['actual_accuracy']
            }
            for e in comparison_by_pred[:5]
        ],
        'predicted_bottom_5': [
            {
                'tasks': e['tasks'],
                'predicted_rank': e['predicted_rank'],
                'actual_rank': e['actual_rank'],
                'actual_accuracy': e['actual_accuracy']
            }
            for e in comparison_by_pred[-5:]
        ],
        'actual_top_5': [
            {
                'tasks': e['tasks'],
                'predicted_rank': e['predicted_rank'],
                'actual_rank': e['actual_rank'],
                'actual_accuracy': e['actual_accuracy']
            }
            for e in comparison_by_actual[:5]
        ],
        'full_comparison': comparison
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Compare predicted vs actual subset rankings')
    parser.add_argument('--merger', type=str, required=True,
                        help='Merger method (tsv, ties, weight_avg, arithmetic, dare, or "all")')
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent / 'results' / 'mergeability' / 'ViT-B-16' / 'subset_rankings'
    output_dir = base_path / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merger == 'all':
        mergers = ['tsv', 'ties', 'weight_avg', 'arithmetic', 'dare']
    else:
        mergers = [args.merger]

    all_results = {}

    for merger in mergers:
        print(f"\n{'='*60}")
        print(f"Comparing rankings for: {merger}")
        print(f"{'='*60}")

        result = compare_rankings(merger, base_path)
        if result is None:
            continue

        all_results[merger] = result

        print(f"Subsets evaluated: {result['n_subsets_evaluated']}/{result['n_subsets_total']}")
        print(f"Spearman correlation (rank): {result['spearman_correlation']:.4f} (p={result['spearman_p_value']:.4e})")
        print(f"Pearson correlation (score vs acc): {result['score_accuracy_correlation']:.4f} (p={result['score_accuracy_p_value']:.4e})")

        print(f"\nPredicted Top 5 subsets:")
        for e in result['predicted_top_5']:
            print(f"  Pred rank {e['predicted_rank']:3d} -> Actual rank {e['actual_rank']:3d} (acc={e['actual_accuracy']:.4f})")

        print(f"\nActual Top 5 subsets:")
        for e in result['actual_top_5']:
            print(f"  Actual rank {e['actual_rank']:3d} <- Pred rank {e['predicted_rank']:3d} (acc={e['actual_accuracy']:.4f})")

        # Save individual result
        output_file = output_dir / f'{merger}_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_file}")

    # Save summary
    if len(all_results) > 0:
        summary = {
            merger: {
                'spearman_correlation': r['spearman_correlation'],
                'spearman_p_value': r['spearman_p_value'],
                'score_accuracy_correlation': r['score_accuracy_correlation'],
                'n_subsets': r['n_subsets_evaluated']
            }
            for merger, r in all_results.items()
        }
        summary_file = output_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
