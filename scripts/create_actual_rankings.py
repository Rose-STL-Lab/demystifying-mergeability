#!/usr/bin/env python3
"""
Create actual rankings from evaluation results and compute correlations with predicted rankings.
"""

import json
import re
from pathlib import Path
from scipy import stats
import numpy as np


def extract_subset_info_from_filename(filename: str) -> tuple:
    """Extract subset_id and tasks from filename like 'subset_0_Task1__Task2__Task3__Task4__Task5.json'"""
    match = re.match(r'subset_(\d+)_(.+)\.json', filename)
    if match:
        subset_id = int(match.group(1))
        tasks = match.group(2).split('__')
        return subset_id, tasks
    return None, None


def load_actual_results(results_dir: Path) -> list:
    """Load all actual evaluation results from a merger folder."""
    results = []

    for f in sorted(results_dir.glob('subset_*.json')):
        if f.name == 'summary.json':
            continue

        subset_id, tasks = extract_subset_info_from_filename(f.name)
        if subset_id is None:
            continue

        with open(f) as fp:
            data = json.load(fp)

        # Extract normalized accuracy
        if 'avg' in data and len(data['avg']) > 0:
            norm_acc = data['avg'][0].get('normalized_acc/test/avg', None)
            raw_acc = data['avg'][0].get('acc/test/avg', None)
        else:
            norm_acc = None
            raw_acc = None

        if norm_acc is not None:
            results.append({
                'subset_id': subset_id,
                'tasks': tasks,
                'normalized_accuracy': norm_acc,
                'raw_accuracy': raw_acc
            })

    return results


def load_predicted_rankings(rankings_path: Path) -> dict:
    """Load predicted rankings."""
    with open(rankings_path) as f:
        return json.load(f)


def create_actual_rankings(merger: str, base_path: Path) -> dict:
    """Create rankings from actual results for a merger."""
    actual_dir = base_path / 'actual_rankings' / merger

    # Load all actual results
    results = load_actual_results(actual_dir)

    if len(results) == 0:
        print(f"No results found for {merger}")
        return None

    # Sort by normalized accuracy (highest first)
    results_sorted = sorted(results, key=lambda x: x['normalized_accuracy'], reverse=True)

    # Add rank
    rankings = []
    for rank, entry in enumerate(results_sorted, 1):
        rankings.append({
            'subset_id': entry['subset_id'],
            'tasks': entry['tasks'],
            'task_indices': [],  # We don't have these, leave empty
            'normalized_accuracy': entry['normalized_accuracy'],
            'raw_accuracy': entry['raw_accuracy'],
            'rank': rank
        })

    # Compute summary stats
    accs = [r['normalized_accuracy'] for r in rankings]

    output = {
        'summary': {
            'method': merger,
            'n_subsets': len(rankings),
            'subset_size': 5,
            'accuracy_mean': np.mean(accs),
            'accuracy_std': np.std(accs),
            'accuracy_min': min(accs),
            'accuracy_max': max(accs),
            'top_5_subsets': rankings[:5],
            'bottom_5_subsets': rankings[-5:]
        },
        'rankings': rankings
    }

    return output


def compute_correlations(merger: str, base_path: Path) -> dict:
    """Compute correlations between predicted and actual rankings."""
    predicted_path = base_path / 'predicted_rankings' / f'{merger}_subset_rankings.json'
    actual_path = base_path / 'actual_rankings' / f'{merger}_rankings.json'

    if not predicted_path.exists() or not actual_path.exists():
        return None

    with open(predicted_path) as f:
        predicted = json.load(f)
    with open(actual_path) as f:
        actual = json.load(f)

    # Build lookup by subset_id
    pred_by_id = {r['subset_id']: r for r in predicted['rankings']}
    actual_by_id = {r['subset_id']: r for r in actual['rankings']}

    # Find common subset_ids
    common_ids = set(pred_by_id.keys()) & set(actual_by_id.keys())

    if len(common_ids) < 2:
        return None

    # Extract paired data
    pred_ranks = []
    actual_ranks = []
    pred_scores = []
    actual_accs = []

    for sid in sorted(common_ids):
        pred_ranks.append(pred_by_id[sid]['rank'])
        actual_ranks.append(actual_by_id[sid]['rank'])
        pred_scores.append(pred_by_id[sid]['subset_mergeability'])
        actual_accs.append(actual_by_id[sid]['normalized_accuracy'])

    # Compute correlations
    spearman_r, spearman_p = stats.spearmanr(pred_ranks, actual_ranks)
    pearson_r, pearson_p = stats.pearsonr(pred_scores, actual_accs)

    # Also compute correlation between predicted score and actual accuracy directly
    score_acc_spearman, score_acc_spearman_p = stats.spearmanr(pred_scores, actual_accs)

    return {
        'merger': merger,
        'n_common_subsets': len(common_ids),
        'rank_spearman_r': spearman_r,
        'rank_spearman_p': spearman_p,
        'score_vs_acc_pearson_r': pearson_r,
        'score_vs_acc_pearson_p': pearson_p,
        'score_vs_acc_spearman_r': score_acc_spearman,
        'score_vs_acc_spearman_p': score_acc_spearman_p
    }


def main():
    base_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/mergeability/ViT-B-16/subset_rankings')

    mergers = ['tsv', 'ties', 'weight_avg', 'arithmetic', 'dare']

    print("=" * 80)
    print("TASK 1: Creating actual rankings")
    print("=" * 80)

    for merger in mergers:
        print(f"\nProcessing {merger}...")
        rankings = create_actual_rankings(merger, base_path)

        if rankings is None:
            continue

        # Save rankings
        output_path = base_path / 'actual_rankings' / f'{merger}_rankings.json'
        with open(output_path, 'w') as f:
            json.dump(rankings, f, indent=2)

        print(f"  Saved {rankings['summary']['n_subsets']} ranked subsets to {output_path}")
        print(f"  Accuracy range: [{rankings['summary']['accuracy_min']:.4f}, {rankings['summary']['accuracy_max']:.4f}]")
        print(f"  Top subset: {rankings['summary']['top_5_subsets'][0]['tasks']}")

    print("\n" + "=" * 80)
    print("TASK 2: Computing correlations")
    print("=" * 80)

    print(f"\n{'Merger':<12} | {'N':>4} | {'Spearman r':>12} | {'Spearman p':>12} | {'Pearson r':>12} | {'Pearson p':>12}")
    print("-" * 80)

    all_correlations = {}

    for merger in mergers:
        corr = compute_correlations(merger, base_path)

        if corr is None:
            print(f"{merger:<12} | {'N/A':>4} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12}")
            continue

        all_correlations[merger] = corr

        print(f"{merger:<12} | {corr['n_common_subsets']:>4} | {corr['rank_spearman_r']:>12.4f} | {corr['rank_spearman_p']:>12.2e} | {corr['score_vs_acc_pearson_r']:>12.4f} | {corr['score_vs_acc_pearson_p']:>12.2e}")

    # Summary
    if all_correlations:
        avg_spearman = np.mean([c['rank_spearman_r'] for c in all_correlations.values()])
        avg_pearson = np.mean([c['score_vs_acc_pearson_r'] for c in all_correlations.values()])

        print("-" * 80)
        print(f"{'AVERAGE':<12} | {'':>4} | {avg_spearman:>12.4f} | {'':>12} | {avg_pearson:>12.4f} | {'':>12}")

    # Save correlations
    corr_path = base_path / 'correlations_summary.json'
    with open(corr_path, 'w') as f:
        json.dump(all_correlations, f, indent=2)
    print(f"\nCorrelations saved to {corr_path}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    for merger, corr in all_correlations.items():
        spearman = corr['rank_spearman_r']
        p_val = corr['rank_spearman_p']
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        if spearman > 0.5:
            strength = "strong positive"
        elif spearman > 0.3:
            strength = "moderate positive"
        elif spearman > 0.1:
            strength = "weak positive"
        elif spearman > -0.1:
            strength = "negligible"
        elif spearman > -0.3:
            strength = "weak negative"
        else:
            strength = "moderate/strong negative"

        print(f"{merger}: {strength} correlation (ρ={spearman:.4f}{sig})")


if __name__ == '__main__':
    main()
