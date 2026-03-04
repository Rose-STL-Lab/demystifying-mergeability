#!/usr/bin/env python3
"""
Evaluate Subset Rankings

Evaluates the exact subsets from predicted rankings to get actual performance.
This ensures we compare apples-to-apples between predicted and actual rankings.

Usage:
    python scripts/evaluate_subset_rankings.py --merger tsv
    python scripts/evaluate_subset_rankings.py --merger ties --start_idx 0 --end_idx 50
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

pylogger = logging.getLogger(__name__)


def load_subset_rankings(rankings_path: Path) -> dict:
    """Load subset rankings JSON."""
    with open(rankings_path) as f:
        return json.load(f)


def get_dataset_config_by_name(benchmark_datasets: list, task_name: str):
    """Find the dataset config object by task name."""
    for ds in benchmark_datasets:
        if ds.name == task_name:
            return ds
    raise ValueError(f"Dataset {task_name} not found in benchmark")


def main():
    parser = argparse.ArgumentParser(description='Evaluate subset rankings')
    parser.add_argument('--merger', type=str, required=True,
                        choices=['tsv', 'ties', 'weight_avg', 'arithmetic', 'dare'],
                        help='Merger method to evaluate')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index (inclusive) for subset evaluation')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index (exclusive) for subset evaluation')
    parser.add_argument('--config_name', type=str, default='multitask.yaml',
                        help='Hydra config name')
    args = parser.parse_args()

    # Paths
    base_path = Path(__file__).parent.parent
    rankings_dir = base_path / 'results' / 'mergeability' / 'ViT-B-16' / 'subset_rankings'
    rankings_file = rankings_dir / 'predicted_rankings' / f'{args.merger}_subset_rankings.json'
    output_dir = rankings_dir / 'actual_rankings' / args.merger
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if rankings file exists, fall back to non-predicted path
    if not rankings_file.exists():
        rankings_file = rankings_dir / f'{args.merger}_subset_rankings.json'

    if not rankings_file.exists():
        print(f"Error: Rankings file not found: {rankings_file}")
        sys.exit(1)

    print(f"Loading rankings from: {rankings_file}")
    rankings_data = load_subset_rankings(rankings_file)
    all_subsets = rankings_data['rankings']

    # Determine range to evaluate
    end_idx = args.end_idx if args.end_idx is not None else len(all_subsets)
    subsets_to_eval = all_subsets[args.start_idx:end_idx]
    print(f"Evaluating subsets {args.start_idx} to {end_idx} ({len(subsets_to_eval)} subsets)")

    # Map merger name to Hydra merger config
    merger_map = {
        'tsv': 'tsv',
        'ties': 'ties',
        'weight_avg': 'weight_avg',
        'arithmetic': 'task_arithmetic',
        'dare': 'dare'
    }
    merger_config = merger_map[args.merger]

    # Initialize Hydra
    config_path = str(base_path / 'conf')

    # Import run_single after setting up paths
    from scripts.evaluate_multitask_merging import run_single

    results_summary = {}
    evaluated = 0
    skipped = 0

    for subset_info in subsets_to_eval:
        subset_id = subset_info['subset_id']
        tasks = subset_info['tasks']
        subset_name = "__".join(tasks)

        # Check if already evaluated
        result_file = output_dir / f'subset_{subset_id}_{subset_name}.json'
        if result_file.exists():
            print(f"[{subset_id}] Skipping (already exists): {tasks}")
            with open(result_file) as f:
                results_summary[subset_name] = json.load(f)
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"[{subset_id}] Evaluating: {tasks}")
        print(f"{'='*60}")

        try:
            # Initialize Hydra for this run
            with initialize_config_dir(config_dir=config_path, version_base=None):
                cfg = compose(
                    config_name=args.config_name,
                    overrides=[
                        f"merger={merger_config}",
                        "benchmark=N20",
                        "alignment=false",
                    ]
                )

                # Get dataset configs for the tasks in this subset
                benchmark_datasets = list(cfg.benchmark.datasets)
                subset_datasets = [get_dataset_config_by_name(benchmark_datasets, task) for task in tasks]

                # Override results path
                OmegaConf.set_struct(cfg, False)
                cfg.misc.results_path = str(output_dir)
                OmegaConf.set_struct(cfg, True)

                # Run evaluation
                result = run_single(
                    cfg,
                    datasets_to_use=subset_datasets,
                    pair_name=subset_name,
                    file_prefix=f"subset_{subset_id}"
                )

                # Save individual result
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)

                results_summary[subset_name] = result
                evaluated += 1

        except Exception as e:
            print(f"Error evaluating subset {subset_id}: {e}")
            import traceback
            traceback.print_exc()
            results_summary[subset_name] = {"error": str(e)}

    # Save summary
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'merger': args.merger,
            'evaluated': evaluated,
            'skipped': skipped,
            'total': len(subsets_to_eval),
            'results': results_summary
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"Evaluated: {evaluated}, Skipped: {skipped}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
