#!/bin/bash
# Run subset evaluations for a given merger method
#
# Usage:
#   ./run_subset_evaluations.sh tsv           # Evaluate all 200 subsets for TSV
#   ./run_subset_evaluations.sh ties 0 50     # Evaluate subsets 0-49 for TIES
#   ./run_subset_evaluations.sh all           # Evaluate all mergers sequentially
#
# Results are saved to:
#   results/mergeability/ViT-B-16/subset_rankings/actual_rankings/{merger}/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

MERGER=${1:-"tsv"}
START_IDX=${2:-0}
END_IDX=${3:-200}

if [ "$MERGER" == "all" ]; then
    echo "Running evaluations for all mergers..."
    for m in tsv ties weight_avg arithmetic dare; do
        echo ""
        echo "=========================================="
        echo "Evaluating merger: $m"
        echo "=========================================="
        python scripts/evaluate_subset_rankings.py --merger "$m" --start_idx "$START_IDX" --end_idx "$END_IDX"
    done
else
    echo "Evaluating merger: $MERGER (subsets $START_IDX to $END_IDX)"
    python scripts/evaluate_subset_rankings.py --merger "$MERGER" --start_idx "$START_IDX" --end_idx "$END_IDX"
fi

echo ""
echo "Done! Results saved to: results/mergeability/ViT-B-16/subset_rankings/actual_rankings/"
