#!/bin/bash
# Run L1 LOTO ablation experiments - exclude each metric category once

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || echo "$(cd "$(dirname "$0")/.." && pwd)")}"
cd "$PROJECT_ROOT"

# Base output directory
BASE_OUTPUT="$PROJECT_ROOT/results/metric_linear_optimization_v2"

# Backup existing baseline L1 results if they exist
if [ -d "${BASE_OUTPUT}/loto_cv_l1_lambda1.0" ]; then
    echo "Backing up existing baseline L1 results..."
    cp -r "${BASE_OUTPUT}/loto_cv_l1_lambda1.0" "${BASE_OUTPUT}/loto_cv_l1_lambda1.0_baseline"
fi

# Define metric groups
EFF_RANK_METRICS="effective_rank effective_rank_mergeability_score layerwise_effective_rank layerwise_effective_rank_mergeability_score stable_rank spectral_gap singular_value_ratio"

GRAD_BASED_METRICS="encoder_gradient_cosine_similarity encoder_gradient_l2_distance encoder_gradient_dot_product input_gradient_cosine_similarity input_gradient_l2_distance input_gradient_dot_product"

ACTIVATION_METRICS="activation_l2_distance activation_cosine_similarity activation_magnitude_ratio activation_dot_product"

SUBSPACE_METRICS="right_subspace_overlap right_subspace_overlap_top_k right_subspace_overlap_bottom_k subspace_overlap singular_value_overlap interaction_matrix_overlap_top_k interaction_matrix_overlap_bottom_k"

TASK_VECTOR_METRICS="task_vector_cosine_similarity task_vector_l2_distance task_vector_dot_product weight_space_angle task_vector_magnitude_ratio"

echo "=============================================================="
echo "Running L1 LOTO Ablation Experiments (lambda=1.0)"
echo "=============================================================="
echo ""

# 1. No effective rank metrics
echo "=============================================================="
echo "1/5: Excluding EFF_RANK metrics"
echo "=============================================================="
python scripts/linear_optimization_loto_l1.py --lambda_l1 1.0 --exclude_metrics $EFF_RANK_METRICS
# Move results to correct folder
mv "${BASE_OUTPUT}/loto_cv_l1_lambda1.0" "${BASE_OUTPUT}/l1_loto_cv_no_eff_rank"
echo "Saved to: ${BASE_OUTPUT}/l1_loto_cv_no_eff_rank"
echo ""

# 2. No gradient-based metrics
echo "=============================================================="
echo "2/5: Excluding GRAD_BASED metrics"
echo "=============================================================="
python scripts/linear_optimization_loto_l1.py --lambda_l1 1.0 --exclude_metrics $GRAD_BASED_METRICS
mv "${BASE_OUTPUT}/loto_cv_l1_lambda1.0" "${BASE_OUTPUT}/l1_loto_cv_no_grad_based"
echo "Saved to: ${BASE_OUTPUT}/l1_loto_cv_no_grad_based"
echo ""

# 3. No activation metrics
echo "=============================================================="
echo "3/5: Excluding ACTIVATION metrics"
echo "=============================================================="
python scripts/linear_optimization_loto_l1.py --lambda_l1 1.0 --exclude_metrics $ACTIVATION_METRICS
mv "${BASE_OUTPUT}/loto_cv_l1_lambda1.0" "${BASE_OUTPUT}/l1_loto_cv_no_activation"
echo "Saved to: ${BASE_OUTPUT}/l1_loto_cv_no_activation"
echo ""

# 4. No subspace metrics
echo "=============================================================="
echo "4/5: Excluding SUBSPACE metrics"
echo "=============================================================="
python scripts/linear_optimization_loto_l1.py --lambda_l1 1.0 --exclude_metrics $SUBSPACE_METRICS
mv "${BASE_OUTPUT}/loto_cv_l1_lambda1.0" "${BASE_OUTPUT}/l1_loto_cv_no_subspace"
echo "Saved to: ${BASE_OUTPUT}/l1_loto_cv_no_subspace"
echo ""

# 5. No task vector metrics
echo "=============================================================="
echo "5/5: Excluding TASK_VECTOR metrics"
echo "=============================================================="
python scripts/linear_optimization_loto_l1.py --lambda_l1 1.0 --exclude_metrics $TASK_VECTOR_METRICS
mv "${BASE_OUTPUT}/loto_cv_l1_lambda1.0" "${BASE_OUTPUT}/l1_loto_cv_no_task_vector"
echo "Saved to: ${BASE_OUTPUT}/l1_loto_cv_no_task_vector"
echo ""

echo "=============================================================="
echo "All ablation experiments complete!"
echo "=============================================================="

# Restore baseline L1 results
if [ -d "${BASE_OUTPUT}/loto_cv_l1_lambda1.0_baseline" ]; then
    echo ""
    echo "Restoring baseline L1 results..."
    rm -rf "${BASE_OUTPUT}/loto_cv_l1_lambda1.0"
    mv "${BASE_OUTPUT}/loto_cv_l1_lambda1.0_baseline" "${BASE_OUTPUT}/loto_cv_l1_lambda1.0"
fi

echo ""
echo "Results saved to:"
echo "  - ${BASE_OUTPUT}/l1_loto_cv_no_eff_rank"
echo "  - ${BASE_OUTPUT}/l1_loto_cv_no_grad_based"
echo "  - ${BASE_OUTPUT}/l1_loto_cv_no_activation"
echo "  - ${BASE_OUTPUT}/l1_loto_cv_no_subspace"
echo "  - ${BASE_OUTPUT}/l1_loto_cv_no_task_vector"
echo ""
echo "Baseline results preserved at:"
echo "  - ${BASE_OUTPUT}/loto_cv_l1_lambda1.0"
