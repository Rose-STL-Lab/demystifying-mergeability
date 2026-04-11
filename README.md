# Mergeability-Bench

A benchmark and analysis framework for **predicting model merge quality without performing the merge**. Given two fine-tuned models, we compute geometric and statistical metrics on their weights, gradients, and activations, then learn a linear predictor that forecasts how well the models will merge under various merging algorithms.

The repository is based on an implementation originally by **Donato Crisostomi** — [donatocrisostomi@gmail.com](mailto:donatocrisostomi@gmail.com).

---

## Installation

```sh
uv sync
```

---

## Overview

The pipeline has three main stages:

1. **Fine-tuning** — train task-specific models on each dataset
2. **Metric computation** — compute pairwise mergeability metrics between all model pairs
3. **Mergeability prediction** — learn a linear (or MLP) predictor from metrics to merge quality (Pearson r with LOTO CV)

---

## Stage 1: Fine-Tuning

Fine-tune ViT models on individual tasks using:

```sh
uv run scripts/finetune.py
```

Configuration is in `conf/finetune.yaml`. Supported backbones: **ViT-B-16**, **ViT-B-32**. Supported optimizers: **AdamW**, **SGD**. Checkpoints are saved to `checkpoints/` and also uploaded to W&B via `scripts/upload_regularized_models_wandb.py`.

---

## Stage 2: Merging Evaluation

Evaluate multi-task merging performance on all pairwise combinations of tasks:

```sh
uv run scripts/evaluate_multitask_merging.py
```

Configure the merger, benchmark (N8/N14/N20), and other options in `conf/multitask.yaml`:

```yaml
merger: tsv        # tsv, task_arithmetic, weight_avg, ties, dare, tall-masks, iso-cts
benchmark: N20     # N8, N14, N20
all_pairwise: true
```

Supported merge methods (configs in `conf/merger/`): `weight_avg`, `task_arithmetic`, `tsv`, `ties`, `dare`, `tall-masks`, `iso-cts`.

---

## Stage 3: Mergeability Metric Computation

Compute pairwise geometric/statistical metrics between fine-tuned model checkpoints:

```sh
uv run scripts/compute_mergeability.py
```

Metrics are organized into five categories:

| Category | Examples |
|----------|---------|
| **Effective Rank** | `effective_rank`, `stable_rank` |
| **Gradient-Based** | `encoder_gradient_l2_distance`, `input_gradient_l2_distance` |
| **Activation** | `activation_l2_distance`, `activation_dot_product`, `activation_cosine_similarity` |
| **Subspace** | `singular_value_overlap`, `right_subspace_overlap_top_k`, `right_subspace_overlap_bottom_k` |
| **Task Vector** | `task_vector_dot_product`, `task_vector_cosine_similarity`, `interaction_matrix_overlap` |

28 metrics total are used in experiments (`right_subspace_overlap` is excluded as it is the average of its top-k and bottom-k variants).

---

## Stage 4: Mergeability Prediction

### L1-Regularized Linear Predictor (primary method)

Leave-One-Task-Out cross-validation with L1 regularization (λ=1.0):

```sh
python scripts/linear_optimization_loto_l1.py \
    --model ViT-B-16_AdamW \
    --lambda_l1 1.0 \
    --output_dir results/metric_linear_optimization_v2
```

Results are saved per merge method under `results/metric_linear_optimization_v2/{model}/loto_cv_l1_lambda{λ}/`.

### Other Predictors

```sh
# MSE objective instead of Pearson correlation
python scripts/linear_optimization_loto_mse.py --model ViT-B-16_AdamW

# Backward elimination (reverse greedy feature selection)
python scripts/linear_optimization_loto_reverse_greedy.py --model ViT-B-16_AdamW

# MLP predictor (separate MLP per merge method, LOTO CV)
python scripts/learnable_mergeability_separate_loto.py \
    "learnable_mergeability.model_name=ViT-B-16_AdamW" \
    "learnable_mergeability.merge_methods=[weight_avg,arithmetic,tsv,ties,dare]"
```

### Single-Fold Experiment

Train on 10 tasks (~45 pairs), validate on disjoint 10 tasks (~45 pairs):

```sh
python scripts/linear_optimization_single_fold_l1.py \
    --model ViT-B-16_AdamW \
    --lambda_l1 1.0 \
    --seed 42
```

---

## Analysis Scripts

```sh
# Metric selection rates and coefficient magnitudes after LOTO
bash scripts/analyze_results_l1_loto.sh

# Compare ViT-B-16 vs ViT-B-32
bash scripts/compare_b16_vs_b32.sh

# Individual metric Pearson correlations (no learning)
python scripts/compute_individual_metric_correlations.py

# Feature importance table for paper
python scripts/generate_coefficient_table.py

# Category ablation (remove one metric category at a time)
bash scripts/run_l1_loto_ablations.sh
```

---

## Results Structure

```
results/
├── metric_linear_optimization_v2/
│   ├── vit-b-16_AdamW/         # AdamW fine-tuned ViT-B-16
│   │   ├── loto_cv_l1_lambda1.0/
│   │   ├── loto_cv_l1_lambda0.0/
│   │   ├── loto_cv_mse/
│   │   ├── loto_cv_reverse_greedy_selection/
│   │   ├── l1_loto_cv_no_{category}/   # category ablations
│   │   └── single_fold_l1_lambda1.0/
│   ├── vit-b-16_SGD/           # SGD fine-tuned ViT-B-16
│   └── vit-b-32_AwamW/         # AdamW fine-tuned ViT-B-32
└── paper_writing/              # LaTeX sections for the paper
    ├── mlp_vs_l1.tex
    ├── sgd_vs_adamw.tex
    ├── loto_vs_single.tex
    ├── b16_vs_b32.tex
    ├── category_removal.tex
    └── ...
```

---

## Key Findings

- **Gradient metrics dominate**: `encoder_gradient_l2_distance` and `input_gradient_l2_distance` are the top-ranked features across all merge methods and both fine-tuning optimizers (AdamW and SGD), selected at 100% frequency with the largest coefficients.
- **L1 > MLP**: The sparse linear predictor outperforms separate per-method MLPs in LOTO CV, due to limited training data (~180 pairs) and the inherent near-linearity of the mergeability signal.
- **L1 helps AdamW, not SGD**: For AdamW models, L1 regularization improves generalization (val r: 0.544 vs. 0.525 without regularization). For SGD models, the unregularized predictor is better (0.655 vs. 0.590), suggesting SGD's implicit regularization reduces the need for explicit penalties.
- **Cross-architecture generalization**: Top features are consistent between ViT-B-16 and ViT-B-32, validating that the metrics capture architecture-agnostic properties.
- **Single-fold generalization**: Training on 10 tasks (~45 pairs) and evaluating on a disjoint 10 tasks yields val r ≈ 0.43, confirming that the predictor captures task-agnostic geometric structure.

---

## Benchmarks

| Benchmark | Tasks | Pairwise Combinations |
|-----------|-------|-----------------------|
| N8        | 8     | 28                    |
| N14       | 14    | 91                    |
| N20       | 20    | 190                   |

Tasks include: CIFAR-10/100, MNIST, SVHN, GTSRB, EuroSAT, DTD, RESISC45, STL10, SUN397, Cars, Food101, Flowers102, OxfordIIITPet, EMNIST, FashionMNIST, KMNIST, PCAM, FER2013, RenderedSST2.