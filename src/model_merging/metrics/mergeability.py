"""
Mergeability metrics for predicting model merging outcomes.

This module provides various metrics to measure the compatibility/similarity
between two task vectors, which can be used to predict the success of model merging.
"""

import math
import copy
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def flatten_task_dict(task_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all tensors in a task dict into a single 1D vector.

    Args:
        task_dict: Dictionary mapping layer names to parameter tensors.

    Returns:
        A single flattened tensor containing all parameters.
    """
    tensors = []
    for key in sorted(task_dict.keys()):
        tensor = task_dict[key]
        if tensor.dtype in [torch.int64, torch.uint8]:
            continue
        tensors.append(tensor.flatten().float())
    return torch.cat(tensors)


def get_layer_vectors(
    task_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Extract flattened vectors for each layer.

    Args:
        task_dict: Dictionary mapping layer names to parameter tensors.

    Returns:
        Dictionary mapping layer names to flattened tensors.
    """
    result = {}
    for key, tensor in task_dict.items():
        if tensor.dtype in [torch.int64, torch.uint8]:
            continue
        result[key] = tensor.flatten().float()
    return result


# =============================================================================
# Per-Layer Computation Wrapper
# =============================================================================


def compute_metric_per_layer(
    metric_fn: Callable,
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute any metric for each layer separately.

    Args:
        metric_fn: A metric function that takes two task dicts and returns a float.
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Dictionary mapping layer names to metric values.
    """
    layers_1 = get_layer_vectors(task_dict_1)
    layers_2 = get_layer_vectors(task_dict_2)

    result = {}
    common_keys = set(layers_1.keys()) & set(layers_2.keys())

    for key in sorted(common_keys):
        # Create single-layer task dicts
        layer_dict_1 = {key: task_dict_1[key]}
        layer_dict_2 = {key: task_dict_2[key]}
        try:
            result[key] = metric_fn(layer_dict_1, layer_dict_2)
        except Exception:
            result[key] = float('nan')

    return result


def compute_metric_layer_wise_avg(
    metric_fn: Callable,
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute average of a metric across all layers.

    Args:
        metric_fn: A metric function that takes two task dicts and returns a float.
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Average metric value across layers.
    """
    per_layer = compute_metric_per_layer(metric_fn, task_dict_1, task_dict_2)
    valid_values = [v for v in per_layer.values() if not math.isnan(v)]
    if not valid_values:
        return 0.0
    return sum(valid_values) / len(valid_values)


# =============================================================================
# Core Metrics
# =============================================================================


def task_vector_cosine_similarity(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute cosine similarity between two task vectors.

    This is one of the most intuitive metrics: if two task vectors point
    in similar directions in weight space, they may be more compatible.

    Args:
        task_dict_1: First task vector (finetuned - pretrained).
        task_dict_2: Second task vector.

    Returns:
        Cosine similarity value in [-1, 1].
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()


def task_vector_l2_distance(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute L2 (Euclidean) distance between two task vectors.

    Measures how far apart the two task vectors are in weight space.
    Smaller distance might indicate more compatible tasks.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        L2 distance (non-negative).
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    return torch.norm(vec1 - vec2, p=2).item()


def task_vector_dot_product(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute dot product between two task vectors.

    Unlike cosine similarity, this is not normalized by magnitude,
    so it captures both direction and magnitude information.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Dot product value.
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    return torch.dot(vec1, vec2).item()


def weight_space_angle(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute angle between two task vectors in weight space (in degrees).

    This is derived from cosine similarity but expressed as an angle,
    which can be more intuitive for interpretation.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Angle in degrees [0, 180].
    """
    cos_sim = task_vector_cosine_similarity(task_dict_1, task_dict_2)
    # Clamp to handle numerical errors
    cos_sim = max(-1.0, min(1.0, cos_sim))
    angle_rad = math.acos(cos_sim)
    return math.degrees(angle_rad)


def task_vector_magnitude_ratio(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute ratio of task vector magnitudes (smaller / larger).

    If one task vector is much larger than the other, the smaller task
    might get "overwhelmed" during merging. A ratio close to 1 suggests
    more balanced contributions.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Magnitude ratio in (0, 1].
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    mag1 = torch.norm(vec1, p=2).item()
    mag2 = torch.norm(vec2, p=2).item()

    if mag1 < 1e-8 or mag2 < 1e-8:
        return 0.0

    return min(mag1, mag2) / max(mag1, mag2)


# =============================================================================
# SVD-based Metrics
# =============================================================================


def effective_rank(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute effective rank (participation ratio) of the two task vectors.

    The effective rank measures the intrinsic dimensionality of the subspace
    spanned by the two task vectors. It is computed using the entropy of the
    normalized singular value distribution.

    Effective Rank = exp(H(p)) where H(p) = -Σ p_i log(p_i)
    and p_i = σ_i / Σσ_j (normalized singular values)

    Interpretation:
        - Effective rank ≈ 1.0: Task vectors are highly aligned (excellent mergeability)
        - Effective rank ≈ 1.5: Moderate alignment (good mergeability)
        - Effective rank ≈ 2.0: Task vectors are orthogonal (poor mergeability)

    This metric is based on the hypothesis that models lying in the linear
    tangent space of the pretrained model should have aligned task vectors,
    resulting in low effective rank.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Effective rank value in [1, 2].
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    # Stack as matrix (2 × D)
    task_matrix = torch.stack([vec1, vec2], dim=0)

    # Compute SVD
    try:
        _, S, _ = torch.linalg.svd(task_matrix, full_matrices=False)
    except Exception:
        return 2.0  # Return worst case on failure

    # Normalize singular values to form probability distribution
    S_normalized = S / (S.sum() + 1e-10)

    # Compute entropy
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()

    # Effective rank
    eff_rank = torch.exp(entropy).item()

    return eff_rank


def effective_rank_mergeability_score(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute mergeability score from effective rank (mapped to [0, 1]).

    This is a normalized version of effective_rank where:
        - Score = 1.0 means perfect alignment (effective rank = 1.0)
        - Score = 0.0 means orthogonal (effective rank = 2.0)

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Mergeability score in [0, 1], where higher is better.
    """
    eff_rank = effective_rank(task_dict_1, task_dict_2)

    # Map [1, 2] to [1, 0]
    score = 2.0 - eff_rank
    score = max(0.0, min(1.0, score))

    return score


def stable_rank(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute stable rank of the two task vectors.

    Stable rank is an alternative measure of effective dimensionality:
    Stable Rank = (Σσ_i)² / Σσ_i²

    This is related to effective rank but uses L2 norm instead of entropy.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Stable rank value in [1, 2].
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    # Stack as matrix (2 × D)
    task_matrix = torch.stack([vec1, vec2], dim=0)

    # Compute SVD
    try:
        _, S, _ = torch.linalg.svd(task_matrix, full_matrices=False)
    except Exception:
        return 2.0

    # Stable rank = (sum of singular values)^2 / sum of squared singular values
    s_rank = (S.sum() ** 2) / ((S ** 2).sum() + 1e-10)

    return s_rank.item()


def spectral_gap(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute spectral gap between the two largest singular values.

    The spectral gap measures the difference between the first and second
    singular values, normalized by the first. A large gap indicates strong
    alignment (one dominant direction).

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Spectral gap in [0, 1], where larger means better alignment.
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    # Stack as matrix (2 × D)
    task_matrix = torch.stack([vec1, vec2], dim=0)

    # Compute SVD
    try:
        _, S, _ = torch.linalg.svd(task_matrix, full_matrices=False)
    except Exception:
        return 0.0

    if len(S) < 2:
        return 1.0  # Only one singular value means perfect alignment

    # Spectral gap = (σ_1 - σ_2) / σ_1
    gap = (S[0] - S[1]) / (S[0] + 1e-10)

    return gap.item()


def singular_value_ratio(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute ratio of second to first singular value.

    This is complementary to spectral_gap. A small ratio indicates
    strong alignment (second direction is weak).

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Ratio in [0, 1], where smaller means better alignment.
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    # Stack as matrix (2 × D)
    task_matrix = torch.stack([vec1, vec2], dim=0)

    # Compute SVD
    try:
        _, S, _ = torch.linalg.svd(task_matrix, full_matrices=False)
    except Exception:
        return 1.0

    if len(S) < 2:
        return 0.0  # Only one singular value means perfect alignment

    # Ratio = σ_2 / σ_1
    ratio = S[1] / (S[0] + 1e-10)

    return ratio.item()


def layerwise_effective_rank(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute weighted average effective rank across all layers.

    This computes the effective rank for each layer separately, then
    takes a weighted average based on the magnitude of updates in each layer.
    This provides more granular insight than global effective rank.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Weighted average effective rank across layers.
    """
    layers_1 = get_layer_vectors(task_dict_1)
    layers_2 = get_layer_vectors(task_dict_2)

    common_keys = set(layers_1.keys()) & set(layers_2.keys())

    layer_ranks = []
    layer_weights = []

    for key in sorted(common_keys):
        delta_A = layers_1[key]
        delta_B = layers_2[key]

        # Skip if no updates
        if delta_A.norm() < 1e-10 or delta_B.norm() < 1e-10:
            continue

        # Stack and compute SVD
        layer_matrix = torch.stack([delta_A, delta_B])

        try:
            _, S, _ = torch.linalg.svd(layer_matrix, full_matrices=False)
        except Exception:
            continue

        # Effective rank for this layer
        S_norm = S / (S.sum() + 1e-10)
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
        eff_rank = torch.exp(entropy).item()

        # Weight by total update magnitude
        weight = (delta_A.norm() + delta_B.norm()).item()

        layer_ranks.append(eff_rank)
        layer_weights.append(weight)

    if not layer_ranks:
        return 2.0  # Return worst case if no valid layers

    # Weighted average
    total_weight = sum(layer_weights)
    weighted_avg = sum(r * w for r, w in zip(layer_ranks, layer_weights)) / total_weight

    return weighted_avg


def layerwise_effective_rank_mergeability_score(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute mergeability score from layerwise effective rank (mapped to [0, 1]).

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Mergeability score in [0, 1], where higher is better.
    """
    eff_rank = layerwise_effective_rank(task_dict_1, task_dict_2)

    # Map [1, 2] to [1, 0]
    score = 2.0 - eff_rank
    score = max(0.0, min(1.0, score))

    return score


def singular_value_overlap(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 100,
) -> float:
    """Compute overlap of top-k singular values across weight matrices.

    This measures whether the two task vectors modify similar "directions"
    of the weight matrices in terms of their singular value structure.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        top_k: Number of top singular values to consider per matrix.

    Returns:
        Average overlap coefficient across all 2D weight matrices.
    """
    overlaps = []

    for key in sorted(task_dict_1.keys()):
        if key not in task_dict_2:
            continue

        tensor1 = task_dict_1[key]
        tensor2 = task_dict_2[key]

        # Only process 2D matrices
        if tensor1.dim() != 2:
            continue

        # Compute SVD for both
        try:
            _, s1, _ = torch.linalg.svd(tensor1.float(), full_matrices=False)
            _, s2, _ = torch.linalg.svd(tensor2.float(), full_matrices=False)
        except Exception:
            continue

        # Normalize singular values
        s1 = s1[:top_k] / (s1.sum() + 1e-8)
        s2 = s2[:top_k] / (s2.sum() + 1e-8)

        # Pad to same length if needed
        max_len = max(len(s1), len(s2))
        if len(s1) < max_len:
            s1 = F.pad(s1, (0, max_len - len(s1)))
        if len(s2) < max_len:
            s2 = F.pad(s2, (0, max_len - len(s2)))

        # Compute overlap as cosine similarity of normalized singular value distributions
        overlap = F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item()
        overlaps.append(overlap)

    if not overlaps:
        return 0.0

    return sum(overlaps) / len(overlaps)


def subspace_overlap(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> float:
    """Compute principal left subspace overlap between task vectors.

    Measures how much the principal left directions (from SVD, using U matrices)
    of the two task vectors overlap. High overlap might indicate task compatibility.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        top_k: Number of top principal directions to consider.

    Returns:
        Average left subspace overlap across all 2D weight matrices.
    """
    overlaps = []

    for key in sorted(task_dict_1.keys()):
        if key not in task_dict_2:
            continue

        tensor1 = task_dict_1[key]
        tensor2 = task_dict_2[key]

        # Only process 2D matrices
        if tensor1.dim() != 2:
            continue

        # Compute SVD for both
        try:
            u1, _, _ = torch.linalg.svd(tensor1.float(), full_matrices=False)
            u2, _, _ = torch.linalg.svd(tensor2.float(), full_matrices=False)
        except Exception:
            continue

        # Take top-k columns of U matrices
        k = min(top_k, u1.shape[1], u2.shape[1])
        u1_k = u1[:, :k]
        u2_k = u2[:, :k]

        # Compute subspace overlap using Frobenius norm of U1^T @ U2
        # Maximum overlap is sqrt(k) when subspaces are identical
        product = u1_k.T @ u2_k
        overlap = torch.norm(product, p='fro').item() / k
        overlaps.append(overlap)

    if not overlaps:
        return 0.0

    return sum(overlaps) / len(overlaps)


def right_subspace_overlap(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> Tuple[float, float]:
    """Compute principal right subspace overlap between task vectors for both top-k and bottom-k.

    Measures how much the principal right directions (from SVD, using V matrices)
    of the two task vectors overlap. High overlap might indicate task compatibility.
    This metric computes overlap for both the strongest (top-k) and weakest (bottom-k)
    singular vectors.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        top_k: Number of principal directions to consider from top and bottom.

    Returns:
        Tuple of (top_k_overlap, bottom_k_overlap):
        - top_k_overlap: Average right subspace overlap using top-k singular vectors
        - bottom_k_overlap: Average right subspace overlap using bottom-k singular vectors
    """
    top_overlaps = []
    bottom_overlaps = []

    for key in sorted(task_dict_1.keys()):
        if key not in task_dict_2:
            continue

        tensor1 = task_dict_1[key]
        tensor2 = task_dict_2[key]

        # Only process 2D matrices
        if tensor1.dim() != 2:
            continue

        # Compute SVD for both
        try:
            _, s1, v1 = torch.linalg.svd(tensor1.float(), full_matrices=False)
            _, s2, v2 = torch.linalg.svd(tensor2.float(), full_matrices=False)
        except Exception:
            continue

        # Determine k for this layer
        k = min(top_k, v1.shape[0], v2.shape[0])

        # Top-k: Take first k rows of V matrices (V is returned as V^H in torch.linalg.svd)
        v1_top_k = v1[:k, :]
        v2_top_k = v2[:k, :]

        # Compute top-k subspace overlap using Frobenius norm of V1 @ V2^T
        # Maximum overlap is sqrt(k) when subspaces are identical
        product_top = v1_top_k @ v2_top_k.T
        overlap_top = torch.norm(product_top, p='fro').item() / (k ** 0.5)
        top_overlaps.append(overlap_top)

        # Bottom-k: Take last k rows of V matrices (weakest singular vectors)
        v1_bottom_k = v1[-k:, :]
        v2_bottom_k = v2[-k:, :]

        # Compute bottom-k subspace overlap
        product_bottom = v1_bottom_k @ v2_bottom_k.T
        overlap_bottom = torch.norm(product_bottom, p='fro').item() / (k ** 0.5)
        bottom_overlaps.append(overlap_bottom)

    if not top_overlaps:
        return 0.0, 0.0

    avg_top = sum(top_overlaps) / len(top_overlaps)
    avg_bottom = sum(bottom_overlaps) / len(bottom_overlaps)

    return avg_top, avg_bottom


def interaction_matrix_overlap(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> Tuple[float, float]:
    """Compute interaction matrix overlap between task vectors for both top-k and bottom-k.

    For each layer, computes the interaction matrix M = V_A^T @ V_B where V_A and V_B
    are the right singular vectors. The singular values of M represent the cosines of
    principal angles between the subspaces. The metric returns the average of squared
    singular values, computed separately for top-k and bottom-k singular vectors.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        top_k: Number of principal directions to consider from top and bottom.

    Returns:
        Tuple of (top_k_overlap, bottom_k_overlap):
        - top_k_overlap: Average of squared singular values using top-k singular vectors
        - bottom_k_overlap: Average of squared singular values using bottom-k singular vectors
    """
    top_overlaps = []
    bottom_overlaps = []

    for key in sorted(task_dict_1.keys()):
        if key not in task_dict_2:
            continue

        tensor1 = task_dict_1[key]
        tensor2 = task_dict_2[key]

        # Only process 2D matrices
        if tensor1.dim() != 2:
            continue

        # Compute SVD for both
        try:
            _, s1, v1 = torch.linalg.svd(tensor1.float(), full_matrices=False)
            _, s2, v2 = torch.linalg.svd(tensor2.float(), full_matrices=False)
        except Exception:
            continue

        # Determine k for this layer
        k = min(top_k, v1.shape[0], v2.shape[0])

        # Top-k: Take first k rows of V matrices
        v1_top_k = v1[:k, :]
        v2_top_k = v2[:k, :]

        # Compute interaction matrix M = V_A^T @ V_B
        # v1_top_k has shape (k, n), v2_top_k has shape (k, n)
        # M = v1_top_k @ v2_top_k.T has shape (k, k)
        interaction_matrix_top = v1_top_k @ v2_top_k.T

        # Compute SVD on interaction matrix to get singular values
        try:
            _, sigma_top, _ = torch.linalg.svd(interaction_matrix_top, full_matrices=False)
        except Exception:
            continue

        # Average of squared singular values
        overlap_top = torch.mean(sigma_top ** 2).item()
        top_overlaps.append(overlap_top)

        # Bottom-k: Take last k rows of V matrices (weakest singular vectors)
        v1_bottom_k = v1[-k:, :]
        v2_bottom_k = v2[-k:, :]

        # Compute interaction matrix for bottom-k
        interaction_matrix_bottom = v1_bottom_k @ v2_bottom_k.T

        # Compute SVD on interaction matrix
        try:
            _, sigma_bottom, _ = torch.linalg.svd(interaction_matrix_bottom, full_matrices=False)
        except Exception:
            continue

        # Average of squared singular values
        overlap_bottom = torch.mean(sigma_bottom ** 2).item()
        bottom_overlaps.append(overlap_bottom)

    if not top_overlaps:
        return 0.0, 0.0

    avg_top = sum(top_overlaps) / len(top_overlaps)
    avg_bottom = sum(bottom_overlaps) / len(bottom_overlaps)

    return avg_top, avg_bottom


# =============================================================================
# Activation-Based Metrics Infrastructure
# =============================================================================


def build_calibration_loader(
    dataset_configs: List[Dict],
    pretrained_encoder,
    n_samples: int = 10,
    batch_size: int = 32,
    device: str = "cuda",
    random_seed: int = 42,
) -> DataLoader:
    """Build a calibration data loader from multiple datasets.

    Args:
        dataset_configs: List of dataset config dictionaries (from Hydra)
        pretrained_encoder: The pretrained encoder model to get preprocessor
        n_samples: Number of samples to take from each dataset's validation set
        batch_size: Batch size for the calibration loader
        device: Device to use
        random_seed: Random seed for reproducible sampling

    Returns:
        DataLoader containing calibration samples from all datasets
    """
    from model_merging.data.dataset import load_dataset
    from hydra.utils import instantiate
    import random

    all_samples = []
    preprocess_fn = pretrained_encoder.val_preprocess

    # Set random seed for reproducibility
    random.seed(random_seed)

    for dataset_cfg in dataset_configs:
        try:
            # Instantiate the HF dataset using Hydra
            hf_dataset = instantiate(dataset_cfg.hf_dataset)

            # Load the dataset
            dataset = load_dataset(
                name=dataset_cfg.name,
                hf_dataset=hf_dataset,
                preprocess_fn=preprocess_fn,
                ft_epochs=dataset_cfg.get("ft_epochs", 10),
                split_map=dataset_cfg.get("split_map", None),
                batch_size=batch_size,
                label_map=dataset_cfg.get("label_map", None),
                classnames_override=dataset_cfg.get("classnames_override", None),
            )

            # Sample n_samples from validation/test set randomly
            test_dataset = dataset.test_dataset
            n_available = len(test_dataset)
            n_to_sample = min(n_samples, n_available)

            # Random sampling with fixed seed for reproducibility
            indices = random.sample(range(n_available), n_to_sample)
            indices.sort()  # Sort for consistent ordering

            for idx in indices:
                all_samples.append(test_dataset[idx])

        except Exception as e:
            print(f"Warning: Failed to load dataset {dataset_cfg.get('name', 'unknown')}: {e}")
            continue

    if not all_samples:
        raise ValueError("No samples could be loaded from any dataset")

    # Create a simple Dataset wrapper
    class CalibrationDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    calibration_dataset = CalibrationDataset(all_samples)
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return calibration_loader


def reconstruct_model_from_task_dict(
    pretrained_model,
    task_dict: Dict[str, torch.Tensor],
    coefficient: float = 1.0,
    device: str = "cuda",
):
    """Reconstruct a finetuned model from pretrained model + task vector.

    Args:
        pretrained_model: The pretrained model
        task_dict: Task vector (finetuned - pretrained weights)
        coefficient: Scaling coefficient for task vector
        device: Device to use

    Returns:
        Reconstructed model on the specified device
    """
    # Import here to avoid circular imports
    from model_merging.utils.utils import apply_dict_to_model

    # Create a deep copy of the pretrained model
    model = copy.deepcopy(pretrained_model)
    model = model.to(device)

    # Apply task vector to get finetuned model
    model = apply_dict_to_model(task_dict, model, coefficient=coefficient)
    model.eval()

    return model


def extract_layer_activations(
    model,
    calibration_loader: DataLoader,
    layer_name: str,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract and average activations from a specific layer over calibration data.

    Args:
        model: The model to extract activations from
        calibration_loader: DataLoader containing calibration samples
        layer_name: Name of the layer to extract activations from
        device: Device to use

    Returns:
        Averaged activation tensor across all calibration samples
    """
    # Import here to avoid circular imports
    from model_merging.utils.utils import get_hook_fn

    model = model.to(device)
    model.eval()

    # Initialize storage for intermediate features
    model.middle_features = {}

    # Find the target module
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Layer {layer_name} not found in model")

    # Register hook
    hook_fn = get_hook_fn(model, layer_name, input_or_output="output")
    handle = target_module.register_forward_hook(hook_fn)

    # Collect activations
    all_activations = []

    with torch.no_grad():
        for batch in calibration_loader:
            images, _ = batch
            images = images.to(device)

            # Forward pass
            _ = model(images)

            # Get activation from this batch
            activation = model.middle_features[layer_name]

            # Average over batch dimension and flatten spatial dimensions if needed
            # Shape is typically (B, seq_len, hidden_dim) or (B, hidden_dim)
            if activation.dim() == 3:
                # Average over sequence dimension: (B, seq_len, hidden_dim) -> (B, hidden_dim)
                activation = activation.mean(dim=1)

            all_activations.append(activation.cpu())

    # Remove hook
    handle.remove()

    # Concatenate all batches and compute average
    all_activations = torch.cat(all_activations, dim=0)  # (N_total, hidden_dim)
    avg_activation = all_activations.mean(dim=0)  # (hidden_dim,)

    return avg_activation


# =============================================================================
# Activation-Based Metrics
# =============================================================================


def activation_l2_distance(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    calibration_loader: Optional[DataLoader] = None,
    layer_name: Optional[str] = None,
    device: str = "cuda",
) -> float:
    """Compute L2 distance between average activations of two models.

    This measures how different the internal representations are between
    two finetuned models on the same calibration data.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required for activation metrics)
        calibration_loader: DataLoader with calibration samples (required)
        layer_name: Name of layer to extract activations from (required)
        device: Device to use

    Returns:
        L2 distance between averaged activations
    """
    if pretrained_model is None or calibration_loader is None or layer_name is None:
        raise ValueError(
            "Activation metrics require pretrained_model, calibration_loader, and layer_name"
        )

    # Reconstruct models
    model_1 = reconstruct_model_from_task_dict(pretrained_model, task_dict_1, device=device)
    model_2 = reconstruct_model_from_task_dict(pretrained_model, task_dict_2, device=device)

    # Extract activations
    act_1 = extract_layer_activations(model_1, calibration_loader, layer_name, device)
    act_2 = extract_layer_activations(model_2, calibration_loader, layer_name, device)

    # Compute L2 distance
    distance = torch.norm(act_1 - act_2, p=2).item()

    # Clean up
    del model_1, model_2
    torch.cuda.empty_cache()

    return distance


def activation_cosine_similarity(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    calibration_loader: Optional[DataLoader] = None,
    layer_name: Optional[str] = None,
    device: str = "cuda",
) -> float:
    """Compute cosine similarity between average activations of two models.

    This measures how aligned the internal representations are between
    two finetuned models in terms of direction.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required for activation metrics)
        calibration_loader: DataLoader with calibration samples (required)
        layer_name: Name of layer to extract activations from (required)
        device: Device to use

    Returns:
        Cosine similarity value in [-1, 1]
    """
    if pretrained_model is None or calibration_loader is None or layer_name is None:
        raise ValueError(
            "Activation metrics require pretrained_model, calibration_loader, and layer_name"
        )

    # Reconstruct models
    model_1 = reconstruct_model_from_task_dict(pretrained_model, task_dict_1, device=device)
    model_2 = reconstruct_model_from_task_dict(pretrained_model, task_dict_2, device=device)

    # Extract activations
    act_1 = extract_layer_activations(model_1, calibration_loader, layer_name, device)
    act_2 = extract_layer_activations(model_2, calibration_loader, layer_name, device)

    # Compute cosine similarity
    similarity = F.cosine_similarity(act_1.unsqueeze(0), act_2.unsqueeze(0)).item()

    # Clean up
    del model_1, model_2
    torch.cuda.empty_cache()

    return similarity


def activation_magnitude_ratio(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    calibration_loader: Optional[DataLoader] = None,
    layer_name: Optional[str] = None,
    device: str = "cuda",
) -> float:
    """Compute ratio of activation magnitudes between two models.

    This measures whether one model produces much stronger or weaker
    activations than another, which could indicate different learning scales.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required for activation metrics)
        calibration_loader: DataLoader with calibration samples (required)
        layer_name: Name of layer to extract activations from (required)
        device: Device to use

    Returns:
        Magnitude ratio (smaller / larger) in (0, 1]
    """
    if pretrained_model is None or calibration_loader is None or layer_name is None:
        raise ValueError(
            "Activation metrics require pretrained_model, calibration_loader, and layer_name"
        )

    # Reconstruct models
    model_1 = reconstruct_model_from_task_dict(pretrained_model, task_dict_1, device=device)
    model_2 = reconstruct_model_from_task_dict(pretrained_model, task_dict_2, device=device)

    # Extract activations
    act_1 = extract_layer_activations(model_1, calibration_loader, layer_name, device)
    act_2 = extract_layer_activations(model_2, calibration_loader, layer_name, device)

    # Compute magnitudes
    mag_1 = torch.norm(act_1, p=2).item()
    mag_2 = torch.norm(act_2, p=2).item()

    # Compute ratio (smaller / larger)
    if mag_1 < 1e-8 or mag_2 < 1e-8:
        ratio = 0.0
    else:
        ratio = min(mag_1, mag_2) / max(mag_1, mag_2)

    # Clean up
    del model_1, model_2
    torch.cuda.empty_cache()

    return ratio


def activation_dot_product(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    calibration_loader: Optional[DataLoader] = None,
    layer_name: Optional[str] = None,
    device: str = "cuda",
) -> float:
    """Compute dot product between average activations of two models.

    Unlike cosine similarity, this captures both direction and magnitude
    of the activation alignment.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required for activation metrics)
        calibration_loader: DataLoader with calibration samples (required)
        layer_name: Name of layer to extract activations from (required)
        device: Device to use

    Returns:
        Dot product value
    """
    if pretrained_model is None or calibration_loader is None or layer_name is None:
        raise ValueError(
            "Activation metrics require pretrained_model, calibration_loader, and layer_name"
        )

    # Reconstruct models
    model_1 = reconstruct_model_from_task_dict(pretrained_model, task_dict_1, device=device)
    model_2 = reconstruct_model_from_task_dict(pretrained_model, task_dict_2, device=device)

    # Extract activations
    act_1 = extract_layer_activations(model_1, calibration_loader, layer_name, device)
    act_2 = extract_layer_activations(model_2, calibration_loader, layer_name, device)

    # Compute dot product
    dot_prod = torch.dot(act_1, act_2).item()

    # Clean up
    del model_1, model_2
    torch.cuda.empty_cache()

    return dot_prod


# =============================================================================
# Gradient-based Metrics
# =============================================================================


def reconstruct_classifier_from_task_dict(
    pretrained_model,
    task_dict: Dict[str, torch.Tensor],
    num_classes: int,
    coefficient: float = 1.0,
    device: str = "cuda",
):
    """Reconstruct a full ImageClassifier from pretrained encoder + task vector.

    Args:
        pretrained_model: The pretrained encoder (ImageEncoder)
        task_dict: Task vector containing both encoder and classification_head params
        num_classes: Number of output classes for the classification head
        coefficient: Scaling coefficient for task vector
        device: Device to use

    Returns:
        ImageClassifier with reconstructed encoder and classification head
    """
    from model_merging.model.image_classifier import ImageClassifier
    from model_merging.model.encoder import ClassificationHead
    from model_merging.utils.utils import apply_dict_to_model

    # Reconstruct encoder (without keeping on GPU during deepcopy to save memory)
    encoder = reconstruct_model_from_task_dict(
        pretrained_model, task_dict, coefficient=coefficient, device="cpu"
    )
    # Move to target device after reconstruction
    encoder = encoder.to(device)

    # Extract classification head parameters from task_dict
    # The task dict has keys like "classification_head.weight" and "classification_head.bias"
    head_weight_key = "classification_head.weight"
    head_bias_key = "classification_head.bias"

    # Get pretrained encoder state dict to find the original head params
    pretrained_state = pretrained_model.state_dict()

    # Compute finetuned head parameters (pretrained + task_vector * coefficient)
    if head_weight_key in task_dict:
        # Task dict has classification head, reconstruct it
        if head_weight_key in pretrained_state:
            head_weight = pretrained_state[head_weight_key] + coefficient * task_dict[head_weight_key]
        else:
            # Pretrained doesn't have head, use task dict directly
            head_weight = coefficient * task_dict[head_weight_key]

        if head_bias_key in task_dict:
            if head_bias_key in pretrained_state:
                head_bias = pretrained_state[head_bias_key] + coefficient * task_dict[head_bias_key]
            else:
                head_bias = coefficient * task_dict[head_bias_key]
        else:
            head_bias = None

        # Create classification head
        classification_head = ClassificationHead(
            normalize=True,
            weights=head_weight.to(device),
            biases=head_bias.to(device) if head_bias is not None else None,
        )
    else:
        # No classification head in task dict, create a random one
        # (This shouldn't normally happen for gradient metrics)
        embedding_dim = 512  # Default for CLIP ViT-B/32
        classification_head = ClassificationHead(
            normalize=True,
            input_size=embedding_dim,
            num_classes=num_classes,
        )

    # Create ImageClassifier
    classifier = ImageClassifier(
        encoder=encoder,
        classifier=classification_head,
    )
    classifier = classifier.to(device)

    return classifier


def reconstruct_dual_head_classifier(
    pretrained_model,
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    num_classes_1: int,
    num_classes_2: int,
    task_id: int,  # Which task this model represents (0 or 1)
    coefficient: float = 1.0,
    device: str = "cuda",
):
    """Reconstruct a classifier with two heads, but only one encoder is task-specific.

    For gradient comparison, we want both models to process the same inputs.
    Each model has its own encoder (task-specific) but both heads present
    so loss can be computed on samples from both tasks.

    Args:
        pretrained_model: The pretrained encoder (ImageEncoder)
        task_dict_1: Task vector for task 1
        task_dict_2: Task vector for task 2
        num_classes_1: Number of classes for task 1
        num_classes_2: Number of classes for task 2
        task_id: Which task this model represents (0 = task 1, 1 = task 2)
        coefficient: Scaling coefficient for task vector
        device: Device to use

    Returns:
        A model with the task-specific encoder and both classification heads
    """
    from model_merging.model.encoder import ClassificationHead

    # Reconstruct the task-specific encoder
    if task_id == 0:
        encoder = reconstruct_model_from_task_dict(
            pretrained_model, task_dict_1, coefficient=coefficient, device=device
        )
        own_task_dict = task_dict_1
        other_task_dict = task_dict_2
    else:
        encoder = reconstruct_model_from_task_dict(
            pretrained_model, task_dict_2, coefficient=coefficient, device=device
        )
        own_task_dict = task_dict_2
        other_task_dict = task_dict_1

    # Helper function to extract classification head
    def extract_head(task_dict, num_classes, pretrained_state):
        head_weight_key = "classification_head.weight"
        head_bias_key = "classification_head.bias"

        if head_weight_key in task_dict:
            if head_weight_key in pretrained_state:
                head_weight = pretrained_state[head_weight_key] + coefficient * task_dict[head_weight_key]
            else:
                head_weight = coefficient * task_dict[head_weight_key]

            if head_bias_key in task_dict:
                if head_bias_key in pretrained_state:
                    head_bias = pretrained_state[head_bias_key] + coefficient * task_dict[head_bias_key]
                else:
                    head_bias = coefficient * task_dict[head_bias_key]
            else:
                head_bias = None

            return ClassificationHead(
                normalize=True,
                weights=head_weight.to(device),
                biases=head_bias.to(device) if head_bias is not None else None,
            )
        else:
            embedding_dim = 512
            return ClassificationHead(
                normalize=True,
                input_size=embedding_dim,
                num_classes=num_classes,
            )

    pretrained_state = pretrained_model.state_dict()

    # Create both heads
    head_1 = extract_head(task_dict_1, num_classes_1, pretrained_state)
    head_2 = extract_head(task_dict_2, num_classes_2, pretrained_state)

    # Create dual-head model
    class DualHeadModel(torch.nn.Module):
        def __init__(self, encoder, head_1, head_2, task_id):
            super().__init__()
            self.encoder = encoder
            self.classification_head_1 = head_1
            self.classification_head_2 = head_2
            self.task_id = task_id

        def forward(self, images, task_ids):
            """Compute logits using appropriate head based on task_ids."""
            features = self.encoder(images)
            logits_1 = self.classification_head_1(features)
            logits_2 = self.classification_head_2(features)

            # For each sample, use the logits from its corresponding head
            batch_size = images.size(0)
            max_classes = max(logits_1.size(1), logits_2.size(1))
            output_logits = torch.zeros(batch_size, max_classes, device=images.device)

            for i in range(batch_size):
                if task_ids[i] == 0:
                    output_logits[i, :logits_1.size(1)] = logits_1[i]
                else:
                    output_logits[i, :logits_2.size(1)] = logits_2[i]

            return output_logits

    model = DualHeadModel(encoder, head_1, head_2, task_id)
    model = model.to(device)

    return model


def build_unified_calibration_loader(
    dataset_config_1: Dict,
    dataset_config_2: Dict,
    pretrained_encoder,
    n_samples: int = 10,
    batch_size: int = 8,
    device: str = "cuda",
    random_seed: int = 42,
) -> Tuple[DataLoader, int, int]:
    """Build a unified calibration data loader combining samples from two datasets.

    This creates a single loader with samples from both tasks, allowing both models
    to compute gradients on the exact same calibration set for fair comparison.

    Args:
        dataset_config_1: First dataset config dictionary (from Hydra)
        dataset_config_2: Second dataset config dictionary (from Hydra)
        pretrained_encoder: The pretrained encoder model to get preprocessor
        n_samples: Number of samples to take from each dataset's validation set
        batch_size: Batch size for the calibration loader
        device: Device to use
        random_seed: Random seed for reproducible sampling

    Returns:
        Tuple of (unified_calibration_loader, num_classes_1, num_classes_2)
        The loader returns (images, labels, task_ids) where task_ids indicate which task each sample belongs to
    """
    from model_merging.data.dataset import load_dataset
    from hydra.utils import instantiate
    import random

    preprocess_fn = pretrained_encoder.val_preprocess
    random.seed(random_seed)

    all_samples = []
    num_classes_list = []

    for task_id, dataset_cfg in enumerate([dataset_config_1, dataset_config_2]):
        # Instantiate the HF dataset using Hydra
        hf_dataset = instantiate(dataset_cfg.hf_dataset)

        # Load the dataset
        dataset = load_dataset(
            name=dataset_cfg.name,
            hf_dataset=hf_dataset,
            preprocess_fn=preprocess_fn,
            ft_epochs=dataset_cfg.get("ft_epochs", 10),
            split_map=dataset_cfg.get("split_map", None),
            batch_size=batch_size,
            label_map=dataset_cfg.get("label_map", None),
            classnames_override=dataset_cfg.get("classnames_override", None),
        )

        # Get number of classes from dataset
        num_classes = len(dataset.classnames)
        num_classes_list.append(num_classes)

        # Sample n_samples from validation/test set randomly
        test_dataset = dataset.test_dataset
        n_available = len(test_dataset)
        n_to_sample = min(n_samples, n_available)

        # Random sampling with fixed seed for reproducibility
        indices = random.sample(range(n_available), n_to_sample)
        indices.sort()  # Sort for consistent ordering

        # Add samples with task_id marker
        for idx in indices:
            image, label = test_dataset[idx]
            all_samples.append((image, label, task_id))

    # Create a unified dataset wrapper
    class UnifiedCalibrationDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    unified_dataset = UnifiedCalibrationDataset(all_samples)
    unified_loader = DataLoader(
        unified_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep deterministic order
        num_workers=0,  # Important for gradient computation
        pin_memory=True,
    )

    return unified_loader, num_classes_list[0], num_classes_list[1]


def build_pairwise_calibration_loader(
    dataset_config_1: Dict,
    dataset_config_2: Dict,
    pretrained_encoder,
    n_samples: int = 10,
    batch_size: int = 8,
    device: str = "cuda",
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Build separate calibration data loaders for two datasets.

    Args:
        dataset_config_1: First dataset config dictionary (from Hydra)
        dataset_config_2: Second dataset config dictionary (from Hydra)
        pretrained_encoder: The pretrained encoder model to get preprocessor
        n_samples: Number of samples to take from each dataset's validation set
        batch_size: Batch size for the calibration loaders
        device: Device to use
        random_seed: Random seed for reproducible sampling

    Returns:
        Tuple of (calibration_loader_1, calibration_loader_2, num_classes_1, num_classes_2)
    """
    from model_merging.data.dataset import load_dataset
    from hydra.utils import instantiate
    import random

    preprocess_fn = pretrained_encoder.val_preprocess
    random.seed(random_seed)

    loaders = []
    num_classes_list = []
    for dataset_cfg in [dataset_config_1, dataset_config_2]:
        # Instantiate the HF dataset using Hydra
        hf_dataset = instantiate(dataset_cfg.hf_dataset)

        # Load the dataset
        dataset = load_dataset(
            name=dataset_cfg.name,
            hf_dataset=hf_dataset,
            preprocess_fn=preprocess_fn,
            ft_epochs=dataset_cfg.get("ft_epochs", 10),
            split_map=dataset_cfg.get("split_map", None),
            batch_size=batch_size,
            label_map=dataset_cfg.get("label_map", None),
            classnames_override=dataset_cfg.get("classnames_override", None),
        )

        # Get number of classes from dataset
        num_classes = len(dataset.classnames)
        num_classes_list.append(num_classes)

        # Sample n_samples from validation/test set randomly
        test_dataset = dataset.test_dataset
        n_available = len(test_dataset)
        n_to_sample = min(n_samples, n_available)

        # Random sampling with fixed seed for reproducibility
        indices = random.sample(range(n_available), n_to_sample)
        indices.sort()  # Sort for consistent ordering

        samples = [test_dataset[idx] for idx in indices]

        # Create a simple Dataset wrapper
        class CalibrationDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        calibration_dataset = CalibrationDataset(samples)
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Important for gradient computation
            pin_memory=True,
        )
        loaders.append(calibration_loader)

    return loaders[0], loaders[1], num_classes_list[0], num_classes_list[1]


def extract_encoder_gradients_unified(
    model,
    unified_calibration_loader: DataLoader,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract encoder gradients from a dual-head model on unified calibration data.

    Both models process the same calibration samples, computing gradients on all samples.

    Args:
        model: The dual-head model with task-specific encoder
        unified_calibration_loader: DataLoader with (images, labels, task_ids)
        device: Device to use

    Returns:
        Flattened tensor containing averaged encoder gradients
    """
    model.to(device)
    model.train()

    # Accumulate gradients across all batches
    encoder_grad_sum = None
    n_samples = 0

    for batch in unified_calibration_loader:
        images, labels, task_ids = batch
        images = images.to(device)
        labels = labels.to(device)
        task_ids = task_ids.to(device)

        # Zero gradients
        model.zero_grad()

        # Forward pass
        logits = model(images, task_ids)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()

        # Accumulate encoder gradients (exclude classification heads)
        if encoder_grad_sum is None:
            encoder_grad_sum = []
            for name, param in model.named_parameters():
                # Include encoder parameters only
                if param.grad is not None and "classification_head" not in name:
                    encoder_grad_sum.append(param.grad.detach().clone())
        else:
            idx = 0
            for name, param in model.named_parameters():
                if param.grad is not None and "classification_head" not in name:
                    encoder_grad_sum[idx] += param.grad.detach().clone()
                    idx += 1

        n_samples += images.size(0)

    if encoder_grad_sum is None or len(encoder_grad_sum) == 0:
        raise ValueError("No encoder gradients found. Check model structure.")

    # Flatten and average gradients
    flat_grads = torch.cat([g.flatten() for g in encoder_grad_sum])
    flat_grads = flat_grads / n_samples

    return flat_grads


def extract_encoder_gradients(
    model,
    calibration_loader: DataLoader,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract encoder gradients from a model on calibration data.

    Computes the average gradient of the encoder parameters w.r.t. the
    cross-entropy loss on the calibration samples.

    Args:
        model: The model (ImageClassifier with encoder and classification_head attributes)
        calibration_loader: DataLoader with calibration samples
        device: Device to use

    Returns:
        Flattened tensor containing averaged encoder gradients
    """
    model.to(device)
    model.train()  # Enable gradient computation

    # Accumulate gradients across all batches
    encoder_grad_sum = None
    n_samples = 0

    for batch in calibration_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        model.zero_grad()

        # Forward pass
        logits = model(images)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()

        # Accumulate encoder gradients (exclude classification head)
        if encoder_grad_sum is None:
            encoder_grad_sum = []
            for name, param in model.named_parameters():
                # Include all parameters EXCEPT classification_head
                if param.grad is not None and "classification_head" not in name:
                    encoder_grad_sum.append(param.grad.detach().clone())
        else:
            idx = 0
            for name, param in model.named_parameters():
                if param.grad is not None and "classification_head" not in name:
                    encoder_grad_sum[idx] += param.grad.detach().clone()
                    idx += 1

        n_samples += images.size(0)

    if encoder_grad_sum is None or len(encoder_grad_sum) == 0:
        raise ValueError("No encoder gradients found. Check model structure.")

    # Flatten and average gradients
    flat_grads = torch.cat([g.flatten() for g in encoder_grad_sum])
    flat_grads = flat_grads / n_samples

    return flat_grads


def extract_input_gradients(
    model,
    calibration_loader: DataLoader,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract input gradients from a model on calibration data.

    Computes the average gradient of the loss w.r.t. the input images
    on the calibration samples.

    Args:
        model: The model (should have encoder and classification_head attributes)
        calibration_loader: DataLoader with calibration samples
        device: Device to use

    Returns:
        Flattened tensor containing averaged input gradients
    """
    model.to(device)
    model.train()  # Enable gradient computation

    all_input_grads = []

    for batch in calibration_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Enable gradient computation for inputs
        images.requires_grad_(True)

        # Zero gradients
        model.zero_grad()

        # Forward pass
        logits = model(images)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        # Backward pass to get input gradients
        loss.backward()

        # Extract input gradients
        if images.grad is not None:
            all_input_grads.append(images.grad.detach().clone().flatten())

    if not all_input_grads:
        raise ValueError("No input gradients found.")

    # Concatenate all input gradients and compute average
    flat_input_grads = torch.cat(all_input_grads)

    return flat_input_grads


def encoder_gradient_cosine_similarity(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    dataset_config_1: Optional[Dict] = None,
    dataset_config_2: Optional[Dict] = None,
    n_calibration_samples: int = 10,
    calibration_batch_size: int = 8,
    calibration_random_seed: int = 42,
    device: str = "cuda",
) -> float:
    """Compute cosine similarity between encoder gradients of two models.

    This measures how similarly the two models' encoders respond to
    gradient updates on their respective tasks.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required)
        dataset_config_1: Config for first dataset (required)
        dataset_config_2: Config for second dataset (required)
        n_calibration_samples: Number of samples per dataset
        calibration_batch_size: Batch size for gradient computation
        calibration_random_seed: Random seed for sampling
        device: Device to use

    Returns:
        Cosine similarity between encoder gradients in [-1, 1]
    """
    if pretrained_model is None or dataset_config_1 is None or dataset_config_2 is None:
        raise ValueError(
            "Encoder gradient metrics require pretrained_model, dataset_config_1, and dataset_config_2"
        )

    # Build pairwise calibration loaders
    cal_loader_1, cal_loader_2, num_classes_1, num_classes_2 = build_pairwise_calibration_loader(
        dataset_config_1,
        dataset_config_2,
        pretrained_model,
        n_samples=n_calibration_samples,
        batch_size=calibration_batch_size,
        device=device,
        random_seed=calibration_random_seed,
    )

    # Reconstruct full classifiers (encoder + classification head)
    model_1 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_1, num_classes_1, device=device
    )
    model_2 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_2, num_classes_2, device=device
    )

    # Extract encoder gradients
    grads_1 = extract_encoder_gradients(model_1, cal_loader_1, device)
    grads_2 = extract_encoder_gradients(model_2, cal_loader_2, device)

    # Compute cosine similarity
    similarity = F.cosine_similarity(grads_1.unsqueeze(0), grads_2.unsqueeze(0)).item()

    # Clean up
    del model_1, model_2, grads_1, grads_2, cal_loader_1, cal_loader_2
    torch.cuda.empty_cache()

    return similarity


def encoder_gradient_l2_distance(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    dataset_config_1: Optional[Dict] = None,
    dataset_config_2: Optional[Dict] = None,
    n_calibration_samples: int = 10,
    calibration_batch_size: int = 8,
    calibration_random_seed: int = 42,
    device: str = "cuda",
) -> float:
    """Compute L2 distance between encoder gradients of two models.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required)
        dataset_config_1: Config for first dataset (required)
        dataset_config_2: Config for second dataset (required)
        n_calibration_samples: Number of samples per dataset
        calibration_batch_size: Batch size for gradient computation
        calibration_random_seed: Random seed for sampling
        device: Device to use

    Returns:
        L2 distance between encoder gradients
    """
    if pretrained_model is None or dataset_config_1 is None or dataset_config_2 is None:
        raise ValueError(
            "Encoder gradient metrics require pretrained_model, dataset_config_1, and dataset_config_2"
        )

    # Build pairwise calibration loaders
    cal_loader_1, cal_loader_2, num_classes_1, num_classes_2 = build_pairwise_calibration_loader(
        dataset_config_1,
        dataset_config_2,
        pretrained_model,
        n_samples=n_calibration_samples,
        batch_size=calibration_batch_size,
        device=device,
        random_seed=calibration_random_seed,
    )

    # Reconstruct models
    model_1 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_1, num_classes_1, device=device
    )
    model_2 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_2, num_classes_2, device=device
    )

    # Extract encoder gradients
    grads_1 = extract_encoder_gradients(model_1, cal_loader_1, device)
    grads_2 = extract_encoder_gradients(model_2, cal_loader_2, device)

    # Compute L2 distance
    distance = torch.norm(grads_1 - grads_2, p=2).item()

    # Clean up
    del model_1, model_2, grads_1, grads_2, cal_loader_1, cal_loader_2
    torch.cuda.empty_cache()

    return distance


def encoder_gradient_dot_product(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    dataset_config_1: Optional[Dict] = None,
    dataset_config_2: Optional[Dict] = None,
    n_calibration_samples: int = 10,
    calibration_batch_size: int = 8,
    calibration_random_seed: int = 42,
    device: str = "cuda",
) -> float:
    """Compute dot product between encoder gradients of two models.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required)
        dataset_config_1: Config for first dataset (required)
        dataset_config_2: Config for second dataset (required)
        n_calibration_samples: Number of samples per dataset
        calibration_batch_size: Batch size for gradient computation
        calibration_random_seed: Random seed for sampling
        device: Device to use

    Returns:
        Dot product between encoder gradients
    """
    if pretrained_model is None or dataset_config_1 is None or dataset_config_2 is None:
        raise ValueError(
            "Encoder gradient metrics require pretrained_model, dataset_config_1, and dataset_config_2"
        )

    # Build pairwise calibration loaders
    cal_loader_1, cal_loader_2, num_classes_1, num_classes_2 = build_pairwise_calibration_loader(
        dataset_config_1,
        dataset_config_2,
        pretrained_model,
        n_samples=n_calibration_samples,
        batch_size=calibration_batch_size,
        device=device,
        random_seed=calibration_random_seed,
    )

    # Reconstruct models
    model_1 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_1, num_classes_1, device=device
    )
    model_2 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_2, num_classes_2, device=device
    )

    # Extract encoder gradients
    grads_1 = extract_encoder_gradients(model_1, cal_loader_1, device)
    grads_2 = extract_encoder_gradients(model_2, cal_loader_2, device)

    # Compute dot product
    dot_prod = torch.dot(grads_1, grads_2).item()

    # Clean up
    del model_1, model_2, grads_1, grads_2, cal_loader_1, cal_loader_2
    torch.cuda.empty_cache()

    return dot_prod


def input_gradient_cosine_similarity(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    dataset_config_1: Optional[Dict] = None,
    dataset_config_2: Optional[Dict] = None,
    n_calibration_samples: int = 10,
    calibration_batch_size: int = 8,
    calibration_random_seed: int = 42,
    device: str = "cuda",
) -> float:
    """Compute cosine similarity between input gradients of two models.

    This measures how similarly the two models respond to input perturbations
    on their respective tasks.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required)
        dataset_config_1: Config for first dataset (required)
        dataset_config_2: Config for second dataset (required)
        n_calibration_samples: Number of samples per dataset
        calibration_batch_size: Batch size for gradient computation
        calibration_random_seed: Random seed for sampling
        device: Device to use

    Returns:
        Cosine similarity between input gradients in [-1, 1]
    """
    if pretrained_model is None or dataset_config_1 is None or dataset_config_2 is None:
        raise ValueError(
            "Input gradient metrics require pretrained_model, dataset_config_1, and dataset_config_2"
        )

    # Build pairwise calibration loaders
    cal_loader_1, cal_loader_2, num_classes_1, num_classes_2 = build_pairwise_calibration_loader(
        dataset_config_1,
        dataset_config_2,
        pretrained_model,
        n_samples=n_calibration_samples,
        batch_size=calibration_batch_size,
        device=device,
        random_seed=calibration_random_seed,
    )

    # Reconstruct models
    model_1 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_1, num_classes_1, device=device
    )
    model_2 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_2, num_classes_2, device=device
    )

    # Extract input gradients
    grads_1 = extract_input_gradients(model_1, cal_loader_1, device)
    grads_2 = extract_input_gradients(model_2, cal_loader_2, device)

    # Compute cosine similarity
    similarity = F.cosine_similarity(grads_1.unsqueeze(0), grads_2.unsqueeze(0)).item()

    # Clean up
    del model_1, model_2, grads_1, grads_2, cal_loader_1, cal_loader_2
    torch.cuda.empty_cache()

    return similarity


def input_gradient_l2_distance(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    dataset_config_1: Optional[Dict] = None,
    dataset_config_2: Optional[Dict] = None,
    n_calibration_samples: int = 10,
    calibration_batch_size: int = 8,
    calibration_random_seed: int = 42,
    device: str = "cuda",
) -> float:
    """Compute L2 distance between input gradients of two models.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required)
        dataset_config_1: Config for first dataset (required)
        dataset_config_2: Config for second dataset (required)
        n_calibration_samples: Number of samples per dataset
        calibration_batch_size: Batch size for gradient computation
        calibration_random_seed: Random seed for sampling
        device: Device to use

    Returns:
        L2 distance between input gradients
    """
    if pretrained_model is None or dataset_config_1 is None or dataset_config_2 is None:
        raise ValueError(
            "Input gradient metrics require pretrained_model, dataset_config_1, and dataset_config_2"
        )

    # Build pairwise calibration loaders
    cal_loader_1, cal_loader_2, num_classes_1, num_classes_2 = build_pairwise_calibration_loader(
        dataset_config_1,
        dataset_config_2,
        pretrained_model,
        n_samples=n_calibration_samples,
        batch_size=calibration_batch_size,
        device=device,
        random_seed=calibration_random_seed,
    )

    # Reconstruct models
    model_1 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_1, num_classes_1, device=device
    )
    model_2 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_2, num_classes_2, device=device
    )

    # Extract input gradients
    grads_1 = extract_input_gradients(model_1, cal_loader_1, device)
    grads_2 = extract_input_gradients(model_2, cal_loader_2, device)

    # Compute L2 distance
    distance = torch.norm(grads_1 - grads_2, p=2).item()

    # Clean up
    del model_1, model_2, grads_1, grads_2, cal_loader_1, cal_loader_2
    torch.cuda.empty_cache()

    return distance


def input_gradient_dot_product(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    dataset_config_1: Optional[Dict] = None,
    dataset_config_2: Optional[Dict] = None,
    n_calibration_samples: int = 10,
    calibration_batch_size: int = 8,
    calibration_random_seed: int = 42,
    device: str = "cuda",
) -> float:
    """Compute dot product between input gradients of two models.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required)
        dataset_config_1: Config for first dataset (required)
        dataset_config_2: Config for second dataset (required)
        n_calibration_samples: Number of samples per dataset
        calibration_batch_size: Batch size for gradient computation
        calibration_random_seed: Random seed for sampling
        device: Device to use

    Returns:
        Dot product between input gradients
    """
    if pretrained_model is None or dataset_config_1 is None or dataset_config_2 is None:
        raise ValueError(
            "Input gradient metrics require pretrained_model, dataset_config_1, and dataset_config_2"
        )

    # Build pairwise calibration loaders
    cal_loader_1, cal_loader_2, num_classes_1, num_classes_2 = build_pairwise_calibration_loader(
        dataset_config_1,
        dataset_config_2,
        pretrained_model,
        n_samples=n_calibration_samples,
        batch_size=calibration_batch_size,
        device=device,
        random_seed=calibration_random_seed,
    )

    # Reconstruct models
    model_1 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_1, num_classes_1, device=device
    )
    model_2 = reconstruct_classifier_from_task_dict(
        pretrained_model, task_dict_2, num_classes_2, device=device
    )

    # Extract input gradients
    grads_1 = extract_input_gradients(model_1, cal_loader_1, device)
    grads_2 = extract_input_gradients(model_2, cal_loader_2, device)

    # Compute dot product
    dot_prod = torch.dot(grads_1, grads_2).item()

    # Clean up
    del model_1, model_2, grads_1, grads_2, cal_loader_1, cal_loader_2
    torch.cuda.empty_cache()

    return dot_prod


# =============================================================================
# Efficient Batched SVD Computation
# =============================================================================


# Define which metrics belong to each SVD group for efficient batch computation
GLOBAL_SVD_METRICS = [
    "effective_rank",
    "effective_rank_mergeability_score",
    "stable_rank",
    "spectral_gap",
    "singular_value_ratio",
]

LAYERWISE_STACKED_SVD_METRICS = [
    "layerwise_effective_rank",
    "layerwise_effective_rank_mergeability_score",
]

PERLAYER_TENSOR_SVD_METRICS = [
    "singular_value_overlap",
    "subspace_overlap",
    "right_subspace_overlap_top_k",
    "right_subspace_overlap_bottom_k",
    "interaction_matrix_overlap_top_k",
    "interaction_matrix_overlap_bottom_k",
]

# All SVD-based metrics combined
ALL_SVD_METRICS = GLOBAL_SVD_METRICS + LAYERWISE_STACKED_SVD_METRICS + PERLAYER_TENSOR_SVD_METRICS


def compute_global_svd_metrics(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute all global SVD-based metrics with a single SVD computation.

    This is more efficient than calling individual metric functions when
    multiple SVD-based metrics are needed. Computes SVD once on the stacked
    task vectors and derives all metrics from the singular values.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Dictionary with keys: 'effective_rank', 'effective_rank_mergeability_score',
        'stable_rank', 'spectral_gap', 'singular_value_ratio'
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    # Stack as matrix (2 × D)
    task_matrix = torch.stack([vec1, vec2], dim=0)

    # Compute SVD once
    try:
        _, S, _ = torch.linalg.svd(task_matrix, full_matrices=False)
    except Exception:
        # Return worst-case values on failure
        return {
            "effective_rank": 2.0,
            "effective_rank_mergeability_score": 0.0,
            "stable_rank": 2.0,
            "spectral_gap": 0.0,
            "singular_value_ratio": 1.0,
        }

    # Compute effective_rank from singular values
    S_normalized = S / (S.sum() + 1e-10)
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    eff_rank = torch.exp(entropy).item()

    # Compute effective_rank_mergeability_score (mapped from [1,2] to [1,0])
    eff_rank_score = max(0.0, min(1.0, 2.0 - eff_rank))

    # Compute stable_rank = (sum of singular values)^2 / sum of squared singular values
    s_rank = ((S.sum() ** 2) / ((S ** 2).sum() + 1e-10)).item()

    # Compute spectral_gap = (σ_1 - σ_2) / σ_1
    if len(S) < 2:
        spec_gap = 1.0
    else:
        spec_gap = ((S[0] - S[1]) / (S[0] + 1e-10)).item()

    # Compute singular_value_ratio = σ_2 / σ_1
    if len(S) < 2:
        sv_ratio = 0.0
    else:
        sv_ratio = (S[1] / (S[0] + 1e-10)).item()

    return {
        "effective_rank": eff_rank,
        "effective_rank_mergeability_score": eff_rank_score,
        "stable_rank": s_rank,
        "spectral_gap": spec_gap,
        "singular_value_ratio": sv_ratio,
    }


def compute_layerwise_stacked_svd_metrics(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute layerwise effective rank metrics with shared SVD per layer.

    For each layer, stacks the two layer vectors and computes SVD once,
    then derives both layerwise_effective_rank and its mergeability score.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Dictionary with keys: 'layerwise_effective_rank',
        'layerwise_effective_rank_mergeability_score'
    """
    layers_1 = get_layer_vectors(task_dict_1)
    layers_2 = get_layer_vectors(task_dict_2)

    common_keys = set(layers_1.keys()) & set(layers_2.keys())

    layer_ranks = []
    layer_weights = []

    for key in sorted(common_keys):
        delta_A = layers_1[key]
        delta_B = layers_2[key]

        # Skip if no updates
        if delta_A.norm() < 1e-10 or delta_B.norm() < 1e-10:
            continue

        # Stack and compute SVD once per layer
        layer_matrix = torch.stack([delta_A, delta_B])

        try:
            _, S, _ = torch.linalg.svd(layer_matrix, full_matrices=False)
        except Exception:
            continue

        # Effective rank for this layer
        S_norm = S / (S.sum() + 1e-10)
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
        eff_rank = torch.exp(entropy).item()

        # Weight by total update magnitude
        weight = (delta_A.norm() + delta_B.norm()).item()

        layer_ranks.append(eff_rank)
        layer_weights.append(weight)

    if not layer_ranks:
        return {
            "layerwise_effective_rank": 2.0,
            "layerwise_effective_rank_mergeability_score": 0.0,
        }

    # Weighted average
    total_weight = sum(layer_weights)
    weighted_avg = sum(r * w for r, w in zip(layer_ranks, layer_weights)) / total_weight

    # Map to mergeability score
    score = max(0.0, min(1.0, 2.0 - weighted_avg))

    return {
        "layerwise_effective_rank": weighted_avg,
        "layerwise_effective_rank_mergeability_score": score,
    }


def compute_perlayer_tensor_svd_metrics(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    sv_overlap_top_k: int = 100,
    subspace_top_k: int = 10,
) -> Dict[str, float]:
    """Compute all per-layer tensor SVD metrics with shared SVD computations.

    For each 2D layer, computes SVD once per tensor (not stacked) and reuses
    U, S, V matrices across all metrics that need them.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        sv_overlap_top_k: Number of singular values for singular_value_overlap (default 100).
        subspace_top_k: Number of vectors for subspace overlap metrics (default 10).

    Returns:
        Dictionary with keys: 'singular_value_overlap', 'subspace_overlap',
        'right_subspace_overlap_top_k', 'right_subspace_overlap_bottom_k',
        'interaction_matrix_overlap_top_k', 'interaction_matrix_overlap_bottom_k'
    """
    # Storage for per-layer results
    sv_overlaps = []
    left_subspace_overlaps = []
    right_top_overlaps = []
    right_bottom_overlaps = []
    interaction_top_overlaps = []
    interaction_bottom_overlaps = []

    for key in sorted(task_dict_1.keys()):
        if key not in task_dict_2:
            continue

        tensor1 = task_dict_1[key]
        tensor2 = task_dict_2[key]

        # Only process 2D matrices
        if tensor1.dim() != 2:
            continue

        # Compute SVD for both tensors ONCE per layer
        try:
            u1, s1, v1 = torch.linalg.svd(tensor1.float(), full_matrices=False)
            u2, s2, v2 = torch.linalg.svd(tensor2.float(), full_matrices=False)
        except Exception:
            continue

        # --- singular_value_overlap ---
        # Normalize singular values and compute cosine similarity
        k_sv = min(sv_overlap_top_k, len(s1), len(s2))
        s1_norm = s1[:k_sv] / (s1.sum() + 1e-8)
        s2_norm = s2[:k_sv] / (s2.sum() + 1e-8)

        # Pad to same length if needed
        max_len_sv = max(len(s1_norm), len(s2_norm))
        if len(s1_norm) < max_len_sv:
            s1_norm = F.pad(s1_norm, (0, max_len_sv - len(s1_norm)))
        if len(s2_norm) < max_len_sv:
            s2_norm = F.pad(s2_norm, (0, max_len_sv - len(s2_norm)))

        overlap_sv = F.cosine_similarity(s1_norm.unsqueeze(0), s2_norm.unsqueeze(0)).item()
        sv_overlaps.append(overlap_sv)

        # --- subspace_overlap (left subspace using U matrices) ---
        k_left = min(subspace_top_k, u1.shape[1], u2.shape[1])
        u1_k = u1[:, :k_left]
        u2_k = u2[:, :k_left]
        product_left = u1_k.T @ u2_k
        overlap_left = torch.norm(product_left, p='fro').item() / k_left
        left_subspace_overlaps.append(overlap_left)

        # --- right_subspace_overlap (V matrices, top-k and bottom-k) ---
        k_right = min(subspace_top_k, v1.shape[0], v2.shape[0])

        # Top-k (strongest singular vectors)
        v1_top_k = v1[:k_right, :]
        v2_top_k = v2[:k_right, :]
        product_top = v1_top_k @ v2_top_k.T
        overlap_top = torch.norm(product_top, p='fro').item() / (k_right ** 0.5)
        right_top_overlaps.append(overlap_top)

        # Bottom-k (weakest singular vectors)
        v1_bottom_k = v1[-k_right:, :]
        v2_bottom_k = v2[-k_right:, :]
        product_bottom = v1_bottom_k @ v2_bottom_k.T
        overlap_bottom = torch.norm(product_bottom, p='fro').item() / (k_right ** 0.5)
        right_bottom_overlaps.append(overlap_bottom)

        # --- interaction_matrix_overlap (SVD on V1 @ V2^T) ---
        # Top-k interaction matrix
        interaction_matrix_top = v1_top_k @ v2_top_k.T
        try:
            _, sigma_top, _ = torch.linalg.svd(interaction_matrix_top, full_matrices=False)
            overlap_int_top = torch.mean(sigma_top ** 2).item()
            interaction_top_overlaps.append(overlap_int_top)
        except Exception:
            pass

        # Bottom-k interaction matrix
        interaction_matrix_bottom = v1_bottom_k @ v2_bottom_k.T
        try:
            _, sigma_bottom, _ = torch.linalg.svd(interaction_matrix_bottom, full_matrices=False)
            overlap_int_bottom = torch.mean(sigma_bottom ** 2).item()
            interaction_bottom_overlaps.append(overlap_int_bottom)
        except Exception:
            pass

    # Compute averages
    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "singular_value_overlap": safe_avg(sv_overlaps),
        "subspace_overlap": safe_avg(left_subspace_overlaps),
        "right_subspace_overlap_top_k": safe_avg(right_top_overlaps),
        "right_subspace_overlap_bottom_k": safe_avg(right_bottom_overlaps),
        "interaction_matrix_overlap_top_k": safe_avg(interaction_top_overlaps),
        "interaction_matrix_overlap_bottom_k": safe_avg(interaction_bottom_overlaps),
    }


def compute_all_svd_metrics(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    sv_overlap_top_k: int = 100,
    subspace_top_k: int = 10,
) -> Dict[str, float]:
    """Compute all SVD-based metrics efficiently with minimal SVD computations.

    This function computes all SVD-based metrics by:
    1. Computing global SVD once for effective_rank, stable_rank, spectral_gap, singular_value_ratio
    2. Computing per-layer stacked SVD for layerwise_effective_rank metrics
    3. Computing per-layer tensor SVD (once per tensor) for subspace overlap metrics

    This is significantly more efficient than calling each metric individually
    when multiple SVD metrics are needed.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        sv_overlap_top_k: Number of singular values for singular_value_overlap.
        subspace_top_k: Number of vectors for subspace overlap metrics.

    Returns:
        Dictionary containing all SVD-based metric values.
    """
    results = {}

    # Compute global SVD metrics (1 SVD total)
    global_metrics = compute_global_svd_metrics(task_dict_1, task_dict_2)
    results.update(global_metrics)

    # Compute layerwise stacked SVD metrics (1 SVD per layer)
    layerwise_metrics = compute_layerwise_stacked_svd_metrics(task_dict_1, task_dict_2)
    results.update(layerwise_metrics)

    # Compute per-layer tensor SVD metrics (2 SVDs per 2D layer, shared across metrics)
    perlayer_metrics = compute_perlayer_tensor_svd_metrics(
        task_dict_1, task_dict_2, sv_overlap_top_k, subspace_top_k
    )
    results.update(perlayer_metrics)

    return results


# =============================================================================
# Metric Registry
# =============================================================================


# Wrapper functions for metrics that return tuples
def right_subspace_overlap_top_k(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> float:
    """Wrapper for right_subspace_overlap returning only top-k overlap."""
    top_overlap, _ = right_subspace_overlap(task_dict_1, task_dict_2, top_k)
    return top_overlap


def right_subspace_overlap_bottom_k(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> float:
    """Wrapper for right_subspace_overlap returning only bottom-k overlap."""
    _, bottom_overlap = right_subspace_overlap(task_dict_1, task_dict_2, top_k)
    return bottom_overlap


def interaction_matrix_overlap_top_k(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> float:
    """Wrapper for interaction_matrix_overlap returning only top-k overlap."""
    top_overlap, _ = interaction_matrix_overlap(task_dict_1, task_dict_2, top_k)
    return top_overlap


def interaction_matrix_overlap_bottom_k(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> float:
    """Wrapper for interaction_matrix_overlap returning only bottom-k overlap."""
    _, bottom_overlap = interaction_matrix_overlap(task_dict_1, task_dict_2, top_k)
    return bottom_overlap


# Update this registry when you add new metrics!
METRIC_REGISTRY: Dict[str, Callable] = {
    # Weight-based metrics
    "task_vector_cosine_similarity": task_vector_cosine_similarity,
    "task_vector_l2_distance": task_vector_l2_distance,
    "task_vector_dot_product": task_vector_dot_product,
    "weight_space_angle": weight_space_angle,
    "task_vector_magnitude_ratio": task_vector_magnitude_ratio,
    # Effective rank metrics (tangent space alignment)
    "effective_rank": effective_rank,
    "effective_rank_mergeability_score": effective_rank_mergeability_score,
    "stable_rank": stable_rank,
    "spectral_gap": spectral_gap,
    "singular_value_ratio": singular_value_ratio,
    "layerwise_effective_rank": layerwise_effective_rank,
    "layerwise_effective_rank_mergeability_score": layerwise_effective_rank_mergeability_score,
    # SVD-based subspace metrics
    "singular_value_overlap": singular_value_overlap,
    "subspace_overlap": subspace_overlap,
    "right_subspace_overlap": right_subspace_overlap,
    "interaction_matrix_overlap": interaction_matrix_overlap,
    "right_subspace_overlap_top_k": right_subspace_overlap_top_k,
    "right_subspace_overlap_bottom_k": right_subspace_overlap_bottom_k,
    "interaction_matrix_overlap_top_k": interaction_matrix_overlap_top_k,
    "interaction_matrix_overlap_bottom_k": interaction_matrix_overlap_bottom_k,
    # Activation-based metrics
    "activation_l2_distance": activation_l2_distance,
    "activation_cosine_similarity": activation_cosine_similarity,
    "activation_magnitude_ratio": activation_magnitude_ratio,
    "activation_dot_product": activation_dot_product,
    # Gradient-based metrics (encoder gradients)
    "encoder_gradient_cosine_similarity": encoder_gradient_cosine_similarity,
    "encoder_gradient_l2_distance": encoder_gradient_l2_distance,
    "encoder_gradient_dot_product": encoder_gradient_dot_product,
    # Gradient-based metrics (input gradients)
    "input_gradient_cosine_similarity": input_gradient_cosine_similarity,
    "input_gradient_l2_distance": input_gradient_l2_distance,
    "input_gradient_dot_product": input_gradient_dot_product,
}

# Registry for metrics that return tuples (metric_name -> list of output names)
TUPLE_METRICS: Dict[str, List[str]] = {
    "right_subspace_overlap": ["right_subspace_overlap_top_k", "right_subspace_overlap_bottom_k"],
    "interaction_matrix_overlap": ["interaction_matrix_overlap_top_k", "interaction_matrix_overlap_bottom_k"],
}


def compute_metric(
    metric_name: str,
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    **kwargs,
) -> Union[float, Dict[str, float]]:
    """Compute a specific metric by name.

    Args:
        metric_name: Name of the metric (must be in METRIC_REGISTRY).
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        **kwargs: Additional arguments passed to the metric function.

    Returns:
        Metric value (float or dict for per-layer metrics).

    Raises:
        ValueError: If metric_name is not in the registry.
    """
    if metric_name not in METRIC_REGISTRY:
        available = ", ".join(METRIC_REGISTRY.keys())
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {available}")

    return METRIC_REGISTRY[metric_name](task_dict_1, task_dict_2, **kwargs)


def compute_all_metrics(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    layer_wise: bool = False,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Compute all registered metrics.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        layer_wise: If True, compute all metrics per-layer and return both
                   per-layer breakdown and average.

    Returns:
        Dictionary mapping metric names to their values.
        If layer_wise=True, each metric has {"per_layer": {...}, "avg": float}
    """
    results = {}

    for name, func in METRIC_REGISTRY.items():
        try:
            if layer_wise:
                per_layer = compute_metric_per_layer(func, task_dict_1, task_dict_2)
                valid_values = [v for v in per_layer.values() if not math.isnan(v)]
                avg = sum(valid_values) / len(valid_values) if valid_values else 0.0
                results[name] = {"per_layer": per_layer, "avg": avg}
            else:
                results[name] = func(task_dict_1, task_dict_2)
        except Exception as e:
            print(f"Warning: Failed to compute {name}: {e}")
            results[name] = None

    return results
