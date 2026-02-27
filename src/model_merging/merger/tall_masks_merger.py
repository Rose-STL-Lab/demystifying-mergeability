import copy
import logging
from collections import OrderedDict
from typing import Dict

import torch
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
)

pylogger = logging.getLogger(__name__)


def _generate_tall_mask(
    pretrained_flat: torch.Tensor,
    finetuned_flat: torch.Tensor,
    merged_flat: torch.Tensor,
    tall_mask_lambda: float = 0.4,
) -> torch.Tensor:
    """
    Generate a TALL mask for a single task.

    TALL masks are generated as: mask_t = |θ_pretrained - θ_finetuned| > |θ_merged - θ_finetuned| * λ

    A parameter is marked as task-specific (True) if the distance from pretrained to
    finetuned is greater than the distance from merged to finetuned (scaled by λ).

    Args:
        pretrained_flat: Flattened pretrained model parameters
        finetuned_flat: Flattened finetuned model parameters for this task
        merged_flat: Flattened merged model parameters
        tall_mask_lambda: Hyperparameter controlling mask sensitivity (default 0.4)

    Returns:
        Boolean mask tensor indicating task-specific parameters
    """
    # L1 distance from pretrained to finetuned
    diff_pretrained_finetuned = (pretrained_flat - finetuned_flat).abs()

    # L1 distance from merged to finetuned
    diff_merged_finetuned = (merged_flat - finetuned_flat).abs()

    # Generate mask: parameter is task-specific if pretrained->finetuned distance
    # is greater than merged->finetuned distance (scaled by lambda)
    mask = diff_pretrained_finetuned > (diff_merged_finetuned * tall_mask_lambda)

    return mask


def _compute_consensus_mask(
    task_masks: torch.Tensor,
    min_task_count: int = 2,
) -> torch.Tensor:
    """
    Compute a consensus mask from multiple task-specific masks.

    This filters out:
    - "Catastrophic weights": parameters not important to any task (all masks = False)
    - "Selfish weights": parameters important to only one task (if min_task_count > 1)

    Args:
        task_masks: Tensor of shape (num_tasks, num_params) with boolean masks
        min_task_count: Minimum number of tasks that must mark a parameter as important

    Returns:
        Boolean consensus mask
    """
    # Count how many tasks mark each parameter as important
    importance_count = task_masks.sum(dim=0)

    # Keep parameters important to at least min_task_count tasks
    consensus_mask = importance_count >= min_task_count

    return consensus_mask


class TALLMasksMerger(TaskVectorBasedMerger):
    """
    TALL-Masks: Task-specific Activation Localization via Language Learning Masks

    From the paper "Localizing Task Information for Improved Model Merging and Compression"
    (Wang et al., 2024)

    This merger:
    1. Computes task vectors for each finetuned model
    2. Sums task vectors to create a merged task vector
    3. Generates TALL masks to identify task-specific parameters
    4. Uses consensus filtering to remove catastrophic and selfish weights
    5. Applies the filtered merged task vector to the pretrained model
    """

    def __init__(
        self,
        scaling_coefficient: float = 1.0,
        tall_mask_lambda: float = 0.4,
        min_task_count: int = None,
        device: str = "cuda",
    ):
        """
        Args:
            scaling_coefficient: Scaling factor for the merged task vector
            tall_mask_lambda: Lambda hyperparameter for TALL mask generation (0.2-0.6 typical)
            min_task_count: Minimum tasks for consensus (None = auto: 2 for >2 tasks, 1 for 2 tasks)
            device: Device to use for computation
        """
        super().__init__()

        self.scaling_coefficient = scaling_coefficient
        self.tall_mask_lambda = tall_mask_lambda
        self.min_task_count = min_task_count
        self.device = device

    def merge(
        self, base_model: ImageEncoder, finetuned_models: Dict[str, Dict]
    ) -> ImageEncoder:

        datasets = list(finetuned_models.keys())
        num_tasks = len(datasets)
        pretrained_model = copy.deepcopy(base_model)
        base_state_dict = base_model.state_dict()

        # Compute task vectors for each finetuned model
        task_vectors = []
        finetuned_state_dicts = []
        for dataset in datasets:
            task_dict = compute_task_dict(base_state_dict, finetuned_models[dataset])
            task_vectors.append(task_dict)
            finetuned_state_dicts.append(finetuned_models[dataset])

        # Free memory from finetuned_models dict
        for dataset in datasets:
            del finetuned_models[dataset]
        torch.cuda.empty_cache()

        # Get all parameter keys
        param_keys = list(task_vectors[0].keys())

        # Sum task vectors to create merged task vector
        merged_task_vector = OrderedDict()
        for key in param_keys:
            merged_task_vector[key] = sum(tv[key] for tv in task_vectors)

        # Compute merged model state dict (for mask generation)
        merged_state_dict = OrderedDict()
        for key in param_keys:
            merged_state_dict[key] = base_state_dict[key] + merged_task_vector[key]

        # Generate TALL masks for each task
        # Process parameter-by-parameter to build task masks
        filtered_task_vector = OrderedDict()

        # For logging: accumulate mask statistics across all parameters
        all_task_masks = []
        all_consensus_masks = []

        # Determine min_count once
        min_count = self.min_task_count
        if min_count is None:
            # Auto: for 2 tasks use 1 (keep if important to at least 1 task)
            # for >2 tasks use 2 (filter out selfish weights)
            min_count = 1 if num_tasks <= 2 else 2

        for key in param_keys:
            pretrained_param = base_state_dict[key].flatten().float()
            merged_param = merged_state_dict[key].flatten().float()

            # Generate mask for each task
            task_masks = []
            for i in range(num_tasks):
                finetuned_param = finetuned_state_dicts[i][key].flatten().float()
                mask = _generate_tall_mask(
                    pretrained_param,
                    finetuned_param,
                    merged_param,
                    self.tall_mask_lambda,
                )
                task_masks.append(mask)

            # Stack masks: shape (num_tasks, num_params)
            task_masks = torch.stack(task_masks, dim=0)
            all_task_masks.append(task_masks)

            consensus_mask = _compute_consensus_mask(task_masks, min_count)
            all_consensus_masks.append(consensus_mask)

            # Apply consensus mask to merged task vector
            original_shape = merged_task_vector[key].shape
            filtered_tv = merged_task_vector[key].flatten().float() * consensus_mask.float()
            filtered_task_vector[key] = filtered_tv.view(original_shape).to(merged_task_vector[key].dtype)

        # Log mask statistics (matching original paper's terminology)
        all_task_masks = torch.cat([m.flatten(1) for m in all_task_masks], dim=1)  # (num_tasks, total_params)
        all_consensus_masks = torch.cat(all_consensus_masks)  # (total_params,)

        # Per-task mask density (what original calls "sparsity")
        task_mask_densities = all_task_masks.float().mean(dim=1)
        for i, dataset in enumerate(datasets):
            pylogger.info(f"TALL mask density for {dataset}: {task_mask_densities[i]:.4f}")

        # Consensus mask density and final filtering stats
        consensus_density = all_consensus_masks.float().mean().item()
        pylogger.info(f"Consensus mask density: {consensus_density:.4f} (min_task_count={min_count}, λ={self.tall_mask_lambda})")

        # Apply filtered task vector to pretrained model
        merged_encoder = apply_dict_to_model(
            filtered_task_vector, pretrained_model, coefficient=self.scaling_coefficient
        )

        return merged_encoder