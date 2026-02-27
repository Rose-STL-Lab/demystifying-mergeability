import copy
import logging
import random
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import torch
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
)

pylogger = logging.getLogger(__name__)


def _set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dare_mask(tensor: torch.Tensor, drop_rate: float, use_rescale: bool, mask_strategy: str) -> torch.Tensor:
    """
    Apply DARE (Drop And REscale) masking to a tensor.

    Args:
        tensor: Input tensor (task vector parameters)
        drop_rate: Fraction of parameters to drop (0 to 1), e.g., 0.9 drops 90%
        use_rescale: Whether to rescale remaining parameters by 1/(1-drop_rate)
        mask_strategy: Masking strategy - "random" or "magnitude"

    Returns:
        Masked (and optionally rescaled) tensor
    """
    if drop_rate <= 0.0:
        return tensor
    if drop_rate >= 1.0:
        return torch.zeros_like(tensor)

    original_shape = tensor.shape
    flat_tensor = tensor.flatten()

    if mask_strategy == "random":
        # Random masking: each parameter dropped with probability drop_rate
        mask = torch.bernoulli(torch.full_like(flat_tensor, fill_value=drop_rate))
        masked_tensor = flat_tensor * (1 - mask)
    elif mask_strategy == "magnitude":
        # Magnitude-based masking: drop the smallest magnitude parameters
        num_to_drop = int(len(flat_tensor) * drop_rate)
        if num_to_drop == 0:
            masked_tensor = flat_tensor
        elif num_to_drop >= len(flat_tensor):
            masked_tensor = torch.zeros_like(flat_tensor)
        else:
            # Find the threshold: the (num_to_drop)-th smallest magnitude value
            kth_value, _ = flat_tensor.abs().kthvalue(k=num_to_drop)
            # Mask parameters with magnitude <= threshold
            mask = flat_tensor.abs() <= kth_value
            masked_tensor = flat_tensor * (~mask)
    else:
        raise ValueError(f"Unknown mask_strategy: {mask_strategy}. Use 'random' or 'magnitude'.")

    # Rescale to maintain expected magnitude
    if use_rescale and drop_rate < 1.0:
        masked_tensor = masked_tensor / (1 - drop_rate)

    return masked_tensor.view(original_shape)


class DAReMerger(TaskVectorBasedMerger):
    """
    DARE Merging: Drop And REscale for Model Merging

    From the paper "Language Models are Super Mario: Absorbing Abilities from
    Homologous Models as a Free Lunch" (ICML 2024).

    DARE works by:
    1. Computing task vectors (finetuned - pretrained) for each model
    2. Randomly dropping a large fraction (e.g., 90%) of delta parameters
    3. Rescaling remaining parameters by 1/(1-drop_rate) to maintain magnitude
    4. Aggregating the masked task vectors via mean or sum
    5. Applying the merged task vector to the pretrained model
    """

    def __init__(
        self,
        scaling_coefficient: float = 1.0,
        drop_rate: float = 0.9,
        use_rescale: bool = True,
        mask_strategy: str = "random",
        merge_func: str = "mean",
        seed: Optional[int] = 0,
        device: str = "cuda",
    ):
        """
        Args:
            scaling_coefficient: Scaling factor for the merged task vector
            drop_rate: Fraction of parameters to drop (0 to 1), default 0.9 (drop 90%)
            use_rescale: Whether to rescale by 1/(1-drop_rate), default True
            mask_strategy: "random" (Bernoulli) or "magnitude" (drop smallest)
            merge_func: Aggregation method - "mean" or "sum"
            seed: Random seed for reproducibility (default 0, set None to skip seeding)
            device: Device for computation
        """
        super().__init__()

        self.scaling_coefficient = scaling_coefficient
        self.drop_rate = drop_rate
        self.use_rescale = use_rescale
        self.mask_strategy = mask_strategy
        self.merge_func = merge_func
        self.seed = seed
        self.device = device

    def merge(
        self, base_model: ImageEncoder, finetuned_models: Dict[str, Dict]
    ) -> ImageEncoder:

        # Set random seed for reproducibility
        if self.seed is not None:
            _set_random_seed(self.seed)

        datasets = list(finetuned_models.keys())
        pretrained_model = copy.deepcopy(base_model)
        base_state_dict = base_model.state_dict()

        # Compute and DARE-mask task vectors for each finetuned model
        masked_task_vectors = []
        for dataset in datasets:
            # Compute task vector: tau = theta_ft - theta_0
            task_dict = compute_task_dict(base_state_dict, finetuned_models[dataset])

            # Apply DARE masking to each parameter
            masked_task_dict = OrderedDict()
            for key, value in task_dict.items():
                masked_task_dict[key] = _dare_mask(
                    tensor=value.float(),
                    drop_rate=self.drop_rate,
                    use_rescale=self.use_rescale,
                    mask_strategy=self.mask_strategy,
                ).to(value.dtype)

            masked_task_vectors.append(masked_task_dict)
            del finetuned_models[dataset]
            torch.cuda.empty_cache()

        # Get all parameter keys
        param_keys = list(masked_task_vectors[0].keys())

        # Aggregate masked task vectors
        merged_task_vector = OrderedDict()
        for key in param_keys:
            stacked = torch.stack([tv[key] for tv in masked_task_vectors], dim=0)

            if self.merge_func == "mean":
                merged_task_vector[key] = stacked.mean(dim=0)
            elif self.merge_func == "sum":
                merged_task_vector[key] = stacked.sum(dim=0)
            else:
                raise ValueError(f"Unknown merge_func: {self.merge_func}. Use 'mean' or 'sum'.")

        # Apply merged task vector to pretrained model
        merged_encoder = apply_dict_to_model(
            merged_task_vector, pretrained_model, coefficient=self.scaling_coefficient
        )

        return merged_encoder