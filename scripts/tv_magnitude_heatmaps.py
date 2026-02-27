"""
Task Vector Magnitude Heatmap Visualization

This script computes task vectors (weight space differences between finetuned and pretrained models)
and visualizes the layer-wise magnitudes as heatmaps.

For each weight matrix, we compute the Frobenius norm of the task vector and plot:
- Y-axis: Tasks (e.g., MNIST, CIFAR10, etc.)
- X-axis: Weight matrix names
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

pylogger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Dataset lists for different benchmarks
BENCHMARKS = {
    "N8": ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"],
    "N14": ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD",
            "Flowers102", "PCAM", "FER2013", "OxfordIIITPet", "STL10", "CIFAR100"],
    "N20": ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD",
            "Flowers102", "PCAM", "FER2013", "OxfordIIITPet", "STL10", "CIFAR100",
            "CIFAR10", "Food101", "FashionMNIST", "EMNIST", "KMNIST", "RenderedSST2"],
}


class ImageEncoder(torch.nn.Module):
    """Lightweight ImageEncoder that wraps an OpenCLIP model."""

    def __init__(self, model_name: str, keep_lang: bool = False):
        super().__init__()

        pylogger.info(f"Loading {model_name} pre-trained weights.")
        if "__pretrained__" in model_name:
            name, pretrained = model_name.split("__pretrained__")
        else:
            name = model_name
            pretrained = "openai"

        self.model, self.train_preprocess, self.val_preprocess = (
            open_clip.create_model_and_transforms(name, pretrained=pretrained)
        )

        if not keep_lang and hasattr(self.model, "transformer"):
            pylogger.info("Removing text transformer from the model.")
            delattr(self.model, "transformer")

    def forward(self, images):
        return self.model.encode_image(images)


def load_model_from_hf(model_name: str, dataset_name: str = "base") -> ImageEncoder:
    """
    Load a model from HuggingFace Hub.

    Args:
        model_name: Name of the model architecture (e.g., "ViT-B-16")
        dataset_name: Name of the dataset the model was finetuned on, or "base" for pretrained

    Returns:
        ImageEncoder with loaded weights
    """
    repo_id = f"crisostomi/{model_name}-{dataset_name}"

    ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    model = ImageEncoder(model_name)
    model.load_state_dict(state_dict)
    return model


def compute_task_vector(
    pretrained_state_dict: Dict[str, torch.Tensor],
    finetuned_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute the task vector (weight space difference) between finetuned and pretrained models.

    Args:
        pretrained_state_dict: State dict of the pretrained model
        finetuned_state_dict: State dict of the finetuned model

    Returns:
        Dictionary mapping weight names to task vectors (finetuned - pretrained)
    """
    task_vector = {}
    for key in pretrained_state_dict.keys():
        if key in finetuned_state_dict:
            task_vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    return task_vector


def compute_weight_magnitudes(
    task_vector: Dict[str, torch.Tensor],
    norm_type: str = "fro",
    row_wise: bool = False,
) -> Dict[str, float]:
    """
    Compute the magnitude (norm) of each weight matrix in the task vector.

    Args:
        task_vector: Dictionary mapping weight names to task vectors
        norm_type: Type of norm to use ("fro" for Frobenius, "l2" for L2, "l1" for L1)
        row_wise: If True, compute average of row-wise norms instead of matrix norm

    Returns:
        Dictionary mapping weight names to their magnitudes
    """
    magnitudes = {}
    for key, tensor in task_vector.items():
        tensor = tensor.float()

        if row_wise and tensor.dim() >= 2:
            # Compute row-wise norms and average them
            # Flatten all dimensions except the first (row) dimension
            rows = tensor.view(tensor.shape[0], -1)
            if norm_type == "fro" or norm_type == "l2":
                row_norms = torch.norm(rows, p=2, dim=1)
            elif norm_type == "l1":
                row_norms = torch.norm(rows, p=1, dim=1)
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")
            mag = row_norms.mean().item()
        else:
            # Original behavior: compute norm of entire tensor
            if norm_type == "fro":
                mag = torch.norm(tensor, p="fro").item()
            elif norm_type == "l2":
                mag = torch.norm(tensor, p=2).item()
            elif norm_type == "l1":
                mag = torch.norm(tensor, p=1).item()
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")
        magnitudes[key] = mag
    return magnitudes


def shorten_weight_name(name: str) -> str:
    """Shorten weight matrix names for better visualization."""
    # Common replacements
    name = name.replace("model.visual.", "")
    name = name.replace("transformer.resblocks.", "blk")
    name = name.replace(".weight", ".w")
    name = name.replace(".bias", ".b")
    name = name.replace("attn.", "a.")
    name = name.replace("mlp.", "m.")
    name = name.replace("ln_", "ln")
    name = name.replace("in_proj", "in")
    name = name.replace("out_proj", "out")
    name = name.replace("c_fc", "fc1")
    name = name.replace("c_proj", "fc2")
    return name


def filter_weight_keys(
    keys: List[str],
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    min_params: int = 0,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> List[str]:
    """
    Filter weight keys based on patterns and parameter count.

    Args:
        keys: List of weight names to filter
        include_patterns: Only include keys containing any of these patterns
        exclude_patterns: Exclude keys containing any of these patterns
        min_params: Minimum number of parameters for a weight to be included
        state_dict: State dict to check parameter counts (required if min_params > 0)

    Returns:
        Filtered list of weight names
    """
    filtered = keys

    if include_patterns:
        filtered = [k for k in filtered if any(p in k for p in include_patterns)]

    if exclude_patterns:
        filtered = [k for k in filtered if not any(p in k for p in exclude_patterns)]

    if min_params > 0 and state_dict is not None:
        filtered = [k for k in filtered if state_dict[k].numel() >= min_params]

    return filtered


def plot_task_vector_heatmap(
    magnitudes_per_task: Dict[str, Dict[str, float]],
    weight_keys: List[str],
    output_path: Path,
    title: str = "Task Vector Magnitudes",
    figsize: tuple = (20, 10),
    normalize: str = "none",
    cmap: str = "viridis",
):
    """
    Plot a heatmap of task vector magnitudes.

    Args:
        magnitudes_per_task: Dictionary mapping task names to their weight magnitudes
        weight_keys: List of weight names to include (determines x-axis order)
        output_path: Path to save the figure
        title: Title for the heatmap
        figsize: Figure size
        normalize: Normalization method ("none", "per_task", "per_weight", "global")
        cmap: Colormap to use
    """
    tasks = list(magnitudes_per_task.keys())

    # Build the data matrix
    data = np.zeros((len(tasks), len(weight_keys)))
    for i, task in enumerate(tasks):
        for j, key in enumerate(weight_keys):
            data[i, j] = magnitudes_per_task[task].get(key, 0.0)

    # Apply normalization
    if normalize == "per_task":
        # Normalize each task (row) to have max 1
        row_max = data.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1  # Avoid division by zero
        data = data / row_max
    elif normalize == "per_weight":
        # Normalize each weight (column) to have max 1
        col_max = data.max(axis=0, keepdims=True)
        col_max[col_max == 0] = 1
        data = data / col_max
    elif normalize == "global":
        # Normalize globally
        if data.max() > 0:
            data = data / data.max()

    # Shorten weight names for display
    short_names = [shorten_weight_name(k) for k in weight_keys]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(data, aspect='auto', cmap=cmap)

    # Set ticks and labels
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xticks(np.arange(len(short_names)))
    ax.set_xticklabels(short_names, rotation=90, ha='center', fontsize=6)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude' + (' (normalized)' if normalize != "none" else ''))

    ax.set_title(title)
    ax.set_xlabel('Weight Matrix')
    ax.set_ylabel('Task')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    pylogger.info(f"Saved heatmap to {output_path}")


def plot_layer_aggregated_heatmap(
    magnitudes_per_task: Dict[str, Dict[str, float]],
    weight_keys: List[str],
    output_path: Path,
    title: str = "Task Vector Magnitudes (Aggregated per Layer)",
    figsize: tuple = (12, 8),
    normalize: str = "none",
    cmap: str = "viridis",
):
    """
    Plot a heatmap with magnitudes aggregated per transformer block.
    """
    import re

    tasks = list(magnitudes_per_task.keys())

    # Aggregate magnitudes per block
    block_magnitudes = {}  # task -> block -> sum of magnitudes
    other_magnitudes = {}  # task -> sum of non-block magnitudes

    for task in tasks:
        block_magnitudes[task] = {}
        other_magnitudes[task] = 0.0

        for key in weight_keys:
            mag = magnitudes_per_task[task].get(key, 0.0)
            match = re.search(r'resblocks\.(\d+)', key)
            if match:
                block_num = int(match.group(1))
                block_magnitudes[task][block_num] = block_magnitudes[task].get(block_num, 0.0) + mag
            else:
                other_magnitudes[task] += mag

    # Get all block numbers
    all_blocks = set()
    for task in tasks:
        all_blocks.update(block_magnitudes[task].keys())
    sorted_blocks = sorted(all_blocks)

    # Column labels
    col_labels = ["Other"] + [f"Block {b}" for b in sorted_blocks]

    # Build data matrix
    data = np.zeros((len(tasks), len(col_labels)))
    for i, task in enumerate(tasks):
        data[i, 0] = other_magnitudes[task]
        for j, block in enumerate(sorted_blocks):
            data[i, j + 1] = block_magnitudes[task].get(block, 0.0)

    # Apply normalization
    if normalize == "per_task":
        row_max = data.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        data = data / row_max
    elif normalize == "per_weight":
        col_max = data.max(axis=0, keepdims=True)
        col_max[col_max == 0] = 1
        data = data / col_max
    elif normalize == "global":
        if data.max() > 0:
            data = data / data.max()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect='auto', cmap=cmap)

    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Summed Magnitude' + (' (normalized)' if normalize != "none" else ''))

    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Task')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    pylogger.info(f"Saved layer-aggregated heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute and visualize task vector magnitudes as heatmaps"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-16",
        help="Model name (default: ViT-B-16)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="N8",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to use (default: N8)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="List of datasets to use (overrides --benchmark)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figs",
        help="Output directory for heatmaps",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="fro",
        choices=["fro", "l2", "l1"],
        help="Norm type for computing magnitudes (default: fro)",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "per_task", "per_weight", "global"],
        help="Normalization method for heatmap (default: none)",
    )
    parser.add_argument(
        "--min_params",
        type=int,
        default=1000,
        help="Minimum parameters for a weight to be included (default: 1000)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap to use (default: viridis)",
    )
    parser.add_argument(
        "--row_wise",
        action="store_true",
        help="If set, compute average of row-wise norms instead of matrix norm",
    )

    args = parser.parse_args()

    # Determine datasets to use
    datasets = args.datasets if args.datasets else BENCHMARKS[args.benchmark]
    pylogger.info(f"Using datasets: {datasets}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    pylogger.info(f"Loading pretrained model: {args.model_name}")
    pretrained_model = load_model_from_hf(model_name=args.model_name, dataset_name="base")
    pretrained_state_dict = pretrained_model.state_dict()

    # Get weight keys (filtering by min_params)
    all_keys = list(pretrained_state_dict.keys())
    weight_keys = filter_weight_keys(
        all_keys,
        min_params=args.min_params,
        state_dict=pretrained_state_dict,
    )
    pylogger.info(f"Using {len(weight_keys)} weight matrices (out of {len(all_keys)} total)")
    if args.row_wise:
        pylogger.info("Using row-wise averaging for magnitude computation")

    # Compute task vectors for each dataset
    magnitudes_per_task = {}

    for dataset_name in tqdm(datasets, desc="Loading finetuned models"):
        pylogger.info(f"Loading finetuned model for {dataset_name}")
        try:
            finetuned_model = load_model_from_hf(
                model_name=args.model_name, dataset_name=dataset_name
            )
            finetuned_state_dict = finetuned_model.state_dict()

            # Compute task vector
            task_vector = compute_task_vector(pretrained_state_dict, finetuned_state_dict)

            # Compute magnitudes
            magnitudes = compute_weight_magnitudes(task_vector, norm_type=args.norm_type, row_wise=args.row_wise)
            magnitudes_per_task[dataset_name] = magnitudes

        except Exception as e:
            pylogger.error(f"Failed to load {dataset_name}: {e}")
            continue

    if not magnitudes_per_task:
        pylogger.error("No models loaded successfully!")
        return

    # Suffix for filenames
    suffix = "_row_wise" if args.row_wise else ""

    # Plot the main layer-wise magnitude heatmap
    main_heatmap_path = output_dir / f"tv_layerwise_magnitude_heatmap{suffix}.png"
    plot_task_vector_heatmap(
        magnitudes_per_task,
        weight_keys,
        main_heatmap_path,
        title=f"Task Vector Layer-wise Magnitudes ({args.model_name}, {args.benchmark})",
        normalize=args.normalize,
        cmap=args.cmap,
    )

    # Plot weights-only heatmap (excluding biases)
    weights_only_keys = [k for k in weight_keys if "bias" not in k.lower()]
    pylogger.info(f"Plotting weights-only heatmap with {len(weights_only_keys)} weight matrices (excluded {len(weight_keys) - len(weights_only_keys)} biases)")
    plot_task_vector_heatmap(
        magnitudes_per_task,
        weights_only_keys,
        output_dir / f"tv_layerwise_magnitude_heatmap_weights_only{suffix}.png",
        title=f"Task Vector Layer-wise Magnitudes - Weights Only ({args.model_name}, {args.benchmark})",
        normalize=args.normalize,
        cmap=args.cmap,
    )

    # Plot layer-aggregated heatmap
    plot_layer_aggregated_heatmap(
        magnitudes_per_task,
        weight_keys,
        output_dir / f"tv_layeragg_magnitude_heatmap{suffix}.png",
        title=f"Task Vector Magnitudes - Layer Aggregated ({args.model_name}, {args.benchmark})",
        normalize=args.normalize,
        cmap=args.cmap,
    )

    pylogger.info(f"Heatmaps saved to {output_dir}")


if __name__ == "__main__":
    main()
