import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_merging.data.dataset import HFImageClassification
from model_merging.model.image_classifier import ImageClassifier
import open_clip
import wandb

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import model_merging  # noqa
from model_merging.model.encoder import ClassificationHead, ImageEncoder
from model_merging.model.heads import (
    get_classification_head,
)
from model_merging.utils.io_utils import (
    boilerplate,
    load_model_from_hf,
)
from model_merging.alignment.rotation_alignment import apply_rotation_alignment
from model_merging.utils.plots import plot_interactive_radar_chart
from model_merging.utils.utils import (
    build_callbacks,
    get_finetuning_accuracies,
    compute_avg_accuracy,
    print_memory,
)
import json
import os

pylogger = logging.getLogger(__name__)


def generate_reg_suffix(cfg: DictConfig) -> str:
    """Generate regularization suffix based on enabled regularizations in config.

    This mirrors the logic in finetune.py to ensure consistent naming.
    """
    if not hasattr(cfg, 'train') or not hasattr(cfg.train, 'regularization'):
        return ""

    reg_parts = []
    reg_cfg = cfg.train.regularization

    moderate_update_enabled = getattr(reg_cfg, 'enable_moderate_update', False)
    grad_magnitude_enabled = getattr(reg_cfg, 'enable_grad_magnitude', False)
    tv_subspace_enabled = getattr(reg_cfg, 'enable_tv_subspace_penalty', False)
    gargiulo_enabled = getattr(reg_cfg, 'enable_gargiulo_penalty', False)

    if moderate_update_enabled:
        mu_lambda = getattr(reg_cfg, 'lambda_moderate_update', 0.01)
        reg_parts.append(f"moderate_update_{mu_lambda}")
    if grad_magnitude_enabled:
        gm_lambda = getattr(reg_cfg, 'lambda_grad_magnitude', 1)
        reg_parts.append(f"grad_magnitude_{gm_lambda}")
    if tv_subspace_enabled:
        tv_vectors = getattr(reg_cfg, 'tv_penalty_singular_vectors', 'V').lower()
        tv_lambda = getattr(reg_cfg, 'lambda_tv_subspace', 0.001)
        reg_parts.append(f"tv_subspace_{tv_vectors}_{tv_lambda}")
    if gargiulo_enabled:
        g_vectors = getattr(reg_cfg, 'gargiulo_singular_vectors', 'U').lower()
        g_lambda = getattr(reg_cfg, 'lambda_gargiulo', 0.0001)
        reg_parts.append(f"gargiulo_{g_vectors}_{g_lambda}")

    if reg_parts:
        return "_" + "_".join(reg_parts)
    return ""

torch.set_float32_matmul_precision("high")


def load_config(
    config_path: str,
    config_name: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Load a Hydra config without launching a full Hydra app.

    Args:
        config_path (str): Path to the folder containing your configs (relative to project root).
        config_name (str): Name of the YAML config file (without `.yaml`).
        overrides (list[str], optional): List of override strings, e.g. ["trainer.max_epochs=20"].

    Returns:
        DictConfig: The loaded configuration.
    """
    overrides = overrides or []
    abs_config_path = str(Path(config_path).absolute())

    with hydra.initialize(config_path=abs_config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

    return cfg


def run_single(cfg: DictConfig, datasets_to_use: Optional[List] = None, pair_name: Optional[str] = None, file_prefix: str = "pair") -> Dict:
    """Run merging evaluation for a single set of datasets.

    Args:
        cfg: run configuration, defined by Hydra in /conf
        datasets_to_use: Optional list of dataset configs to use (for pairwise mode)
        pair_name: Optional name for the pair/subset (for logging/saving)
        file_prefix: Prefix for the result filename (e.g., "pair", "subset_5")

    Returns:
        Dictionary containing evaluation results
    """
    seed_index_everything(cfg)

    logger, template_core = boilerplate(cfg)

    # Use provided datasets or fall back to config
    datasets = datasets_to_use if datasets_to_use is not None else list(cfg.benchmark.datasets)
    num_tasks = len(datasets)

    # Temporarily disable struct mode to allow dynamic update
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = num_tasks  # Now we can safely update it
    omegaconf.OmegaConf.set_struct(cfg, True)  # Re-enable struct mode

    # upperbound accuracies, used for logging the normalized accuracy
    finetuned_accuracies: Dict[str, float] = get_finetuning_accuracies(
        cfg.misc.finetuned_accuracy_path
    )[cfg.nn.encoder.model_name]

    # only has vision encoder, no text transformer
    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )

    # Load finetuned models - either from local checkpoints or HuggingFace
    # Use explicit reg_suffix if set, otherwise generate from regularization config
    reg_suffix = getattr(cfg.misc, 'reg_suffix', '')
    if not reg_suffix:
        reg_suffix = generate_reg_suffix(cfg)
        if reg_suffix:
            pylogger.info(f"Auto-generated regularization suffix: {reg_suffix}")
    finetuned_models = {}

    for dataset in datasets:
        if reg_suffix:
            # Load from local checkpoints with regularization suffix
            dataset_name_with_suffix = f"{dataset.name}{reg_suffix}"
            local_checkpoint_path = os.path.join(cfg.misc.ckpt_path, dataset_name_with_suffix, "model.pt")

            if os.path.exists(local_checkpoint_path):
                pylogger.info(f"Loading {dataset.name} from local: {local_checkpoint_path}")
                from model_merging.utils.io_utils import load_model_from_disk
                model = load_model_from_disk(local_checkpoint_path, model_name=cfg.nn.encoder.model_name)
                finetuned_models[dataset] = model.state_dict()
            else:
                pylogger.warning(f"Local checkpoint not found: {local_checkpoint_path}, falling back to HuggingFace")
                finetuned_models[dataset] = load_model_from_hf(
                    model_name=cfg.nn.encoder.model_name, dataset_name=dataset_name_with_suffix
                ).state_dict()
        else:
            # Load from HuggingFace (original behavior)
            pylogger.info(f"Loading {dataset.name} from HuggingFace")
            finetuned_models[dataset] = load_model_from_hf(
                model_name=cfg.nn.encoder.model_name, dataset_name=dataset.name
            ).state_dict()

    # Apply rotation symmetry alignment if enabled
    if cfg.alignment:
        pylogger.info("Applying rotation symmetry alignment...")

        # Save original weights for comparison
        import torch
        original_weights = {}
        original_models_full = {}
        for dataset_key in finetuned_models.keys():
            # Sample a few weights to check if they change
            original_weights[dataset_key] = {
                'qkv_weight': finetuned_models[dataset_key]['model.visual.transformer.resblocks.0.attn.in_proj_weight'].clone(),
                'out_proj': finetuned_models[dataset_key]['model.visual.transformer.resblocks.0.attn.out_proj.weight'].clone()
            }
            # Save full state dict for later comparison
            original_models_full[dataset_key] = {k: v.clone() for k, v in finetuned_models[dataset_key].items()}

        finetuned_models = apply_rotation_alignment(
            finetuned_state_dicts=finetuned_models,
            model_name=cfg.nn.encoder.model_name,
            device=cfg.device,
            logger=pylogger
        )

        # Check if weights actually changed
        pylogger.info("\n" + "="*70)
        pylogger.info("WEIGHT CHANGE VERIFICATION")
        pylogger.info("="*70)
        for dataset_key in finetuned_models.keys():
            dataset_name = dataset_key.name if hasattr(dataset_key, 'name') else str(dataset_key)
            qkv_diff = torch.abs(finetuned_models[dataset_key]['model.visual.transformer.resblocks.0.attn.in_proj_weight'] -
                                 original_weights[dataset_key]['qkv_weight']).max().item()
            out_diff = torch.abs(finetuned_models[dataset_key]['model.visual.transformer.resblocks.0.attn.out_proj.weight'] -
                                 original_weights[dataset_key]['out_proj']).max().item()
            pylogger.info(f"{dataset_name}:")
            pylogger.info(f"  QKV weight max diff: {qkv_diff:.6e}")
            pylogger.info(f"  Out proj weight max diff: {out_diff:.6e}")
        pylogger.info("="*70 + "\n")

    # Debug: Log state dict info after alignment
    pylogger.info("\n" + "="*70)
    pylogger.info("STATE DICT DEBUG INFO (after alignment)")
    pylogger.info("="*70)
    for dataset_key, state_dict in finetuned_models.items():
        dataset_name = dataset_key.name if hasattr(dataset_key, 'name') else str(dataset_key)
        sample_keys = list(state_dict.keys())[:5]
        pylogger.info(f"{dataset_name}:")
        pylogger.info(f"  Total keys: {len(state_dict)}")
        pylogger.info(f"  Sample keys: {sample_keys}")
        pylogger.info(f"  Has 'model.visual' keys: {'model.visual.conv1.weight' in state_dict}")
        pylogger.info(f"  Has 'model.positional_embedding': {'model.positional_embedding' in state_dict}")
    pylogger.info("="*70 + "\n")

    if pair_name:
        pylogger.info(f"=== Evaluating pair: {pair_name} ===")
    pylogger.info(f"Number of tasks: {num_tasks}")
    pylogger.info(f"Finetuned models: {[d.name for d in datasets]}")
    pylogger.info(f"Using merger: {cfg.merger._target_}")

    merger = instantiate(cfg.merger)

    # Debug: Log a sample weight from finetuned_models before merging
    if cfg.alignment:
        pylogger.info("\n" + "="*70)
        pylogger.info("PRE-MERGE WEIGHT CHECK")
        pylogger.info("="*70)
        for dataset_key in finetuned_models.keys():
            dataset_name = dataset_key.name if hasattr(dataset_key, 'name') else str(dataset_key)
            sample_weight = finetuned_models[dataset_key]['model.visual.transformer.resblocks.0.attn.in_proj_weight'][0, :5]
            pylogger.info(f"{dataset_name} sample weight: {sample_weight}")
        pylogger.info("="*70 + "\n")

    merged_encoder = merger.merge(zeroshot_encoder, finetuned_models)

    # Debug: Log merged weight
    if cfg.alignment:
        pylogger.info("\n" + "="*70)
        pylogger.info("POST-MERGE WEIGHT CHECK")
        pylogger.info("="*70)
        merged_state = merged_encoder.state_dict()
        sample_merged = merged_state['model.visual.transformer.resblocks.0.attn.in_proj_weight'][0, :5]
        pylogger.info(f"Merged sample weight: {sample_merged}")
        pylogger.info("="*70 + "\n")

    logger.log_configuration(merged_encoder, cfg)

    results = {}
    print_memory("before eval")
    for dataset_cfg in datasets:

        dataset = instantiate(
            dataset_cfg, preprocess_fn=zeroshot_encoder.val_preprocess
        )

        classification_head = get_classification_head(
            cfg.nn.encoder.model_name,
            dataset_cfg.name,
            ckpt_path=cfg.misc.ckpt_path,
            openclip_cachedir=cfg.misc.openclip_cachedir,
            device=cfg.device,
        )

        model = ImageClassifier(
            encoder=merged_encoder,
            classifier=classification_head,
            x_key=cfg.conventions.x_key,
            y_key=cfg.conventions.y_key,
        )

        model.set_metrics(len(dataset.classnames))
        model.set_task(dataset_cfg.name)
        model.set_finetuning_accuracy(
            finetuned_accuracies[
                dataset_cfg.name + "Val" if cfg.eval_on_train else dataset_cfg.name
            ]
        )

        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        trainer = pl.Trainer(
            default_root_dir=cfg.core.storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=logger,
            callbacks=callbacks,
            limit_test_batches=(
                cfg.number_of_train_batches if cfg.eval_on_train else None
            ),
            **cfg.train.trainer,
        )

        if cfg.eval_on_train:
            pylogger.error("For now evaluation supported only on val-set")
            pylogger.info(f"Evaluating on {dataset_cfg.name} the training set")
            test_results = trainer.test(model=model, dataloaders=dataset.train_loader)

        else:
            pylogger.info(f"Evaluating on the {dataset_cfg.name} test set!")
            test_results = trainer.test(model=model, dataloaders=dataset.test_loader)

        results[dataset_cfg.name] = test_results

    avg = compute_avg_accuracy(results)
    results["avg"] = [
        avg
    ]  # as a list for consistency due to lightning logging stuff this way

    logger.experiment.log(avg)

    pylogger.info(results)

    # Extract merger name from target (e.g., "model_merging.merger.weight_avg_merger.WeightAvgMerger" -> "weight_avg")
    merger_name = cfg.merger._target_.split(".")[-2].replace("_merger", "")

    # Add regularization suffix to merger name if specified
    reg_suffix = getattr(cfg.misc, 'reg_suffix', '')
    if not reg_suffix:
        reg_suffix = generate_reg_suffix(cfg)
    merger_name_with_suffix = f"{merger_name}{reg_suffix}" if reg_suffix else merger_name

    # Create merger-specific folder with regularization suffix
    results_path = Path(cfg.misc.results_path) / merger_name_with_suffix
    results_path.mkdir(parents=True, exist_ok=True)

    # Use pair_name for filename if provided, otherwise use num_tasks
    # Add suffix if rotation alignment was used
    alignment_suffix = "_rot_aligned" if cfg.alignment else ""

    if pair_name:
        filename = f"{file_prefix}_{pair_name}{alignment_suffix}.json"
    else:
        filename = f"{num_tasks}{alignment_suffix}.json"

    with open(results_path / filename, "w+") as f:
        json.dump(results, f, indent=4)

    radarchart = plot_interactive_radar_chart(results, title="Radar Chart")
    logger.experiment.log({"radar": wandb.Plotly(radarchart)})

    pylogger.info(f"Results saved to {results_path / filename}")

    logger.experiment.log_artifact(
        wandb.Artifact(
            f"results_{cfg.nn.encoder.model_name}_{pair_name or num_tasks}",
            type="results",
            metadata={"results": str(results_path)},
        )
    )

    if logger is not None:
        logger.experiment.finish()

    return results


def run_random_subsets(cfg: DictConfig):
    """Run merging evaluation on random subsets of specified cardinality.

    Args:
        cfg: run configuration with subset_cardinality and num_random_subsets
    """
    import random
    from itertools import combinations

    datasets = list(cfg.benchmark.datasets)
    n_datasets = len(datasets)
    subset_size = cfg.get("subset_cardinality", 5)
    num_subsets = cfg.get("num_random_subsets", 200)
    random_seed = cfg.get("subset_random_seed", 42)

    # Validate
    if subset_size > n_datasets:
        raise ValueError(f"subset_cardinality ({subset_size}) > number of datasets ({n_datasets})")

    # Calculate total possible combinations
    from math import comb
    total_combinations = comb(n_datasets, subset_size)

    pylogger.info(f"Running RANDOM SUBSETS merging evaluation")
    pylogger.info(f"Benchmark has {n_datasets} datasets")
    pylogger.info(f"Subset cardinality: {subset_size}")
    pylogger.info(f"Total possible combinations: {total_combinations}")
    pylogger.info(f"Evaluating {num_subsets} random subsets (seed={random_seed})")

    # Generate random subsets
    random.seed(random_seed)
    if num_subsets >= total_combinations:
        # If requesting more than possible, use all combinations
        pylogger.info(f"Requested {num_subsets} subsets but only {total_combinations} exist. Using all.")
        all_subsets = list(combinations(range(n_datasets), subset_size))
        num_subsets = len(all_subsets)
    else:
        # Random sample without replacement
        all_indices = list(combinations(range(n_datasets), subset_size))
        all_subsets = random.sample(all_indices, num_subsets)

    # Determine results path with run subfolder
    benchmark_name = cfg.benchmark.get("name", f"N{n_datasets}")
    run_folder = f"random_subsets_k{subset_size}_n{num_subsets}_{benchmark_name}"

    merger_name = cfg.merger._target_.split(".")[-2].replace("_merger", "")
    reg_suffix = getattr(cfg.misc, 'reg_suffix', '')
    if not reg_suffix:
        reg_suffix = generate_reg_suffix(cfg)
    merger_name_with_suffix = f"{merger_name}{reg_suffix}" if reg_suffix else merger_name
    results_path = Path(cfg.misc.results_path) / run_folder / merger_name_with_suffix

    all_results = {}
    skipped = 0

    for idx, subset_indices in enumerate(all_subsets):
        subset_datasets = [datasets[i] for i in subset_indices]
        subset_name = "__".join(d.name for d in subset_datasets)

        # Check if result already exists
        alignment_suffix = "_rot_aligned" if cfg.alignment else ""
        subset_file = results_path / f"subset_{subset_size}_{subset_name}{alignment_suffix}.json"
        if subset_file.exists():
            pylogger.info(f"[{idx+1}/{num_subsets}] Skipping {subset_name} (already exists)")
            with open(subset_file, 'r') as f:
                all_results[subset_name] = json.load(f)
            skipped += 1
            continue

        pylogger.info(f"\n{'='*60}")
        pylogger.info(f"[{idx+1}/{num_subsets}] Evaluating subset: {[d.name for d in subset_datasets]}")
        pylogger.info(f"{'='*60}\n")

        try:
            subset_results = run_single(
                cfg,
                datasets_to_use=subset_datasets,
                pair_name=subset_name,
                file_prefix=f"subset_{subset_size}"
            )
            all_results[subset_name] = subset_results
        except Exception as e:
            pylogger.error(f"Failed to evaluate subset {subset_name}: {e}")
            all_results[subset_name] = {"error": str(e)}

    pylogger.info(f"\nSkipped {skipped} existing subsets, evaluated {num_subsets - skipped} new subsets")

    # Save summary
    results_path.mkdir(parents=True, exist_ok=True)
    alignment_suffix = "_rot_aligned" if cfg.alignment else ""
    summary_file = results_path / f"summary{alignment_suffix}.json"
    with open(summary_file, "w+") as f:
        json.dump(all_results, f, indent=4)

    pylogger.info(f"\n{'='*60}")
    pylogger.info(f"RANDOM SUBSETS EVALUATION COMPLETE")
    pylogger.info(f"Summary saved to: {summary_file}")
    pylogger.info(f"{'='*60}")

    return all_results


def run(cfg: DictConfig):
    """Main entry point - handles single run, all_pairwise, and random_subsets modes.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Mode priority: random_subsets > all_pairwise > single run
    """
    all_pairwise = cfg.get("all_pairwise", False)
    random_subsets = cfg.get("random_subsets", False)

    if random_subsets and all_pairwise:
        pylogger.warning("Both random_subsets and all_pairwise are True. Using random_subsets mode (ignoring all_pairwise).")

    if random_subsets:
        # Random subsets mode
        return run_random_subsets(cfg)
    elif not all_pairwise:
        # Standard single run with all datasets in benchmark
        return run_single(cfg)

    # All pairwise mode: run merging for each pair of datasets
    datasets = list(cfg.benchmark.datasets)
    n_datasets = len(datasets)
    n_pairs = n_datasets * (n_datasets - 1) // 2

    pylogger.info(f"Running ALL PAIRWISE merging evaluation")
    pylogger.info(f"Benchmark has {n_datasets} datasets: {[d.name for d in datasets]}")
    pylogger.info(f"Total pairs to evaluate: {n_pairs}")

    # Determine results path for checking existing results
    merger_name = cfg.merger._target_.split(".")[-2].replace("_merger", "")
    reg_suffix = getattr(cfg.misc, 'reg_suffix', '')
    if not reg_suffix:
        reg_suffix = generate_reg_suffix(cfg)
    merger_name_with_suffix = f"{merger_name}{reg_suffix}" if reg_suffix else merger_name
    results_path = Path(cfg.misc.results_path) / merger_name_with_suffix

    all_results = {}
    pair_idx = 0
    skipped = 0

    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            pair_idx += 1
            dataset_i = datasets[i]
            dataset_j = datasets[j]
            pair_name = f"{dataset_i.name}__{dataset_j.name}"

            # Check if result already exists - skip if so
            alignment_suffix = "_rot_aligned" if cfg.alignment else ""
            pair_file = results_path / f"pair_{pair_name}{alignment_suffix}.json"
            if pair_file.exists():
                pylogger.info(f"[{pair_idx}/{n_pairs}] Skipping {pair_name} (already exists)")
                with open(pair_file, 'r') as f:
                    all_results[pair_name] = json.load(f)
                skipped += 1
                continue

            pylogger.info(f"\n{'='*60}")
            pylogger.info(f"[{pair_idx}/{n_pairs}] Evaluating pair: {pair_name}")
            pylogger.info(f"{'='*60}\n")

            try:
                pair_results = run_single(
                    cfg,
                    datasets_to_use=[dataset_i, dataset_j],
                    pair_name=pair_name
                )
                all_results[pair_name] = pair_results
            except Exception as e:
                pylogger.error(f"Failed to evaluate pair {pair_name}: {e}")
                all_results[pair_name] = {"error": str(e)}

    pylogger.info(f"\nSkipped {skipped} existing pairs, evaluated {n_pairs - skipped} new pairs")

    # Save summary of all pairwise results
    # Extract merger name from target (e.g., "model_merging.merger.weight_avg_merger.WeightAvgMerger" -> "weight_avg")
    merger_name = cfg.merger._target_.split(".")[-2].replace("_merger", "")

    # Add regularization suffix to merger name if specified
    reg_suffix = getattr(cfg.misc, 'reg_suffix', '')
    if not reg_suffix:
        reg_suffix = generate_reg_suffix(cfg)
    merger_name_with_suffix = f"{merger_name}{reg_suffix}" if reg_suffix else merger_name

    # Create merger-specific folder with regularization suffix
    results_path = Path(cfg.misc.results_path) / merger_name_with_suffix
    results_path.mkdir(parents=True, exist_ok=True)

    # Get benchmark name from config (e.g., "N8", "N20")
    benchmark_name = cfg.benchmark.get("name", f"N{n_datasets}")
    alignment_suffix = "_rot_aligned" if cfg.alignment else ""
    summary_file = results_path / f"all_pairwise_summary_{benchmark_name}{alignment_suffix}.json"
    with open(summary_file, "w+") as f:
        json.dump(all_results, f, indent=4)

    pylogger.info(f"\n{'='*60}")
    pylogger.info(f"ALL PAIRWISE EVALUATION COMPLETE")
    pylogger.info(f"Summary saved to: {summary_file}")
    pylogger.info(f"{'='*60}")

    return all_results


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="multitask.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()