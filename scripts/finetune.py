import json
import logging
import os

from typing import Dict, List, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningModule
from tqdm import tqdm

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

from model_merging.model.encoder import ImageEncoder
from model_merging.model.heads import get_classification_head
from model_merging.model.image_classifier import ImageClassifier
from model_merging.utils.io_utils import (
    load_model_from_hf,
    upload_model_to_hf,
)
from hydra.utils import instantiate

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def compute_subspace_projectors(
    pretrained_state_dict: Dict[str, torch.Tensor],
    k: int,
    singular_vectors: str = "V",
) -> Dict[str, torch.Tensor]:
    """Compute truncated singular vectors for each 2D weight matrix.

    Args:
        pretrained_state_dict: State dict of pretrained encoder weights
        k: Number of top-k singular vectors to keep
        singular_vectors: "V" for right, "U" for left, "UV" for both

    Returns:
        Dict mapping parameter names to truncated singular vectors.
        For "V" or "U": maps to single tensor (Vk or Uk)
        For "UV": maps to dict {"U": Uk, "V": Vk}
    """
    subspace_projectors = {}

    for name, param in pretrained_state_dict.items():
        if param.ndim == 2:  # Only process 2D matrices
            # Perform SVD: W = U @ S @ V^T
            U, S, Vh = torch.linalg.svd(param.float(), full_matrices=False)
            # U shape: (out_features, min(out, in))
            # S shape: (min(out, in),)
            # Vh shape: (min(out, in), in_features)

            # Truncate to top-k
            actual_k = min(k, S.shape[0])

            if singular_vectors == "V":
                # V: right singular vectors, Vh[:k] gives top-k rows of V^T
                # Vk shape: (k, in_features)
                Vk = Vh[:actual_k, :]
                subspace_projectors[name] = Vk
            elif singular_vectors == "U":
                # U: left singular vectors, U[:, :k] gives top-k columns
                # Uk shape: (out_features, k), we transpose to (k, out_features)
                Uk = U[:, :actual_k].T
                subspace_projectors[name] = Uk
            elif singular_vectors == "UV":
                # Both U and V
                Vk = Vh[:actual_k, :]
                Uk = U[:, :actual_k].T
                subspace_projectors[name] = {"U": Uk, "V": Vk}

            pylogger.debug(f"Computed SVD for {name}: shape={param.shape}, k={actual_k}")

    return subspace_projectors


def compute_gargiulo_pretrained_svd(
    pretrained_state_dict: Dict[str, torch.Tensor],
    k: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute truncated SVD for Gargiulo penalty (always stores both U and V).

    Args:
        pretrained_state_dict: State dict of pretrained encoder weights
        k: Number of top-k singular vectors to keep

    Returns:
        Dict mapping parameter names to {"U": Uk, "V": Vk}
        Uk shape: (k, out_features)
        Vk shape: (k, in_features)
    """
    gargiulo_svd = {}

    for name, param in pretrained_state_dict.items():
        if param.ndim == 2:  # Only process 2D matrices
            # Perform full SVD and truncate (faster than svd_lowrank for moderate-sized matrices)
            U, S, Vh = torch.linalg.svd(param.float(), full_matrices=False)
            # U shape: (out_features, min(out, in))
            # Vh shape: (min(out, in), in_features)

            # Truncate to top-k
            actual_k = min(k, S.shape[0])

            # Store both U and V transposed (rows are singular vectors)
            Uk = U[:, :actual_k].T  # (k, out_features)
            Vk = Vh[:actual_k, :]   # (k, in_features)

            gargiulo_svd[name] = {"U": Uk, "V": Vk}

            pylogger.debug(f"Computed Gargiulo SVD for {name}: shape={param.shape}, k={actual_k}")

    return gargiulo_svd


def run(cfg: DictConfig):
    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )

    classification_head = get_classification_head(
        cfg.nn.encoder.model_name,
        cfg.dataset.name,
        ckpt_path=cfg.misc.ckpt_path,
        openclip_cachedir=cfg.misc.openclip_cachedir,
        device=cfg.device,
    )

    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )

    # Save pretrained encoder state dict for regularization
    pretrained_state_dict = {
        name: param.clone().detach()
        for name, param in zeroshot_encoder.named_parameters()
    }

    model: ImageClassifier = hydra.utils.instantiate(
        cfg.nn.module,
        encoder=zeroshot_encoder,
        classifier=classification_head,
        _recursive_=False,
    )

    model.task_name = cfg.dataset.name

    # Configure mergeability regularization if enabled
    if hasattr(cfg.train, 'regularization'):
        # Compute subspace projectors if TV subspace penalty is enabled
        subspace_projectors = None
        enable_tv_subspace = cfg.train.regularization.get('enable_tv_subspace_penalty', False)
        tv_singular_vectors = cfg.train.regularization.get('tv_penalty_singular_vectors', 'V')
        subspace_k = cfg.train.regularization.get('subspace_top_k', 10)
        lambda_tv_subspace = cfg.train.regularization.get('lambda_tv_subspace', 0.0)

        if enable_tv_subspace:
            pylogger.info(f"Computing SVD for TV subspace penalty (k={subspace_k}, vectors={tv_singular_vectors}, lambda={lambda_tv_subspace})...")
            subspace_projectors = compute_subspace_projectors(
                pretrained_state_dict=pretrained_state_dict,
                k=subspace_k,
                singular_vectors=tv_singular_vectors,
            )
            pylogger.info(f"Computed subspace projectors for {len(subspace_projectors)} 2D matrices")

        # Compute Gargiulo pretrained SVD if enabled
        gargiulo_pretrained_svd = None
        enable_gargiulo = cfg.train.regularization.get('enable_gargiulo_penalty', False)
        gargiulo_singular_vectors = cfg.train.regularization.get('gargiulo_singular_vectors', 'U')
        gargiulo_top_k = cfg.train.regularization.get('gargiulo_top_k', 10)
        lambda_gargiulo = cfg.train.regularization.get('lambda_gargiulo', 0.0)
        gargiulo_svd_interval = cfg.train.regularization.get('gargiulo_svd_interval', 50)

        if enable_gargiulo:
            pylogger.info(f"Computing SVD for Gargiulo penalty (k={gargiulo_top_k}, vectors={gargiulo_singular_vectors}, lambda={lambda_gargiulo}, interval={gargiulo_svd_interval})...")
            gargiulo_pretrained_svd = compute_gargiulo_pretrained_svd(
                pretrained_state_dict=pretrained_state_dict,
                k=gargiulo_top_k,
            )
            pylogger.info(f"Computed Gargiulo pretrained SVD for {len(gargiulo_pretrained_svd)} 2D matrices")

        model.set_regularization_config(
            pretrained_state_dict=pretrained_state_dict,
            enable_moderate_update=cfg.train.regularization.enable_moderate_update,
            lambda_moderate_update=cfg.train.regularization.lambda_moderate_update,
            enable_grad_magnitude=cfg.train.regularization.enable_grad_magnitude,
            lambda_grad_magnitude=cfg.train.regularization.lambda_grad_magnitude,
            enable_tv_subspace_penalty=enable_tv_subspace,
            tv_penalty_singular_vectors=tv_singular_vectors,
            subspace_k=subspace_k,
            lambda_tv_subspace=lambda_tv_subspace,
            subspace_projectors=subspace_projectors,
            enable_gargiulo_penalty=enable_gargiulo,
            gargiulo_singular_vectors=gargiulo_singular_vectors,
            gargiulo_top_k=gargiulo_top_k,
            lambda_gargiulo=lambda_gargiulo,
            gargiulo_svd_interval=gargiulo_svd_interval,
            gargiulo_pretrained_svd=gargiulo_pretrained_svd,
        )
        pylogger.info("Regularization configured:")
        pylogger.info(f"  R2 (Moderate Update): {cfg.train.regularization.enable_moderate_update}, λ={cfg.train.regularization.lambda_moderate_update}")
        pylogger.info(f"  R3 (Grad Magnitude): {cfg.train.regularization.enable_grad_magnitude}, λ={cfg.train.regularization.lambda_grad_magnitude}")
        pylogger.info(f"  TV Subspace Penalty: {enable_tv_subspace}, k={subspace_k}, vectors={tv_singular_vectors}, λ={lambda_tv_subspace}")
        pylogger.info(f"  Gargiulo Penalty: {enable_gargiulo}, k={gargiulo_top_k}, vectors={gargiulo_singular_vectors}, λ={lambda_gargiulo}, interval={gargiulo_svd_interval}")

    dataset = instantiate(
        cfg.dataset,
        preprocess_fn=zeroshot_encoder.val_preprocess,
        batch_size=cfg.train.batch_size,
    )

    model.freeze_head()

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=cfg.core.storage_dir,
        logger=logger,
        enable_checkpointing=False,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(
        model=model,
        train_dataloaders=dataset.train_loader,
    )

    pylogger.info("Starting testing!")
    test_results = trainer.test(model=model, dataloaders=dataset.test_loader)

    # Generate suffix and folder structure based on enabled regularizations
    reg_suffix = ""
    parent_folder = ""
    if hasattr(cfg.train, 'regularization'):
        reg_parts = []
        moderate_update_enabled = cfg.train.regularization.enable_moderate_update
        grad_magnitude_enabled = cfg.train.regularization.enable_grad_magnitude
        tv_subspace_enabled = cfg.train.regularization.get('enable_tv_subspace_penalty', False)
        gargiulo_enabled = cfg.train.regularization.get('enable_gargiulo_penalty', False)

        if moderate_update_enabled:
            mu_lambda = cfg.train.regularization.get('lambda_moderate_update', 0.01)
            reg_parts.append(f"moderate_update_{mu_lambda}")
        if grad_magnitude_enabled:
            gm_lambda = cfg.train.regularization.get('lambda_grad_magnitude', 1)
            reg_parts.append(f"grad_magnitude_{gm_lambda}")
        if tv_subspace_enabled:
            tv_vectors = cfg.train.regularization.get('tv_penalty_singular_vectors', 'V').lower()
            tv_lambda = cfg.train.regularization.get('lambda_tv_subspace', 0.001)
            reg_parts.append(f"tv_subspace_{tv_vectors}_{tv_lambda}")
        if gargiulo_enabled:
            g_vectors = cfg.train.regularization.get('gargiulo_singular_vectors', 'U').lower()
            g_lambda = cfg.train.regularization.get('lambda_gargiulo', 0.0001)
            reg_parts.append(f"gargiulo_{g_vectors}_{g_lambda}")

        if reg_parts:
            reg_suffix = "_" + "_".join(reg_parts)

            # Determine parent folder based on which regularizations are enabled
            if gargiulo_enabled:
                # Gargiulo penalty gets its own dedicated folder
                parent_folder = "gargiulo_penalty"
            elif tv_subspace_enabled:
                # TV subspace penalty gets its own dedicated folder
                parent_folder = "weight_space_subspace_penalty"
            elif moderate_update_enabled and grad_magnitude_enabled:
                parent_folder = "both"
            elif moderate_update_enabled:
                parent_folder = "moderate_update"
            elif grad_magnitude_enabled:
                parent_folder = "grad_magnitude"

    # Save model locally with regularization suffix in appropriate folder
    if parent_folder:
        # Create parent folder first
        parent_dir = os.path.join(cfg.misc.ckpt_path, parent_folder)
        os.makedirs(parent_dir, exist_ok=True)

        # Create checkpoint folder inside parent folder
        local_model_dir = os.path.join(
            parent_dir,
            f"{cfg.dataset.name}{reg_suffix}"
        )
    else:
        # No regularization, save directly in ckpt_path
        local_model_dir = os.path.join(
            cfg.misc.ckpt_path,
            cfg.dataset.name
        )

    os.makedirs(local_model_dir, exist_ok=True)
    local_model_path = os.path.join(local_model_dir, "model.pt")
    torch.save(model.encoder.state_dict(), local_model_path)
    pylogger.info(f"Saved model locally to {local_model_path}")

    # Save test accuracy to JSON results file
    # Extract test accuracy from test_results
    test_acc_key = f"acc/test/{cfg.dataset.name}"
    test_accuracy = None
    if test_results and len(test_results) > 0:
        test_accuracy = test_results[0].get(test_acc_key)

    if test_accuracy is not None:
        # Generate filename suffix (replace dots with underscores for filesystem safety)
        if reg_suffix:
            # Remove leading underscore and replace dots with underscores
            file_suffix = reg_suffix[1:].replace(".", "_")  # e.g., "gargiulo_u_0_001"
        else:
            file_suffix = "baseline"

        # Results file path
        results_dir = os.path.join(PROJECT_ROOT, "results", "finetuning")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"accs_{file_suffix}.json")

        # Load existing results or create new dict
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results_dict = json.load(f)
        else:
            results_dict = {}

        # Get model name (e.g., "ViT-B-32", "ViT-L-14")
        model_name = cfg.nn.encoder.model_name

        # Initialize model entry if not exists
        if model_name not in results_dict:
            results_dict[model_name] = {}

        # Add/update dataset accuracy
        results_dict[model_name][cfg.dataset.name] = test_accuracy

        # Save results
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        pylogger.info(f"Saved test accuracy ({test_accuracy:.4f}) to {results_file}")
    else:
        pylogger.warning(f"Could not extract test accuracy from results: {test_results}")

    # Upload to HuggingFace with regularization suffix in dataset name (optional)
    # Uncomment if you have HuggingFace credentials configured
    # dataset_name_with_suffix = f"{cfg.dataset.name}{reg_suffix}"
    # upload_model_to_hf(model.encoder, cfg.nn.encoder.model_name, dataset_name_with_suffix)

    logger.log_configuration(model, cfg)

    if logger is not None:
        logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
