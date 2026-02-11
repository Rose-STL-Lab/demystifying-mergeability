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
        )
        pylogger.info("Regularization configured:")
        pylogger.info(f"  R2 (Moderate Update): {cfg.train.regularization.enable_moderate_update}, λ={cfg.train.regularization.lambda_moderate_update}")
        pylogger.info(f"  R3 (Grad Magnitude): {cfg.train.regularization.enable_grad_magnitude}, λ={cfg.train.regularization.lambda_grad_magnitude}")
        pylogger.info(f"  TV Subspace Penalty: {enable_tv_subspace}, k={subspace_k}, vectors={tv_singular_vectors}, λ={lambda_tv_subspace}")

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
    trainer.test(model=model, dataloaders=dataset.test_loader)

    # Generate suffix and folder structure based on enabled regularizations
    reg_suffix = ""
    parent_folder = ""
    if hasattr(cfg.train, 'regularization'):
        reg_parts = []
        moderate_update_enabled = cfg.train.regularization.enable_moderate_update
        grad_magnitude_enabled = cfg.train.regularization.enable_grad_magnitude
        tv_subspace_enabled = cfg.train.regularization.get('enable_tv_subspace_penalty', False)

        if moderate_update_enabled:
            reg_parts.append("moderate_update")
        if grad_magnitude_enabled:
            reg_parts.append("grad_magnitude")
        if tv_subspace_enabled:
            reg_parts.append("tv_subspace")

        if reg_parts:
            reg_suffix = "_" + "_".join(reg_parts)

            # Determine parent folder based on which regularizations are enabled
            if tv_subspace_enabled:
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
