import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim import Optimizer

from nn_core.model_logging import NNLogger

from model_merging.data.datamodule import MetaData
from model_merging.data.dataset import maybe_dictionarize
from model_merging.utils.utils import torch_load, torch_save

pylogger = logging.getLogger(__name__)


class ImageClassifier(pl.LightningModule):
    logger: NNLogger

    def __init__(
        self, encoder, classifier, metadata: Optional[MetaData] = None, *args, **kwargs
    ) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata
        self.num_classes = classifier.out_features

        metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes, top_k=1
        )
        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()

        self.encoder = encoder
        self.classification_head = classifier

        self.log_fn = lambda metric, val: self.log(
            metric, val, on_step=True, on_epoch=True
        )

        self.finetuning_accuracy = None

        # Mergeability regularization configuration
        self.pretrained_state_dict = None
        self.enable_moderate_update = False
        self.lambda_moderate_update = 0.0
        self.enable_grad_magnitude = False
        self.lambda_grad_magnitude = 0.0

        # TV subspace penalty configuration
        self.enable_tv_subspace_penalty = False
        self.tv_penalty_singular_vectors = "V"  # "V" or "U"
        self.subspace_k = 10
        self.lambda_tv_subspace = 0.0
        self.subspace_projectors = None  # Dict[name, Vk or Uk tensor]

        # Gargiulo penalty configuration
        self.enable_gargiulo_penalty = False
        self.gargiulo_singular_vectors = "U"  # "U", "V", or "UV"
        self.gargiulo_top_k = 10
        self.lambda_gargiulo = 0.0
        self.gargiulo_svd_interval = 50  # Compute SVD every N steps
        self.gargiulo_pretrained_svd = None  # Dict[name, {"U": Uk, "V": Vk}]
        self.gargiulo_current_svd = None  # Cached current SVD (updated periodically)
        self.gargiulo_step_counter = 0  # Counter for SVD interval

    def set_encoder(self, encoder: torch.nn.Module):
        """Set the encoder of the model.

        Args:
            encoder (torch.nn.Module): The new encoder to set.
        """
        self.encoder = encoder

    def set_head(self, head: torch.nn.Module):
        """Set the classification head of the model.

        Args:
            head (torch.nn.Module): The new classification head to set.
        """
        self.classification_head = head

    def set_metrics(self, num_classes):

        self.num_classes = num_classes

        metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )

        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        embeddings = self.encoder(x)

        logits = self.classification_head(embeddings)

        return logits

    def _step(self, batch: Dict[str, torch.Tensor], split: str) -> Mapping[str, Any]:
        batch = maybe_dictionarize(batch, self.hparams.x_key, self.hparams.y_key)

        x = batch[self.hparams.x_key]
        gt_y = batch[self.hparams.y_key]

        logits = self(x)

        loss = F.cross_entropy(logits, gt_y)
        preds = torch.softmax(logits, dim=-1)

        metrics = getattr(self, f"{split}_acc")
        metrics.update(preds, gt_y)

        self.log_fn(f"acc/{split}/{self.task_name}", metrics)
        self.log_fn(f"loss/{split}/{self.task_name}", loss)

        return {"logits": logits.detach(), "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        result = self._step(batch=batch, split="train")
        loss = result["loss"]

        # Add mergeability regularization terms
        if self.pretrained_state_dict is not None:
            reg_loss = 0.0

            # R2: Moderate Update Regularization
            # Penalize large deviations from pretrained weights
            if self.enable_moderate_update and self.lambda_moderate_update > 0:
                moderate_update_loss = 0.0
                for name, param in self.encoder.named_parameters():
                    if name in self.pretrained_state_dict and param.requires_grad:
                        pretrained_param = self.pretrained_state_dict[name].to(param.device)
                        moderate_update_loss += torch.sum((param - pretrained_param) ** 2)

                reg_loss += self.lambda_moderate_update * moderate_update_loss
                self.log_fn(f"reg/moderate_update/{self.task_name}", moderate_update_loss.detach())

            # R3: Gradient Magnitude Regularization
            # Encourage moderate gradient magnitudes
            if self.enable_grad_magnitude and self.lambda_grad_magnitude > 0:
                grad_magnitude_loss = 0.0
                for param in self.encoder.parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_magnitude_loss += torch.sum(param.grad ** 2)

                # Only add if gradients exist (after first backward pass)
                if grad_magnitude_loss > 0:
                    reg_loss += self.lambda_grad_magnitude * grad_magnitude_loss
                    self.log_fn(f"reg/grad_magnitude/{self.task_name}", grad_magnitude_loss.detach())

            # TV Subspace Penalty Regularization
            # Penalize task vectors that go outside the dominant top-k subspace
            if self.enable_tv_subspace_penalty and self.lambda_tv_subspace > 0 and self.subspace_projectors is not None:
                tv_subspace_loss = 0.0
                for name, param in self.encoder.named_parameters():
                    if name in self.subspace_projectors and name in self.pretrained_state_dict and param.requires_grad:
                        pretrained_param = self.pretrained_state_dict[name].to(param.device)
                        task_vector = param - pretrained_param  # shape: (out_features, in_features)

                        projector_data = self.subspace_projectors[name]

                        if self.tv_penalty_singular_vectors == "V":
                            # V case: penalty = ||task_vector @ (I - Vk^T @ Vk)||_F^2
                            # Vk shape: (k, in_features), task_vector shape: (out, in)
                            Vk = projector_data.to(param.device)
                            projection = task_vector @ Vk.T @ Vk  # (out, in)
                            orthogonal_component = task_vector - projection
                            tv_subspace_loss += torch.sum(orthogonal_component ** 2)

                        elif self.tv_penalty_singular_vectors == "U":
                            # U case: penalty = ||(I - Uk @ Uk^T) @ task_vector||_F^2
                            # Uk shape: (k, out_features), task_vector shape: (out, in)
                            Uk = projector_data.to(param.device)
                            projection = Uk.T @ Uk @ task_vector  # (out, in)
                            orthogonal_component = task_vector - projection
                            tv_subspace_loss += torch.sum(orthogonal_component ** 2)

                        elif self.tv_penalty_singular_vectors == "UV":
                            # UV case: combine both penalties
                            Uk = projector_data["U"].to(param.device)
                            Vk = projector_data["V"].to(param.device)

                            # V penalty: orthogonal to right singular vectors
                            projection_v = task_vector @ Vk.T @ Vk
                            orthogonal_v = task_vector - projection_v
                            tv_subspace_loss += torch.sum(orthogonal_v ** 2)

                            # U penalty: orthogonal to left singular vectors
                            projection_u = Uk.T @ Uk @ task_vector
                            orthogonal_u = task_vector - projection_u
                            tv_subspace_loss += torch.sum(orthogonal_u ** 2)

                if tv_subspace_loss > 0:
                    reg_loss += self.lambda_tv_subspace * tv_subspace_loss
                    self.log_fn(f"reg/tv_subspace/{self.task_name}", tv_subspace_loss.detach())

            # Gargiulo Penalty Regularization
            # Penalize rotation of top-k singular vectors from pretrained
            if self.enable_gargiulo_penalty and self.lambda_gargiulo > 0 and self.gargiulo_pretrained_svd is not None:
                # Update current SVD periodically
                self.gargiulo_step_counter += 1
                if self.gargiulo_current_svd is None or self.gargiulo_step_counter % self.gargiulo_svd_interval == 0:
                    self._update_gargiulo_current_svd()

                gargiulo_loss = 0.0
                for name, param in self.encoder.named_parameters():
                    if name in self.gargiulo_pretrained_svd and name in self.gargiulo_current_svd and param.requires_grad:
                        pretrained_svd = self.gargiulo_pretrained_svd[name]
                        current_svd = self.gargiulo_current_svd[name]

                        if self.gargiulo_singular_vectors in ["U", "UV"]:
                            # U penalty: ||I_k - U_pre^T @ U_current||_F^2
                            U_pre = pretrained_svd["U"].to(param.device)  # (k, out_features)
                            U_cur = current_svd["U"].to(param.device)  # (k, out_features)
                            k = U_pre.shape[0]
                            alignment_u = U_pre @ U_cur.T  # (k, k)
                            identity_k = torch.eye(k, device=param.device)
                            gargiulo_loss += torch.sum((identity_k - alignment_u) ** 2)

                        if self.gargiulo_singular_vectors in ["V", "UV"]:
                            # V penalty: ||I_k - V_pre^T @ V_current||_F^2
                            V_pre = pretrained_svd["V"].to(param.device)  # (k, in_features)
                            V_cur = current_svd["V"].to(param.device)  # (k, in_features)
                            k = V_pre.shape[0]
                            alignment_v = V_pre @ V_cur.T  # (k, k)
                            identity_k = torch.eye(k, device=param.device)
                            gargiulo_loss += torch.sum((identity_k - alignment_v) ** 2)

                if gargiulo_loss > 0:
                    reg_loss += self.lambda_gargiulo * gargiulo_loss
                    self.log_fn(f"reg/gargiulo/{self.task_name}", gargiulo_loss.detach())

            if reg_loss > 0:
                loss = loss + reg_loss
                self.log_fn(f"reg/total/{self.task_name}", reg_loss.detach())

        result["loss"] = loss
        return result

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="val")

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="test")

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return torch_load(filename)

    def set_task(self, task_name):
        self.task_name = task_name

    def set_finetuning_accuracy(self, finetuning_accuracy):
        self.finetuning_accuracy = finetuning_accuracy

    def set_regularization_config(
        self,
        pretrained_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        enable_moderate_update: bool = False,
        lambda_moderate_update: float = 0.0,
        enable_grad_magnitude: bool = False,
        lambda_grad_magnitude: float = 0.0,
        enable_tv_subspace_penalty: bool = False,
        tv_penalty_singular_vectors: str = "V",
        subspace_k: int = 10,
        lambda_tv_subspace: float = 0.0,
        subspace_projectors: Optional[Dict[str, torch.Tensor]] = None,
        enable_gargiulo_penalty: bool = False,
        gargiulo_singular_vectors: str = "U",
        gargiulo_top_k: int = 10,
        lambda_gargiulo: float = 0.0,
        gargiulo_svd_interval: int = 50,
        gargiulo_pretrained_svd: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        """Configure mergeability regularization.

        Args:
            pretrained_state_dict: State dict of pretrained encoder weights
            enable_moderate_update: Enable R2 (moderate update) regularization
            lambda_moderate_update: Weight for R2 regularization
            enable_grad_magnitude: Enable R3 (gradient magnitude) regularization
            lambda_grad_magnitude: Weight for R3 regularization
            enable_tv_subspace_penalty: Enable task vector subspace penalty
            tv_penalty_singular_vectors: "V" for right singular vectors, "U" for left
            subspace_k: Number of top-k singular vectors to keep
            lambda_tv_subspace: Weight for TV subspace penalty
            subspace_projectors: Dict mapping param names to Vk or Uk tensors
            enable_gargiulo_penalty: Enable Gargiulo singular vector alignment penalty
            gargiulo_singular_vectors: "U", "V", or "UV" for which singular vectors to align
            gargiulo_top_k: Number of top-k singular vectors to keep aligned
            lambda_gargiulo: Weight for Gargiulo penalty
            gargiulo_svd_interval: Compute SVD of current weights every N steps
            gargiulo_pretrained_svd: Dict mapping param names to {"U": Uk, "V": Vk}
        """
        # Store as CPU tensors, will be moved to device during training
        self.pretrained_state_dict = pretrained_state_dict
        self.enable_moderate_update = enable_moderate_update
        self.lambda_moderate_update = lambda_moderate_update
        self.enable_grad_magnitude = enable_grad_magnitude
        self.lambda_grad_magnitude = lambda_grad_magnitude

        # TV subspace penalty config
        self.enable_tv_subspace_penalty = enable_tv_subspace_penalty
        self.tv_penalty_singular_vectors = tv_penalty_singular_vectors
        self.subspace_k = subspace_k
        self.lambda_tv_subspace = lambda_tv_subspace
        self.subspace_projectors = subspace_projectors

        # Gargiulo penalty config
        self.enable_gargiulo_penalty = enable_gargiulo_penalty
        self.gargiulo_singular_vectors = gargiulo_singular_vectors
        self.gargiulo_top_k = gargiulo_top_k
        self.lambda_gargiulo = lambda_gargiulo
        self.gargiulo_svd_interval = gargiulo_svd_interval
        self.gargiulo_pretrained_svd = gargiulo_pretrained_svd
        self.gargiulo_current_svd = None
        self.gargiulo_step_counter = 0

    def _update_gargiulo_current_svd(self):
        """Compute truncated SVD of current weight matrices for Gargiulo penalty."""
        self.gargiulo_current_svd = {}
        k = self.gargiulo_top_k

        with torch.no_grad():
            for name, param in self.encoder.named_parameters():
                if name in self.gargiulo_pretrained_svd and param.ndim == 2:
                    # Use truncated SVD for efficiency
                    # svd_lowrank returns (U, S, V) where A ≈ U @ diag(S) @ V.T
                    U, S, V = torch.svd_lowrank(param.float(), q=k)
                    # U shape: (out_features, k) - columns are left singular vectors
                    # V shape: (in_features, k) - columns are right singular vectors
                    # Transpose both to match pretrained format (rows are vectors)
                    self.gargiulo_current_svd[name] = {
                        "U": U.T.detach(),  # (k, out_features)
                        "V": V.T.detach(),  # (k, in_features)
                    }

    def on_test_epoch_end(self):
        # Compute and log the test accuracy
        accuracy = self.test_acc.compute().cpu().item()
        self.log(f"acc/test/{self.task_name}", accuracy, on_step=False, on_epoch=True)

        if self.finetuning_accuracy is not None:
            normalized_acc = accuracy / self.finetuning_accuracy
            self.log(f"normalized_acc/test/{self.task_name}", normalized_acc, on_step=False, on_epoch=True)
