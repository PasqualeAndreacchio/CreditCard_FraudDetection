"""
Training pipeline for the Autoencoder.

Handles:
    - Optimizer and LR scheduler setup (from config)
    - Training loop with progress bars
    - Validation at every epoch
    - Early stopping with configurable patience
    - Best-model checkpointing
    - Training history logging
"""

from __future__ import annotations

import os
import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score as sklearn_f1, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.Autoencoder import FraudAutoencoder

logger = logging.getLogger(__name__)


# ─── Loss Registry ──────────────────────────────────────────────────────────

_LOSSES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.SmoothL1Loss,
    "bce": nn.BCEWithLogitsLoss,
    "ce": nn.CrossEntropyLoss
}


def _build_loss(name: str, weight: torch.Tensor | None = None) -> nn.Module:
    """Return a loss module by name.
    
    Args:
        name: One of the keys in _LOSSES.
        weight: Optional class-weight tensor passed to CrossEntropyLoss
                (ignored for other loss types).
    """
    key = name.lower()
    if key not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(_LOSSES.keys())}.")
    if key == "ce" and weight is not None:
        return nn.CrossEntropyLoss(weight=weight)
    return _LOSSES[key]()


# ─── Early Stopping ─────────────────────────────────────────────────────────

class EarlyStopping:
    """Monitors validation loss and triggers early stopping.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum decrease in loss to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Update state with the latest validation loss.

        Args:
            val_loss: Current epoch's validation loss.

        Returns:
            ``True`` if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.patience,
                )

        return self.should_stop


# ─── Trainer ────────────────────────────────────────────────────────────────

class Trainer:
    """Orchestrates the training lifecycle of an :class:`FFNNAutoencoder`.

    Reads all hyperparameters (optimizer, scheduler, early stopping, etc.)
    from the configuration dictionary.

    Args:
        model: The :class:`FFNNAutoencoder` to train.
        config: Full configuration dictionary (parsed from YAML).
        device: Torch device for computation.

    Example::

        trainer = Trainer(model, config, device)
        history = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        class_weight: torch.Tensor | None = None,
    ) -> None:
        
        self.config = config
        self.device = config.get("device")
        self.model = model.to(self.device)

        training_cfg = config["training"]

        # Task
        self.task = config['model']['task']

        # Loss — passa i class weights a CrossEntropyLoss se forniti
        if class_weight is not None:
            class_weight = class_weight.to(self.device)
        self.criterion = _build_loss(training_cfg["loss"], weight=class_weight)

        # Optimizer
        self.optimizer = self._build_optimizer(training_cfg)

        # Scheduler
        self.scheduler = self._build_scheduler(training_cfg)

        # Early stopping
        es_cfg = training_cfg.get("early_stopping", {})
        self.early_stopping: EarlyStopping | None = None
        if es_cfg.get("enabled", False):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 10),
                min_delta=es_cfg.get("min_delta", 1e-4),
            )

        # Paths
        self.checkpoint_dir = config["paths"].get("checkpoint_dir", "checkpoints/")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "lr": [],
        }

    # ── Builder Helpers ─────────────────────────────────────────────────

    def _build_optimizer(self, training_cfg: dict) -> torch.optim.Optimizer:
        """Create an optimizer from configuration."""
        name = training_cfg.get("optimizer", "adam").lower()
        lr = training_cfg["learning_rate"]
        wd = training_cfg.get("weight_decay", 0.0)

        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer '{name}'. Use 'adam', 'adamw', or 'sgd'.")

    def _build_scheduler(
        self, training_cfg: dict
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Create an LR scheduler from configuration (or ``None``)."""
        sched_cfg = training_cfg.get("scheduler", {})
        sched_type = sched_cfg.get("type", "none").lower()

        if sched_type == "none":
            return None
        elif sched_type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=sched_cfg.get("patience", 5),
                factor=sched_cfg.get("factor", 0.5),
            )
        elif sched_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_cfg["epochs"],
            )
        elif sched_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.get("step_size", 30),
                gamma=sched_cfg.get("factor", 0.5),
            )
        else:
            raise ValueError(
                f"Unknown scheduler '{sched_type}'. "
                "Use 'reduce_on_plateau', 'cosine', 'step', or 'none'."
            )

    # ── Training Loop ───────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_labels: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            train_loader: DataLoader yielding training batches.
            val_loader: DataLoader yielding validation batches.
            val_labels: Optional ground-truth integer labels for the validation set
                (shape ``(N,)``). Required when ``val_metric = "f1"`` in the config.

        Returns:
            Training history dictionary with keys
            ``train_loss``, ``val_loss``, ``val_f1``, ``lr``.
        """
        epochs = self.config["training"]["epochs"]
        val_metric = self.config["training"].get("val_metric", "loss")
        use_f1 = (val_metric == "f1" and val_labels is not None)

        # For F1 metric: higher is better. For loss: lower is better.
        best_metric = 0.0 if use_f1 else float("inf")

        logger.info(
            "Starting training for up to %d epochs (checkpointing on %s).",
            epochs, "val_f1" if use_f1 else "val_loss",
        )
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            # ── Train ───────────────────────────────────────────────────────
            train_loss = self._train_epoch(train_loader, epoch, epochs)

            # ── Validate ────────────────────────────────────────────────────
            val_loss = self._validate_epoch(val_loader)
            val_f1 = self._compute_val_f1(val_loader, val_labels) if use_f1 else 0.0

            # ── Record ──────────────────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            logger.info(
                "Epoch %3d/%d  |  train_loss=%.6f  |  val_loss=%.6f  |  val_f1=%.4f  |  lr=%.2e",
                epoch, epochs, train_loss, val_loss, val_f1, current_lr,
            )

            # ── Checkpointing ───────────────────────────────────────────────
            metric_value = val_f1 if use_f1 else val_loss
            is_better = (metric_value > best_metric) if use_f1 else (metric_value < best_metric)

            if is_better:
                best_metric = metric_value
                checkpoint_name = self.config["paths"].get("checkpoint_name", "model_best_new.pt")
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, checkpoint_name),
                    epoch=epoch,
                    val_loss=val_loss,
                )
                logger.info(
                    "  ↳ Best model saved (%s=%.4f).",
                    "val_f1" if use_f1 else "val_loss", best_metric,
                )

            # ── Scheduler Step (sempre su val_loss) ──────────────────────────
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ── Early Stopping ──────────────────────────────────────
            if self.early_stopping is not None:
                # Per F1 (higher=better), passiamo il negativo all'EarlyStopping
                # che internamente cerca un valore decrescente
                es_value = -val_f1 if use_f1 else val_loss
                if self.early_stopping.step(es_value):
                    logger.info("Training stopped early at epoch %d.", epoch)
                    break

        elapsed = time.time() - t_start
        logger.info(
            "Training complete in %.1f s.  Best %s=%.4f",
            elapsed,
            "val_f1" if use_f1 else "val_loss",
            best_metric,
        )
        return self.history

    def _train_epoch(
        self, loader: DataLoader, epoch: int, total_epochs: int
    ) -> float:
        """Run a single training epoch.

        Args:
            loader: Training DataLoader.
            epoch: Current epoch number (1-indexed).
            total_epochs: Total number of epochs.

        Returns:
            Average training loss for this epoch.
        """
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch:3d}/{total_epochs}",
            leave=False,
            unit="batch",
        )

        for batch in pbar:
            # Support both plain tensors and (features, labels) tuples.
            # During training we only use the features.
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            self.optimizer.zero_grad()
            
            if self.task == "classification":
                y = batch[1]
                y = y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
            elif self.task == "reconstruction":
                x_hat = self.model(x)
                loss = self.criterion(x_hat, x)
            else:
                raise ValueError(f"The task {self.task} is not implemented")

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        return running_loss / max(n_batches, 1)

    def _validate_epoch(self, loader: DataLoader) -> float:
        """Run a single validation epoch (no gradients)."""
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(self.device)
                
                if self.task == "classification":
                    y = batch[1]
                    y = y.to(self.device)
                    y_pred = self.model(x)
                    loss = self.criterion(y_pred, y)
                    
                elif self.task == "reconstruction":
                    x_hat = self.model(x)
                    # Compare output to original input
                    loss = self.criterion(x_hat, x)
                    
                else:
                    raise ValueError(f"The task '{self.task}' is not implemented")

                running_loss += loss.item()
                n_batches += 1

        return running_loss / max(n_batches, 1)

    @torch.no_grad()
    def _compute_val_f1(
        self, loader: DataLoader, labels: np.ndarray
    ) -> float:
        """Compute the F1-optimal score on the validation set.

        Dynamically computes the correct anomaly score based on the task:
        - Classification: Uses the softmax probability of the positive class.
        - Reconstruction: Uses the Mean Squared Error (MSE) per sample.
        
        It then finds the threshold that maximises F1 via the precision-recall curve.

        Args:
            loader: Validation DataLoader.
            labels: Ground-truth binary integer labels (0=normal, 1=fraud).

        Returns:
            Best achievable F1-score on the validation set.
        """
        self.model.eval()
        all_scores: list[float] = []

        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            outputs = self.model(x)
            
            if self.task == "classification":
                # Score is the probability of the Fraud class (Class 1)
                probs = torch.softmax(outputs, dim=1)
                scores = probs[:, 1].cpu().numpy().tolist()
                
            elif self.task == "reconstruction":
                # Score is the Reconstruction Error (MSE) per individual sample.
                # Average across all dimensions except the batch dimension (dim 0).
                mse_per_sample = torch.mean((outputs - x) ** 2, dim=tuple(range(1, x.ndim)))
                scores = mse_per_sample.cpu().numpy().tolist()
                
            else:
                raise ValueError(f"The task '{self.task}' is not implemented")
                
            all_scores.extend(scores)

        scores = np.array(all_scores)

        # Find the threshold that maximises F1 on the validation set
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_values = 2 * precisions * recalls / (precisions + recalls)
        f1_values = np.nan_to_num(f1_values)

        best_idx = int(np.argmax(f1_values))
        best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
        preds = (scores > best_threshold).astype(int)

        return float(sklearn_f1(labels, preds, zero_division=0))

    # ── Checkpointing ──────────────────────────────────────────────────────

    def save_checkpoint(
        self, path: str, epoch: int = 0, val_loss: float = 0.0
    ) -> None:
        """Save a model checkpoint.

        The checkpoint includes model state, optimizer state, epoch, and
        validation loss — enough to resume training or run inference.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch number.
            val_loss: Current validation loss.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
            },
            path,
        )
        logger.debug("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """Load a model checkpoint and restore model/optimizer state.

        Args:
            path: Path to a ``.pt`` checkpoint file.

        Returns:
            The full checkpoint dictionary (with ``epoch``, ``val_loss``, etc.).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(
            "Checkpoint loaded from %s (epoch=%d, val_loss=%.6f).",
            path, checkpoint["epoch"], checkpoint["val_loss"],
        )
        return checkpoint

    def get_training_history(self) -> dict[str, list[float]]:
        """Return the training history dictionary.

        Returns:
            Dictionary with keys ``train_loss``, ``val_loss``, ``lr``.
        """
        return self.history
