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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.Model import Complete_Autoencoder

logger = logging.getLogger(__name__)


# ─── Loss Registry ──────────────────────────────────────────────────────────

_LOSSES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.SmoothL1Loss,
    "bce": nn.BCEWithLogitsLoss
}


def _build_loss(name: str) -> nn.Module:
    """Return a loss module by name."""
    key = name.lower()
    if key not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(_LOSSES.keys())}.")
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
        model: Complete_Autoencoder,
        config: dict[str, Any],
    ) -> None:
        
        self.config = config
        self.device = config.get("device")
        self.model = model.to(self.device)

        training_cfg = config["training"]

        # Task
        self.task = config['model']['task']

        # Loss
        self.criterion = _build_loss(training_cfg["loss"])

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
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            train_loader: DataLoader yielding batches of *normal* transactions
                (tensors of shape ``(batch, input_dim)``).
            val_loader: DataLoader yielding validation batches (same format).

        Returns:
            Training history dictionary with keys
            ``train_loss``, ``val_loss``, ``lr``.
        """
        epochs = self.config["training"]["epochs"]
        best_val_loss = float("inf")

        logger.info("Starting training for up to %d epochs.", epochs)
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            # ── Train ───────────────────────────────────────────────
            train_loss = self._train_epoch(train_loader, epoch, epochs)

            # ── Validate ────────────────────────────────────────────
            val_loss = self._validate_epoch(val_loader)

            # ── Record ──────────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            logger.info(
                "Epoch %3d/%d  |  train_loss=%.6f  |  val_loss=%.6f  |  lr=%.2e",
                epoch, epochs, train_loss, val_loss, current_lr,
            )

            # ── Checkpointing ───────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, "Autoencoder_best.pt"),
                    epoch=epoch,
                    val_loss=val_loss,
                )
                logger.info("  ↳ Best model saved (val_loss=%.6f).", val_loss)

            # ── Scheduler Step ──────────────────────────────────────
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ── Early Stopping ──────────────────────────────────────
            if self.early_stopping is not None:
                if self.early_stopping.step(val_loss):
                    logger.info("Training stopped early at epoch %d.", epoch)
                    break

        elapsed = time.time() - t_start
        logger.info(
            "Training complete in %.1f s.  Best val_loss=%.6f",
            elapsed, best_val_loss,
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
            elif self.task == "recostruction":
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
        """Run a single validation epoch (no gradients).

        Args:
            loader: Validation DataLoader.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    y = batch[1]
                else:
                    x = batch

                x = x.to(self.device)
                x_hat = self.model(x)
                loss = self.criterion(x_hat, y)

                running_loss += loss.item()
                n_batches += 1

        return running_loss / max(n_batches, 1)

    # ── Checkpointing ───────────────────────────────────────────────────

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
