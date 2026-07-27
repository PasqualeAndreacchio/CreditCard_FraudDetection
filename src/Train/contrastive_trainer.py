from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.Autoencoder import ContrastiveModel

logger = logging.getLogger(__name__)


class ContrastiveTrainer:
    """
    Manages the contrastive pre-training of a ContrastiveModel.
    
    Uses TripletMarginLoss to push normal-transaction embeddings together
    and pull fraud-transaction embeddings apart. Each batch is expected to
    be an (anchor, positive, negative) triplet, as produced by ContrastiveDataset.
    
    At the end of training only the backbone encoder is saved to disk —
    the projection head is discarded, since downstream tasks only need
    the feature extractor.
    """

    def __init__(
        self,
        model: ContrastiveModel,
        config: dict[str, Any],
    ) -> None:

        contrastive_cfg = config.get("contrastive", {})

        # Device & model
        self.device: torch.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = config
        self.model = model.to(self.device)

        # Loss: L(a, p, n) = max(d(a,p) - d(a,n) + margin, 0)
        # where d is the Lp-distance between L2-normalised projections.
        margin: float = contrastive_cfg.get("margin", 1.0)
        p: int = contrastive_cfg.get("p", 2)
        self.criterion = nn.TripletMarginLoss(margin=margin, p=p)

        # Optimizer
        lr: float = contrastive_cfg.get("learning_rate", 1e-3)
        wd: float = contrastive_cfg.get("weight_decay", 0.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=wd
        )

        # Where the backbone will be saved after training
        self.backbone_save_path: str = contrastive_cfg.get(
            "backbone_save_path", "pretrained_tabular_encoder.pth"
        )

        # Training history
        self.history: dict[str, list[float]] = {"train_loss": []}

    
    def fit(self, train_loader: DataLoader) -> dict[str, list[float]]:
        """
        Run the full contrastive training loop and save the backbone encoder.
        Each batch from train_loader must be an (anchor, positive, negative)
        triplet of float32 tensors, as yielded by ContrastiveDataset.
        Args:
            train_loader: DataLoader wrapping a ContrastiveDataset.
        Returns:
            Training history dictionary with key 'train_loss', containing
            one average loss value per epoch.
        """
        epochs: int = self.config.get("contrastive", {}).get("epochs", 20)

        logger.info(
            "Starting contrastive pre-training for %d epochs on %s.",
            epochs,
            self.device,
        )
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            epoch_loss = self._train_epoch(train_loader, epoch, epochs)
            self.history["train_loss"].append(epoch_loss)
            logger.info(
                "Epoch %3d/%d  |  triplet_loss=%.8f", epoch, epochs, epoch_loss
            )

        elapsed = time.time() - t_start
        logger.info("Contrastive pre-training complete in %.1f s.", elapsed)

        # Save the backbone (projection head is discarded)
        self._save_backbone(self.backbone_save_path)

        return self.history


    def get_training_history(self) -> dict[str, list[float]]:
        """
        Return the training history accumulated during fit().
        Returns:
            Dictionary with key 'train_loss' and a list of per-epoch
            average triplet losses.
        """
        return self.history

    
    # Private helpers
    def _train_epoch(
        self, loader: DataLoader, epoch: int, total_epochs: int
    ) -> float:
        """
        Run a single training epoch.

        Args:
            loader:       DataLoader yielding (anchor, positive, negative) batches.
            epoch:        Current epoch number (1-indexed).
            total_epochs: Total number of epochs, for the progress bar label.
        Returns:
            Average triplet loss over all batches in this epoch.
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

        for anchor, positive, negative in pbar:
            anchor   = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            self.optimizer.zero_grad()

            # All three go through backbone + projection head.
            # Outputs are L2-normalised in ContrastiveModel.forward(), so
            # Euclidean distance equals angular distance here.
            proj_anchor = self.model(anchor)
            proj_pos    = self.model(positive)
            proj_neg    = self.model(negative)

            # Minimise distance to the positive, maximise distance to the negative
            loss = self.criterion(proj_anchor, proj_pos, proj_neg)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            n_batches    += 1
            pbar.set_postfix(loss=f"{loss.item():.8f}")

        return running_loss / max(n_batches, 1)


    def _save_backbone(self, path: str) -> None:
        """
        Save only the backbone encoder's weights to disk.
        The projection head is not saved — once pre-training is done,
        only the backbone is needed for downstream tasks. Parent directories
        are created automatically if they don't exist.
        Args:
            path: File path for the .pth checkpoint.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        torch.save(self.model.backbone.state_dict(), path)
        logger.info("Backbone encoder saved to '%s'.", path)
