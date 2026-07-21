"""
Utility functions for the FFNN Autoencoder pipeline.

Provides:
    - YAML configuration loading and validation
    - Reproducibility seeding (torch, numpy, random)
    - Device detection
    - Logging setup
    - Model parameter counting
"""

import os
import random
import logging
from typing import Any

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────────────

def load_config(path: str) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required top-level keys are missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    """Validate that all required sections and keys are present.

    Raises:
        ValueError: On missing or malformed configuration entries.
    """
    required_sections = ["model", "training", "anomaly", "paths"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # Model section
    model_cfg = config["model"]
    for key in ["input_dim", "encoder_layers", "latent_dim", "decoder_layers"]:
        if key not in model_cfg:
            raise ValueError(f"Missing required model key: '{key}'")

    if not isinstance(model_cfg["encoder_layers"], list) or len(model_cfg["encoder_layers"]) == 0:
        raise ValueError("'encoder_layers' must be a non-empty list of integers.")
    if not isinstance(model_cfg["decoder_layers"], list) or len(model_cfg["decoder_layers"]) == 0:
        raise ValueError("'decoder_layers' must be a non-empty list of integers.")

    # Training section
    training_cfg = config["training"]
    for key in ["epochs", "batch_size", "learning_rate", "loss"]:
        if key not in training_cfg:
            raise ValueError(f"Missing required training key: '{key}'")

    valid_losses = {"mse", "mae", "huber"}
    if training_cfg["loss"] not in valid_losses:
        raise ValueError(f"Invalid loss '{training_cfg['loss']}'. Must be one of {valid_losses}.")

    # Anomaly section
    anomaly_cfg = config["anomaly"]
    valid_methods = {"percentile", "mean_std", "f1_optimal"}
    if anomaly_cfg.get("threshold_method", "percentile") not in valid_methods:
        raise ValueError(
            f"Invalid threshold_method '{anomaly_cfg['threshold_method']}'. "
            f"Must be one of {valid_methods}."
        )

    logger.debug("Configuration validated successfully.")


# ─── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Deterministic behaviour (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d.", seed)


# ─── Device ─────────────────────────────────────────────────────────────────

def get_device(config: dict) -> torch.device:
    """Resolve the compute device from configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        torch.device for model and data placement.
    """
    requested = config.get("device", "cpu")
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    logger.info("Using device: %s", device)
    return device


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str | None = None, level: int = logging.INFO) -> None:
    """Configure root logger with console and optional file handler.

    Args:
        log_dir: If provided, logs are also written to ``log_dir/training.log``.
        level: Logging level (default: INFO).
    """
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "training.log"), encoding="utf-8"
        )
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    logger.debug("Logging initialised (level=%s, log_dir=%s).", level, log_dir)


# ─── Model Helpers ──────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: A PyTorch module.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape (batch_size, dim) from the projection head.
            labels: Tensor of shape (batch_size,) containing the class labels.
        """
        
        # Normalize embeddings to calculate cosine similarity via dot product
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute the cosine similarity matrix
        # Resulting shape: (batch_size, batch_size)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create the mask for positive pairs based on labels
        # Reshape labels to (batch_size, 1) to compare every label with every other label
        labels = labels.contiguous().view(-1, 1)
        # mask[i, j] = 1 if labels[i] == labels[j], else 0
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-comparisons
        # The model shouldn't get "free points" for matching an item to itself
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask
        
        # Numerical Stability Trick
        # Subtract the max similarity in each row to prevent exploding exponentials
        sim_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - sim_max.detach()
        
        # Compute log probabilities
        # Mask out self-comparisons from the denominator sum
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)
        
        # Compute the mean log-likelihood for positive pairs
        # Find how many positive pairs exist for each item in the batch
        mask_sum = mask.sum(dim=1)
        
        # Edge case: If a transaction is the ONLY one of its class in the batch (e.g., 1 Fraud),
        # prevent division by zero by setting its sum to 1. 
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        # Multiply log_prob by the mask to only keep scores for positive pairs,
        # sum them up, and divide by the number of positives.
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        
        # 8. Final loss is the negative mean over the entire batch
        loss = -mean_log_prob_pos.mean()
        
        return loss