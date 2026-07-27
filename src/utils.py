"""
Utility functions for the Credit Card Fraud Detection pipeline.

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

    Supports both reconstruction (config.yaml) and classification
    (classification_config.yaml) configuration layouts.

    Raises:
        ValueError: On missing or malformed configuration entries.
    """
    # ── Required top-level sections ──────────────────────────────────
    required_sections = ["model", "training", "paths"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # ── Model section ────────────────────────────────────────────────
    model_cfg = config["model"]
    task = model_cfg.get("task")

    if task == "reconstruction":
        # Autoencoder layout: model.input_dim, model.hidden_dims
        for key in ["input_dim", "hidden_dims"]:
            if key not in model_cfg:
                raise ValueError(f"Missing required model key: '{key}'")
        if not isinstance(model_cfg["hidden_dims"], list) or len(model_cfg["hidden_dims"]) == 0:
            raise ValueError("'hidden_dims' must be a non-empty list of integers.")

    elif task == "classification":
        # Classifier layout: model.encoder and model.decoder sub-dicts
        for key in ["encoder", "decoder"]:
            if key not in model_cfg:
                raise ValueError(f"Missing required model key: '{key}'")

    elif task is None:
        raise ValueError("Missing required model key: 'task'. Must be 'reconstruction' or 'classification'.")
    else:
        raise ValueError(f"Unknown task '{task}'. Must be 'reconstruction' or 'classification'.")

    # ── Training section ─────────────────────────────────────────────
    training_cfg = config["training"]
    for key in ["epochs", "learning_rate", "loss"]:
        if key not in training_cfg:
            raise ValueError(f"Missing required training key: '{key}'")

    valid_losses = {"mse", "mae", "huber", "bce", "ce"}
    if training_cfg["loss"] not in valid_losses:
        raise ValueError(f"Invalid loss '{training_cfg['loss']}'. Must be one of {valid_losses}.")

    # ── Anomaly section (optional, only for reconstruction) ──────────
    anomaly_cfg = config.get("anomaly", {})
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