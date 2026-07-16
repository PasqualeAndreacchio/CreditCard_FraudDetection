"""
Training entry-point for the FFNN Autoencoder.

Usage::

    python train_ffnn.py --config configs/ffnn_config.yaml

This script:
    1. Loads the YAML configuration
    2. Sets seed and device for reproducibility
    3. Loads preprocessed data (normal-only for training, mixed for validation)
    4. Instantiates the FFNNAutoencoder from config
    5. Trains the model with early stopping and checkpointing
    6. Saves the best model and training history plot
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, set_seed, get_device, setup_logging
from src.models.FFNNDecoder import FFNNAutoencoder
from src.Train.trainer import Trainer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the FFNN Autoencoder for credit card fraud detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ffnn_config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_data(
    config: dict,
) -> tuple[DataLoader, DataLoader]:
    """Load preprocessed data and build DataLoaders.

    This function attempts to load the data from the preprocessing module.
    If the preprocessing module is not yet available, it falls back to
    loading raw data directly and performs a basic split.

    Args:
        config: Full configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader).
        train_loader contains *only normal* transactions.
        val_loader contains *all* transactions (for threshold tuning).
    """
    import pandas as pd

    batch_size = config["training"]["batch_size"]
    data_dir = config["paths"]["data_dir"]

    # ── Try loading preprocessed tensors ────────────────────────────
    preprocessed_path = os.path.join(data_dir, "preprocessed")
    if os.path.isdir(preprocessed_path):
        logger.info("Loading preprocessed data from %s", preprocessed_path)
        X_train = torch.load(os.path.join(preprocessed_path, "X_train.pt"), weights_only=True)
        y_train = torch.load(os.path.join(preprocessed_path, "y_train.pt"), weights_only=True)
        X_val = torch.load(os.path.join(preprocessed_path, "X_val.pt"), weights_only=True)
        y_val = torch.load(os.path.join(preprocessed_path, "y_val.pt"), weights_only=True)
    else:
        # ── Fallback: load raw CSV and do a basic split ─────────────
        logger.warning(
            "Preprocessed data not found at %s. "
            "Falling back to raw CSV loading.",
            preprocessed_path,
        )
        csv_path = os.path.join(data_dir, "creditcard.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        features = df.drop(columns=["Class"]).values.astype(np.float32)
        labels = df["Class"].values.astype(np.float32)

        # Standard scaling (basic — the preprocessing module will do better)
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features = (features - mean) / std

        # Train/Val split (80/20) — stratified
        from sklearn.model_selection import train_test_split

        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            features, labels, test_size=0.2, random_state=config.get("seed", 42),
            stratify=labels,
        )

        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)
        X_val = torch.tensor(X_val_np, dtype=torch.float32)
        y_val = torch.tensor(y_val_np, dtype=torch.float32)

    # ── Filter: train only on normal transactions ───────────────────
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    logger.info(
        "Training data: %d normal samples (filtered from %d total).",
        X_train_normal.shape[0], X_train.shape[0],
    )
    logger.info("Validation data: %d total samples.", X_val.shape[0])

    train_dataset = TensorDataset(X_train_normal)
    val_dataset = TensorDataset(X_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
    )

    return train_loader, val_loader


def plot_training_history(
    history: dict[str, list[float]], save_dir: str
) -> None:
    """Save training curves (loss and learning rate).

    Args:
        history: Dictionary with ``train_loss``, ``val_loss``, ``lr``.
        save_dir: Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss curves ─────────────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#3498db", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#e74c3c", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("FFNN Autoencoder — Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Learning rate ───────────────────────────────────────────────
    ax2.plot(epochs, history["lr"], color="#2ecc71", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "ffnn_training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Training history plot saved to %s", save_path)


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    # ── Configuration ───────────────────────────────────────────────
    config = load_config(args.config)
    setup_logging(config["paths"].get("log_dir"))
    set_seed(config.get("seed", 42))
    device = get_device(config)

    # ── Data ────────────────────────────────────────────────────────
    train_loader, val_loader = load_data(config)

    # ── Model ───────────────────────────────────────────────────────
    model = FFNNAutoencoder.from_config(config["model"])
    logger.info("\n%s", model.summary())

    # ── Training ────────────────────────────────────────────────────
    trainer = Trainer(model, config, device)
    history = trainer.fit(train_loader, val_loader)

    # ── Post-training ───────────────────────────────────────────────
    plot_training_history(history, config["paths"].get("log_dir", "logs/"))

    logger.info("Training complete. Best model saved to %s",
                os.path.join(config["paths"]["checkpoint_dir"], "ffnn_autoencoder_best.pt"))


if __name__ == "__main__":
    main()
