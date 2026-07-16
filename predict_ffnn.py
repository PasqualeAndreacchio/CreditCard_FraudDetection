"""
Prediction / Evaluation entry-point for the FFNN Autoencoder.

Usage::

    python predict_ffnn.py --config configs/ffnn_config.yaml \\
                           --checkpoint checkpoints/ffnn_autoencoder_best.pt

This script:
    1. Loads the YAML configuration and model checkpoint
    2. Loads the test dataset
    3. Computes anomaly scores (reconstruction error)
    4. Determines the anomaly threshold
    5. Generates binary predictions and a full evaluation report
    6. Saves diagnostic plots
"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, set_seed, get_device, setup_logging
from src.models.FFNNDecoder import FFNNAutoencoder
from src.evaluator import Evaluator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained FFNN Autoencoder on the test set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ffnn_config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ffnn_autoencoder_best.pt",
        help="Path to the trained model checkpoint.",
    )
    return parser.parse_args()


def load_test_data(config: dict) -> tuple[DataLoader, np.ndarray]:
    """Load the test dataset and return a DataLoader and labels.

    Attempts to load preprocessed tensors first. Falls back to raw CSV
    if they are not available.

    Args:
        config: Full configuration dictionary.

    Returns:
        Tuple of (test_loader, test_labels).
    """
    import pandas as pd

    batch_size = config["training"]["batch_size"]
    data_dir = config["paths"]["data_dir"]

    # ── Try preprocessed tensors ────────────────────────────────────
    preprocessed_path = os.path.join(data_dir, "preprocessed")
    if os.path.isdir(preprocessed_path):
        logger.info("Loading preprocessed test data from %s", preprocessed_path)
        X_test = torch.load(os.path.join(preprocessed_path, "X_test.pt"), weights_only=True)
        y_test = torch.load(os.path.join(preprocessed_path, "y_test.pt"), weights_only=True)
    else:
        # ── Fallback: raw CSV ───────────────────────────────────────
        logger.warning(
            "Preprocessed data not found. Falling back to raw CSV."
        )
        csv_path = os.path.join(data_dir, "creditcard.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        features = df.drop(columns=["Class"]).values.astype(np.float32)
        labels = df["Class"].values.astype(np.float32)

        # Standard scaling
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features = (features - mean) / std

        # Use last 20% as test set (consistent with train script fallback)
        from sklearn.model_selection import train_test_split

        _, X_test_np, _, y_test_np = train_test_split(
            features, labels, test_size=0.2, random_state=config.get("seed", 42),
            stratify=labels,
        )

        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        y_test = torch.tensor(y_test_np, dtype=torch.float32)

    logger.info("Test data: %d samples.", X_test.shape[0])

    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_labels = y_test.numpy()

    return test_loader, test_labels


def main() -> None:
    """Main prediction / evaluation entrypoint."""
    args = parse_args()

    # ── Configuration ───────────────────────────────────────────────
    config = load_config(args.config)
    setup_logging(config["paths"].get("log_dir"))
    set_seed(config.get("seed", 42))
    device = get_device(config)

    # ── Load Model ──────────────────────────────────────────────────
    model = FFNNAutoencoder.from_config(config["model"])

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train the model first."
        )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    logger.info(
        "Model loaded from %s (trained for %d epochs, val_loss=%.6f).",
        args.checkpoint,
        checkpoint.get("epoch", -1),
        checkpoint.get("val_loss", -1),
    )
    logger.info("\n%s", model.summary())

    # ── Test Data ───────────────────────────────────────────────────
    test_loader, test_labels = load_test_data(config)

    # ── Evaluation ──────────────────────────────────────────────────
    evaluator = Evaluator(model, config, device)

    # Compute anomaly scores
    scores = evaluator.compute_anomaly_scores(test_loader)

    # Find threshold
    threshold = evaluator.find_optimal_threshold(scores, test_labels)

    # Full evaluation report
    metrics = evaluator.evaluate(test_loader, test_labels, threshold=threshold)

    # Diagnostic plots
    plots_dir = "plots/"
    evaluator.plot_results(scores, test_labels, threshold, save_dir=plots_dir)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FFNN AUTOENCODER — EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Threshold : {metrics['threshold']:.6f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  AUPRC     : {metrics['auprc']:.4f}")
    print(f"  AUROC     : {metrics['auroc']:.4f}")
    print("=" * 60)
    print(f"\n  Diagnostic plots saved to: {plots_dir}")
    print(f"  - {plots_dir}ffnn_error_distribution.png")
    print(f"  - {plots_dir}ffnn_precision_recall_curve.png")
    print(f"  - {plots_dir}ffnn_confusion_matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
