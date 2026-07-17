"""
Prediction / Evaluation entry-point for the Complete_Autoencoder classifier.

Usage::

    python classification_evaluator.py --config configs/classification_config.yaml \\
                                       --checkpoint checkpoints/classifier_best.pt

This script:
    1. Loads the YAML configuration and model checkpoint
    2. Loads the test dataset (preprocessed tensors or raw CSV)
    3. Computes per-sample fraud probabilities via softmax
    4. Determines the optimal decision threshold
    5. Generates binary predictions and a full evaluation report
    6. Saves diagnostic plots
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.Model import Complete_Autoencoder
from src.Evaluation.classification_evaluator import ClassificationEvaluator
from src.utils import set_seed, get_device, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained Complete_Autoencoder classifier on the test set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/classification_config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/classifier_best.pt",
        help="Path to the trained model checkpoint.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_data(config: dict) -> tuple[DataLoader, np.ndarray]:
    """Load the test dataset and return a DataLoader and labels.

    Tries preprocessed tensors first, falls back to raw CSV.

    Args:
        config: Full configuration dictionary.

    Returns:
        Tuple of (test_loader, test_labels).
    """
    batch_size = config["training"]["batch_size"]
    data_dir = config["paths"]["data_dir"]

    preprocessed_path = os.path.join(data_dir, "preprocessed")
    if os.path.isdir(preprocessed_path):
        logger.info("Loading preprocessed test data from %s", preprocessed_path)
        X_test = torch.load(os.path.join(preprocessed_path, "X_test.pt"), weights_only=True)
        y_test = torch.load(os.path.join(preprocessed_path, "y_test.pt"), weights_only=True)
    else:
        logger.warning("Preprocessed data not found. Falling back to raw CSV.")
        csv_path = os.path.join(data_dir, "creditcard.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        features = df.drop(columns=["Class"]).values.astype(np.float32)
        labels = df["Class"].values.astype(np.float32)

        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features = (features - mean) / std

        from sklearn.model_selection import train_test_split
        _, X_test_np, _, y_test_np = train_test_split(
            features, labels,
            test_size=config.get("test_size", 0.2),
            random_state=config.get("seed", 42),
            stratify=labels,
        )

        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        y_test = torch.tensor(y_test_np, dtype=torch.float32)

    logger.info("Test data: %d samples.", X_test.shape[0])

    test_loader = DataLoader(TensorDataset(X_test), batch_size=batch_size, shuffle=False)
    return test_loader, y_test.numpy()


def main() -> None:
    """Main prediction / evaluation entrypoint."""
    args = parse_args()

    config = load_config(args.config)
    setup_logging(config["paths"].get("log_dir"))
    set_seed(config.get("seed", 42))
    device = get_device(config)

    # Model
    model = Complete_Autoencoder(config=config)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train the model first."
        )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    logger.info(
        "Model loaded from %s (trained for %d epochs).",
        args.checkpoint,
        checkpoint.get("epoch", -1),
    )

    # Data
    test_loader, test_labels = load_test_data(config)

    # Evaluation
    evaluator = ClassificationEvaluator(model, config, device)

    probs = evaluator.compute_probabilities(test_loader)
    threshold = evaluator._find_optimal_threshold(probs[:, 1], test_labels)
    evaluator.evaluate(test_loader, test_labels, threshold=threshold)


if __name__ == "__main__":
    main()
