"""
Train, evaluate, and test the FraudAutoencoder for anomaly-based fraud detection.

Pipeline:
    1. Load configuration (configs/config.yaml)
    2. Preprocess data (stratified split, RobustScaler, normal-only training set)
    3. (Optional) Load contrastive pre-trained encoder weights
    4. Train with Trainer (early stopping, checkpointing on val AUPRC)
    5. Evaluate with ReconstructionEvaluator (full metrics + 6 diagnostic plots)
    6. Plot training history
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, setup_logging, set_seed, get_device, count_parameters
from src.datasets.preprocess import Preprocessing
from src.models.autoencoder import FraudAutoencoder
from src.train.trainer import Trainer
from src.evaluation.reconstruction_evaluator import ReconstructionEvaluator
from src.evaluation.plots import plot_training_history

logger = logging.getLogger(__name__)


def train_and_evaluate(config_path: str = "configs/config.yaml", eval_only: bool = False) -> None:
    """
    Full autoencoder pipeline: preprocess -> train -> evaluate.

    Args:
        config_path: Path to the YAML configuration file.
        eval_only: If True, skip training and load existing checkpoint for evaluation.
    """

    # Configuration and Setup
    config = load_config(config_path)
    setup_logging(log_dir=config["paths"].get("log_dir"))
    set_seed(config.get("seed", 42))
    device = get_device(config)
    config["device"] = device  # Inject resolved device into config for Trainer

    # Training parameters
    batch_size = config.get("batch_size", 256)
    test_size = config.get("test_size", 0.2)
    val_size = config.get("val_size", 0.15)
    seed = config.get("seed", 42)
    drop_time = config.get("drop_time", True)
    freeze_encoder = config.get("freeze_encoder", True)

    # Preprocessing 
    data_dir = config["paths"].get("data_dir", "data/")
    data_path = os.path.join(data_dir, "creditcard.csv")
    df = pd.read_csv(data_path)
    preprocessor = Preprocessing(df, drop_time=drop_time)

    # Split data into train, validation, and test sets
    X_train_tensor, X_val_tensor, X_test_tensor, y_val_tensor, y_test_tensor = (
        preprocessor.get_dataset(
            test_size=test_size,
            val_size=val_size,
            random_state=seed,
            autoencoder=True,
        )
    )

    # Train: plain tensor (unsupervised, normal-only so the target is the input itself)
    train_loader = DataLoader(
        X_train_tensor, batch_size=batch_size, shuffle=True,
    )
    # Val: TensorDataset with labels (needed for AUPRC/F1 computation during training)
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False,
    )
    # Test: TensorDataset with labels (for final evaluation)
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False,
    )

    # Model Initialization
    model = FraudAutoencoder(config=config).to(device)
    logger.info("FraudAutoencoder — %s trainable parameters", f"{count_parameters(model):,}")

    val_labels = y_val_tensor.cpu().numpy().astype(int)
    trainer = Trainer(model=model, config=config)

    if not eval_only:
        # Optionally load contrastive pre-trained encoder weights
        pretrained_path = config.get("contrastive", {}).get(
            "backbone_save_path", "pretrained_tabular_encoder.pth"
        )
        if os.path.isfile(pretrained_path):
            model.encoder.load_state_dict(
                torch.load(pretrained_path, map_location=device, weights_only=True)
            )
            logger.info("Loaded pre-trained encoder weights from: %s", pretrained_path)
        else:
            logger.info("No pre-trained encoder found — training from scratch.")

        # Optionally freeze encoder (train decoder only)
        if freeze_encoder and os.path.isfile(pretrained_path):
            for param in model.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen — training decoder only.")

        # Training and training history plotting
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            val_labels=val_labels,
        )

        log_dir = config["paths"].get("log_dir", "logs/")
        plot_training_history(
            history,
            save_dir=log_dir,
            filename="autoencoder_training_history.png",
            title="Autoencoder Training",
        )
    else:
        logger.info("Running in EVALUATION ONLY mode (skipping training).")

    # Load best checkpoint
    checkpoint_name = config["paths"].get("checkpoint_name", "autoencoder_best.pt")
    best_ckpt_path = os.path.join(trainer.checkpoint_dir, checkpoint_name)
    if os.path.isfile(best_ckpt_path):
        trainer.load_checkpoint(best_ckpt_path)
        logger.info("Loaded best checkpoint: %s", best_ckpt_path)
    else:
        if eval_only:
            raise FileNotFoundError(f"Checkpoint not found at {best_ckpt_path}. Run training first.")

    # Determine Optimal Threshold on Validation Set to perform the evaluation on the test set
    evaluator = ReconstructionEvaluator(model=model, config=config, device=device)
    val_scores = evaluator.compute_anomaly_scores(val_loader)
    val_threshold = evaluator.find_optimal_threshold(val_scores, val_labels)
    logger.info("Optimal threshold determined on Validation set: %.6f", val_threshold)

    # Evaluation on Test Set (using Validation Threshold)
    test_labels = y_test_tensor.cpu().numpy().astype(int)
    metrics = evaluator.evaluate(test_loader, test_labels, threshold=val_threshold)

    logger.info("=" * 55)
    logger.info("  FINAL TEST RESULTS (Threshold from Validation Set)")
    logger.info("=" * 55)
    logger.info("  Threshold : %.6f", val_threshold)
    logger.info("  F1        : %.4f", metrics['f1'])
    logger.info("  F2        : %.4f", metrics['f2'])
    logger.info("  MCC       : %.4f", metrics['mcc'])
    logger.info("  AUPRC     : %.4f", metrics['auprc'])
    logger.info("  AUROC     : %.4f", metrics['auroc'])
    logger.info("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the FraudAutoencoder for anomaly detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and evaluate using the saved checkpoint immediately",
    )
    args = parser.parse_args()
    train_and_evaluate(config_path=args.config, eval_only=args.eval_only)


if __name__ == "__main__":
    main()