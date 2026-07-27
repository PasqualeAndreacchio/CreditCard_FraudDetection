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
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, setup_logging, set_seed, get_device, count_parameters
from src.Datasets.preprocess import Preprocessing
from src.models.Autoencoder import FraudAutoencoder
from src.Train.trainer import Trainer
from src.Evaluation.reconstruction_evaluator import ReconstructionEvaluator
from src.Evaluation.plots import plot_training_history


def train_and_evaluate(config_path: str = "configs/config.yaml", eval_only: bool = False) -> None:
    """Full autoencoder pipeline: preprocess → train → evaluate.

    Args:
        config_path: Path to the YAML configuration file.
        eval_only: If True, skip training and load existing checkpoint for evaluation.
    """
    # ── 1. Configuration & Setup ─────────────────────────────────────────
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

    # ── 2. Data Preparation ──────────────────────────────────────────────
    data_dir = config["paths"].get("data_dir", "data/")
    data_path = os.path.join(data_dir, "creditcard.csv")
    df = pd.read_csv(data_path)
    preprocessor = Preprocessing(df, drop_time=drop_time)

    # get_dataset(autoencoder=True) with val_size returns:
    #   X_train (normal only), X_val (all), X_test (all), y_val, y_test
    X_train_tensor, X_val_tensor, X_test_tensor, y_val_tensor, y_test_tensor = (
        preprocessor.get_dataset(
            test_size=test_size,
            val_size=val_size,
            random_state=seed,
            autoencoder=True,
        )
    )

    # Train: plain tensor (unsupervised, normal-only → target is input itself)
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

    # ── 3. Model Initialization ──────────────────────────────────────────
    model = FraudAutoencoder(config=config).to(device)
    print(f"FraudAutoencoder — {count_parameters(model):,} trainable parameters")

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
            print(f"Loaded pre-trained encoder weights from: {pretrained_path}")
        else:
            print("No pre-trained encoder found — training from scratch.")

        # Optionally freeze encoder (train decoder only)
        if freeze_encoder and os.path.isfile(pretrained_path):
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen — training decoder only.")

        # ── 4. Training ──────────────────────────────────────────────────
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            val_labels=val_labels,
        )

        # Plot training history
        log_dir = config["paths"].get("log_dir", "logs/")
        plot_training_history(
            history,
            save_dir=log_dir,
            filename="autoencoder_training_history.png",
            title="Autoencoder Training",
        )
    else:
        print("\n[*] Running in EVALUATION ONLY mode (skipping training)...")

    # ── 5. Load Best Checkpoint ──────────────────────────────────────────
    checkpoint_name = config["paths"].get("checkpoint_name", "autoencoder_best.pt")
    best_ckpt_path = os.path.join(trainer.checkpoint_dir, checkpoint_name)
    if os.path.isfile(best_ckpt_path):
        trainer.load_checkpoint(best_ckpt_path)
        print(f"Loaded best checkpoint: {best_ckpt_path}")
    else:
        if eval_only:
            raise FileNotFoundError(f"Checkpoint not found at {best_ckpt_path}. Run training first.")

    # ── 6. Determine Optimal Threshold on Validation Set ───────────────
    evaluator = ReconstructionEvaluator(model=model, config=config, device=device)
    val_scores = evaluator.compute_anomaly_scores(val_loader)
    val_threshold = evaluator.find_optimal_threshold(val_scores, val_labels)
    print(f"Optimal threshold determined on Validation set: {val_threshold:.6f}")

    # ── 7. Evaluation on Test Set (using Validation Threshold) ──────────
    test_labels = y_test_tensor.cpu().numpy().astype(int)
    metrics = evaluator.evaluate(test_loader, test_labels, threshold=val_threshold)

    print("\n" + "=" * 55)
    print("  FINAL TEST RESULTS (Threshold from Validation Set)")
    print("=" * 55)
    print(f"  Threshold: {val_threshold:.6f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  F2:        {metrics['f2']:.4f}")
    print(f"  MCC:       {metrics['mcc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print("=" * 55)


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