"""
Train and evaluate the FraudDetectionMLP classifier for supervised fraud detection.

Pipeline:
    1. Load configuration (configs/classification_config.yaml)
    2. Preprocess data (SMOTE or class weights, stratified split)
    3. Train with Trainer (early stopping, checkpointing on val F1)
    4. Plot training history
    5. Load best checkpoint
    6. Evaluate on Test set with ClassificationEvaluator (all metrics + 6 plots)
"""

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, setup_logging, set_seed, get_device, count_parameters
from src.Datasets.preprocess import Preprocessing
from src.models.Classifier import FraudDetectionMLP
from src.Train.trainer import Trainer
from src.Evaluation.classification_evaluator import ClassificationEvaluator
from src.Evaluation.plots import plot_training_history


def train_classifier(config_path: str = "configs/classification_config.yaml", eval_only: bool = False) -> None:
    """Full classifier pipeline: preprocess → train → evaluate.

    Args:
        config_path: Path to the YAML configuration file.
        eval_only: If True, skip training and load existing checkpoint for evaluation.
    """
    config = load_config(config_path)
    setup_logging(log_dir=config["paths"].get("log_dir"))
    set_seed(config.get("seed", 42))
    device = get_device(config)
    config["device"] = device

    seed       = config.get("seed", 42)
    test_size  = config.get("test_size", 0.2)
    val_size   = config.get("val_size", 0.1)
    batch_size = config["training"].get("batch_size", 512)
    drop_time  = config.get("drop_time", True)
    use_smote  = config.get("use_smote", True)

    # Build the dataset via Preprocessing (scaling + stratified split)
    data = pd.read_csv("data/creditcard.csv")
    preprocessor = Preprocessing(data, drop_time=drop_time)

    if use_smote:
        # SMOTE is applied to the training set only; val and test are left untouched
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_smote_dataset(
            test_size=test_size, val_size=val_size, random_state=seed
        )
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_dataset(
            test_size=test_size, val_size=val_size, random_state=seed, autoencoder=False
        )

    # Wrap tensors in DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    # Build the model
    input_dim = X_train.shape[1]
    model = FraudDetectionMLP(input_dim=input_dim).to(device)
    print(f"FraudDetectionMLP — {count_parameters(model):,} trainable parameters")

    # val_labels are needed by Trainer to compute F1/AUPRC on the validation set
    val_labels = y_val.squeeze().cpu().numpy().astype(int)

    # Delegate the full training lifecycle to Trainer
    trainer = Trainer(model=model, config=config)

    if not eval_only:
        # Delegate the full training lifecycle to Trainer
        history = trainer.fit(train_loader=train_loader, val_loader=val_loader, val_labels=val_labels)

        # Plot training history
        log_dir = config["paths"].get("log_dir", "logs/")
        plot_training_history(
            history,
            save_dir=log_dir,
            filename="classifier_training_history.png",
            title="Classifier Training",
        )
    else:
        print("\n[*] Running in EVALUATION ONLY mode (skipping training)...")

    # Load Best Checkpoint
    checkpoint_name = config["paths"].get("checkpoint_name", "Classifier_best.pt")
    best_ckpt_path = os.path.join(trainer.checkpoint_dir, checkpoint_name)
    if os.path.isfile(best_ckpt_path):
        trainer.load_checkpoint(best_ckpt_path)
        print(f"Loaded best checkpoint: {best_ckpt_path}")
    else:
        if eval_only:
            raise FileNotFoundError(f"Checkpoint not found at {best_ckpt_path}. Run training first.")

    # Determine Optimal Threshold on Validation Set
    evaluator = ClassificationEvaluator(model=model, config=config, device=device)
    val_probs = evaluator.compute_probabilities(val_loader)
    val_threshold = evaluator._find_optimal_threshold(val_probs, val_labels)
    print(f"Optimal threshold determined on Validation set: {val_threshold:.6f}")

    # Evaluation on Test Set (using Validation Threshold)
    test_labels = y_test.squeeze().cpu().numpy().astype(int)
    metrics = evaluator.evaluate(test_loader, test_labels, threshold=val_threshold)

    print("\n" + "=" * 55)
    print("  FINAL CLASSIFIER TEST RESULTS (Threshold from Validation Set)")
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
        description="Train and evaluate the FraudDetectionMLP classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/classification_config.yaml",
        help="Path to the YAML configuration file (default: configs/classification_config.yaml)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and evaluate using the saved checkpoint immediately",
    )
    args = parser.parse_args()
    train_classifier(config_path=args.config, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
