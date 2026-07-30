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
import logging
import os

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, setup_logging, set_seed, get_device, count_parameters
from src.datasets.preprocess import Preprocessing
from src.models.classifier import FraudDetectionMLP
from src.train.trainer import Trainer
from src.evaluation.classification_evaluator import ClassificationEvaluator
from src.evaluation.plots import plot_training_history

logger = logging.getLogger(__name__)


def train_classifier(config_path: str = "configs/classification_config.yaml", eval_only: bool = False) -> None:
    """Full classifier pipeline: preprocess → train → evaluate.

    Args:
        config_path: Path to the YAML configuration file.
        eval_only: If True, skip training and load existing checkpoint for evaluation.
    """
    config = load_config(config_path)

    # Load encoder architecture (input_dim, hidden_dims) from the base model
    # config (config.yaml) so FraudDetectionMLP is built with the exact same
    # encoder shape as the contrastive pre-training — required for correct
    # weight loading.
    base_cfg_path = config.get("base_model_config", "configs/config.yaml")
    with open(base_cfg_path, "r") as _f:
        _base_cfg = yaml.safe_load(_f)
    config["model"]["input_dim"]   = _base_cfg["model"]["input_dim"]
    config["model"]["hidden_dims"] = _base_cfg["model"]["hidden_dims"]

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
    freeze_encoder = config.get("freeze_encoder", True)

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
    model = FraudDetectionMLP(config=config).to(device)
    logger.info("FraudDetectionMLP — %s trainable parameters", f"{count_parameters(model):,}")

    # val_labels are needed by Trainer to compute F1/AUPRC on the validation set
    val_labels = y_val.squeeze().cpu().numpy().astype(int)

    # Delegate the full training lifecycle to Trainer
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
            logger.info("Encoder frozen — training classifier head only.")

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
        logger.info("Running in EVALUATION ONLY mode (skipping training).")

    # Load Best Checkpoint
    checkpoint_name = config["paths"].get("checkpoint_name", "Classifier_best.pt")
    best_ckpt_path = os.path.join(trainer.checkpoint_dir, checkpoint_name)
    if os.path.isfile(best_ckpt_path):
        trainer.load_checkpoint(best_ckpt_path)
        logger.info("Loaded best checkpoint: %s", best_ckpt_path)
    else:
        if eval_only:
            raise FileNotFoundError(f"Checkpoint not found at {best_ckpt_path}. Run training first.")

    # Determine Optimal Threshold on Validation Set
    evaluator = ClassificationEvaluator(model=model, config=config, device=device)
    val_probs = evaluator.compute_probabilities(val_loader)
    val_threshold = evaluator._find_optimal_threshold(val_probs, val_labels)
    logger.info("Optimal threshold determined on Validation set: %.6f", val_threshold)

    # Evaluation on Test Set (using Validation Threshold)
    test_labels = y_test.squeeze().cpu().numpy().astype(int)
    metrics = evaluator.evaluate(test_loader, test_labels, threshold=val_threshold)

    logger.info("=" * 55)
    logger.info("  FINAL CLASSIFIER TEST RESULTS (Threshold from Validation Set)")
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
