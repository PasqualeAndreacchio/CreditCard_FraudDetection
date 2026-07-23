import argparse
import json
import logging
import os
import sys
import copy
import pandas as pd
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader

import optuna
from optuna.samplers import TPESampler

from src.models.LSTM_Autoencoder import LSTM_Autoencoder
from src.Train.trainer import Trainer
from src.Datasets.preprocess import Preprocessing
from src.Evaluation.reconstruction_evaluator import ReconstructionEvaluator

logger = logging.getLogger(__name__)


def objective(
    trial: optuna.Trial,
    preprocess: Preprocessing,
    base_config: dict,
    num_epochs: int = 30
) -> float:
    """Optuna objective function to evaluate a single hyperparameter configuration."""

    # 1. Suggest Hyperparameters
    loss_type = trial.suggest_categorical("loss", ["mse", "mae", "huber"])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 12, 16])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    seq_len = trial.suggest_categorical("seq_len", [3, 5, 7, 10])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])

    # 2. Build Trial Configuration
    config = copy.deepcopy(base_config)
    config["lstm_model"]["hidden_dim"] = hidden_dim
    config["lstm_model"]["latent_dim"] = latent_dim
    config["lstm_model"]["num_layers"] = num_layers
    config["lstm_model"]["dropout"] = dropout

    config["sequence"]["seq_len"] = seq_len

    config["batch_size"] = batch_size
    config["training"]["epochs"] = num_epochs
    config["training"]["learning_rate"] = learning_rate
    config["training"]["weight_decay"] = weight_decay
    config["training"]["optimizer"] = optimizer_name
    config["training"]["loss"] = loss_type
    config["training"]["val_metric"] = "auprc"
    config["training"]["early_stopping"]["patience"] = 5

    # Force a validation split for hyperparameter tuning
    val_size = config.get("val_size", 0.15)
    if val_size == 0:
        val_size = 0.15
        config["val_size"] = val_size
    test_size = config.get("test_size", 0.2)
    only_normal = config.get("sequence", {}).get("only_normal", True)
    stride = config.get("sequence", {}).get("stride", 1)

    # 3. Data Preparation
    X_train_seq, X_val_seq, X_test_seq, y_val_seq, y_test_seq = preprocess.get_sequence_dataset(
        seq_len=seq_len,
        stride=stride,
        test_size=test_size,
        val_size=val_size,
        autoencoder=True,
        only_normal=only_normal
    )

    train_dataset = TensorDataset(X_train_seq)
    val_dataset = TensorDataset(X_val_seq, y_val_seq)

    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. Model Instantiation
    input_dim = X_train_seq.shape[2]
    model = LSTM_Autoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    # 5. Training with AUPRC checkpointing
    labels_val = (y_val_seq.cpu().numpy() > 0).astype(int)
    trainer = Trainer(model=model, config=config)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        val_labels=labels_val
    )

    # 6. Validation Evaluation on Best Checkpoint
    checkpoint_name = config.get("paths", {}).get("checkpoint_name", "lstm_autoencoder_best.pt")
    best_ckpt_path = os.path.join(trainer.checkpoint_dir, checkpoint_name)
    if os.path.exists(best_ckpt_path):
        trainer.load_checkpoint(best_ckpt_path)

    device = torch.device(config.get("device", "cpu"))
    evaluator = ReconstructionEvaluator(model=model, config=config, device=device)

    scores_val = evaluator.compute_anomaly_scores(val_loader)

    from sklearn.metrics import average_precision_score
    val_auprc = float(average_precision_score(labels_val, scores_val, pos_label=1))

    return val_auprc


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for LSTM Autoencoder (AUPRC Metric)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_LSTM_auprc.yaml",
        help="Path to base configuration YAML file"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=15,
        help="Number of Optuna trial runs (default: 15)"
    )
    parser.add_argument(
        "--epochs_per_trial",
        type=int,
        default=20,
        help="Maximum epochs per trial (default: 20)"
    )
    parser.add_argument(
        "--output_config",
        type=str,
        default="configs/config_LSTM_auprc_best.yaml",
        help="Path to save best configuration YAML"
    )
    args = parser.parse_args()

    # Load base configuration
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    # Logging Setup
    log_dir = base_config.get("paths", {}).get("log_dir", "logs/lstm_auprc")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "tune.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

    logger.info("=" * 60)
    logger.info("  OPTUNA HYPERPARAMETER TUNING — LSTM AUTOENCODER (AUPRC METRIC)")
    logger.info("=" * 60)
    logger.info(f"Logging initialized. Output saved to: {log_file_path}")

    # Load data once
    drop_time = base_config.get("sequence", {}).get("drop_time", True)
    logger.info(f"Loading raw dataset (drop_time={drop_time})...")
    rawdata = pd.read_csv("data/creditcard.csv")
    preprocess = Preprocessing(rawdata, drop_time=drop_time)

    # Create Optuna study
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = TPESampler(seed=base_config.get("seed", 42))
    study = optuna.create_study(
        study_name="lstm_autoencoder_tuning_auprc",
        direction="maximize",
        sampler=sampler
    )

    logger.info(f"Starting {args.n_trials} trials (max {args.epochs_per_trial} epochs each)...")
    study.optimize(
        lambda trial: objective(trial, preprocess, base_config, num_epochs=args.epochs_per_trial),
        n_trials=args.n_trials
    )

    logger.info("=" * 60)
    logger.info("  TUNING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Best Trial AUPRC Score: {study.best_value:.4f}")
    logger.info("Best Hyperparameters:")
    for k, v in study.best_params.items():
        logger.info(f"  - {k}: {v}")

    # Build and save updated best config
    best_config = copy.deepcopy(base_config)
    bp = study.best_params
    best_config["lstm_model"]["hidden_dim"] = bp["hidden_dim"]
    best_config["lstm_model"]["latent_dim"] = bp["latent_dim"]
    best_config["lstm_model"]["num_layers"] = bp["num_layers"]
    best_config["lstm_model"]["dropout"] = bp["dropout"]
    best_config["sequence"]["seq_len"] = bp["seq_len"]
    best_config["batch_size"] = bp["batch_size"]
    best_config["training"]["learning_rate"] = bp["learning_rate"]
    best_config["training"]["weight_decay"] = bp["weight_decay"]
    best_config["training"]["optimizer"] = bp["optimizer"]
    best_config["training"]["loss"] = bp["loss"]

    with open(args.output_config, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Best configuration saved to: {args.output_config}")

    # Also update config_LSTM.yaml
    with open(args.config, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Updated base configuration file: {args.config}")


if __name__ == "__main__":
    main()
