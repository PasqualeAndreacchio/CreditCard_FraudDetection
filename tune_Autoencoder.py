import argparse
import copy
import logging
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score

import optuna
from optuna.samplers import TPESampler

from src.models.Autoencoder import ContrastiveModel, FraudAutoencoder
from src.Train.trainer import Trainer
from src.Datasets.preprocess import Preprocessing
from src.Evaluation.reconstruction_evaluator import ReconstructionEvaluator

logger = logging.getLogger(__name__)


# Contrastive Pre-training

def pretrain_encoder(
    input_dim: int,
    hidden_dims: list,
    preprocess_encoder: Preprocessing,
    device: torch.device,
    batch_size: int = 256,
    epochs: int = 20,
    learning_rate: float = 1e-3,
) -> dict:
    """Pre-train the encoder backbone using contrastive (triplet) learning.

    Follows the same approach as Contrastive_training.py:
    - Builds a ContrastiveModel (Encoder backbone + projection head).
    - Trains with TripletMarginLoss on anchor/positive/negative triplets.
    - Returns the backbone's state_dict (projection head is discarded).

    Args:
        input_dim: Number of input features.
        hidden_dims: Encoder layer widths (same as for the autoencoder).
        preprocess_encoder: Preprocessing object wrapping the encoder half
            of the data, used to build the ContrastiveDataset.
        device: Torch device.
        batch_size: Mini-batch size for contrastive training.
        epochs: Number of contrastive training epochs.
        learning_rate: Learning rate for contrastive training.

    Returns:
        The backbone encoder's state_dict.
    """
    dataset = preprocess_encoder.get_contrastive_dataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ContrastiveModel(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            proj_anchor = model(anchor)
            proj_pos = model(positive)
            proj_neg = model(negative)

            loss = criterion(proj_anchor, proj_pos, proj_neg)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(
            "  [Contrastive] Epoch %2d/%d  |  triplet_loss=%.8f",
            epoch, epochs, avg_loss,
        )

    return model.backbone.state_dict()


# Objective Function

def objective(
    trial: optuna.Trial,
    preprocess_encoder: Preprocessing,
    preprocess_decoder: Preprocessing,
    base_config: dict,
    contrastive_epochs: int = 20,
    num_epochs: int = 30,
) -> float:
    """Optuna objective function to evaluate a single hyperparameter configuration.

    Each trial:
    1. Suggests architecture and training hyperparameters.
    2. Pre-trains the encoder with contrastive learning (triplet loss).
    3. Builds a FraudAutoencoder and loads the pre-trained encoder weights.
    4. Trains the full autoencoder on reconstruction (decoder half of data).
    5. Returns the AUPRC on the validation set.

    Args:
        trial: The Optuna trial object for hyperparameter suggestion.
        preprocess_encoder: Pre-initialised Preprocessing object for the
            encoder half of the data (contrastive pre-training).
        preprocess_decoder: Pre-initialised Preprocessing object for the
            decoder half of the data (reconstruction training).
        base_config: Base configuration dictionary to deep-copy and override.
        contrastive_epochs: Number of epochs for contrastive pre-training.
        num_epochs: Maximum number of reconstruction training epochs per trial.

    Returns:
        Validation AUPRC score (higher is better).
    """

    # Suggest Hyperparameters for the architecture
    num_layers = trial.suggest_int("num_layers", 2, 4)
    hidden_dim_1 = trial.suggest_categorical("hidden_dim_1", [20, 24, 28])
    hidden_dim_2 = trial.suggest_categorical("hidden_dim_2", [12, 14, 16, 18])

    hidden_dims = [hidden_dim_1, hidden_dim_2]

    if num_layers >= 3:
        hidden_dim_3 = trial.suggest_categorical("hidden_dim_3", [6, 8, 10, 12])
        # Enforce monotonic decrease (compression path)
        if hidden_dim_3 >= hidden_dim_2:
            raise optuna.TrialPruned(
                f"Monotonic decrease violated: dim3={hidden_dim_3} >= dim2={hidden_dim_2}"
            )
        hidden_dims.append(hidden_dim_3)

    if num_layers >= 4:
        hidden_dim_4 = trial.suggest_categorical("hidden_dim_4", [3, 4, 6])
        # Enforce monotonic decrease (compression path)
        if hidden_dim_4 >= hidden_dims[-1]:
            raise optuna.TrialPruned(
                f"Monotonic decrease violated: dim4={hidden_dim_4} >= dim3={hidden_dims[-1]}"
            )
        hidden_dims.append(hidden_dim_4)

    # Suggest Hyperparameters for the training
    loss_type = trial.suggest_categorical("loss", ["mse", "mae", "huber"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])

    # Build Trial Configuration
    config = copy.deepcopy(base_config)
    config["model"]["hidden_dims"] = hidden_dims

    config["batch_size"] = batch_size
    config["training"]["epochs"] = num_epochs
    config["training"]["learning_rate"] = learning_rate
    config["training"]["weight_decay"] = weight_decay
    config["training"]["optimizer"] = optimizer_name
    config["training"]["loss"] = loss_type
    config["training"]["val_metric"] = "auprc"
    config["training"]["early_stopping"]["patience"] = 5
    config["paths"]["checkpoint_name"] = "autoencoder_tune_best.pt"

    # Force a validation split for hyperparameter tuning
    val_size = config.get("val_size", 0.15)
    if val_size == 0:
        val_size = 0.15
        config["val_size"] = val_size
    test_size = config.get("test_size", 0.2)

    device = torch.device(config.get("device", "cpu"))
    input_dim = config["model"]["input_dim"]

    # Contrastive Pre-training (Encoder)
    logger.info(
        "  Trial %d — Contrastive pre-training (hidden_dims=%s)...",
        trial.number, hidden_dims,
    )
    encoder_state_dict = pretrain_encoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        preprocess_encoder=preprocess_encoder,
        device=device,
        batch_size=batch_size,
        epochs=contrastive_epochs,
        learning_rate=1e-3,
    )

    # Data Preparation (Decoder half)
    # The decoder Preprocessing object uses only the reconstruction half
    # of the data (decoder_df), following the same split as
    # Contrastive_training.py.
    X_train_tensor, X_val_tensor, X_test_tensor, y_val_tensor, y_test_tensor = (
        preprocess_decoder.get_dataset(
            test_size=test_size,
            val_size=val_size,
            autoencoder=True,
        )
    )

    # Train set is a plain tensor (normal-only, unsupervised reconstruction).
    train_dataset = X_train_tensor
    # Val set needs labels for AUPRC evaluation.
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

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

    # Model Instantiation with pre-trained encoder
    model = FraudAutoencoder(config=config)

    # Load the contrastive pre-trained weights into the encoder.
    # The decoder starts from scratch.
    model.encoder.load_state_dict(encoder_state_dict)
    logger.info("  Loaded contrastive pre-trained encoder weights.")

    # Training with AUPRC Checkpointing
    labels_val = (y_val_tensor.cpu().numpy() > 0).astype(int)
    trainer = Trainer(model=model, config=config)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        val_labels=labels_val,
    )

    # Validation Evaluation on Best Checkpoint
    checkpoint_name = config["paths"].get("checkpoint_name", "autoencoder_tune_best.pt")
    best_ckpt_path = os.path.join(trainer.checkpoint_dir, checkpoint_name)
    if os.path.exists(best_ckpt_path):
        trainer.load_checkpoint(best_ckpt_path)

    evaluator = ReconstructionEvaluator(model=model, config=config, device=device)
    scores_val = evaluator.compute_anomaly_scores(val_loader)

    val_auprc = float(average_precision_score(labels_val, scores_val, pos_label=1))

    return val_auprc


# Data Splitting
def prepare_data_split(data_path: str = "data/creditcard.csv", random_state: int = 42):
    """Split the raw dataset 50/50 into encoder (contrastive) and decoder
    (reconstruction) halves to avoid data leakage between pre-training
    and reconstruction training.

    Each half is wrapped in a Preprocessing object so that downstream
    code can call get_contrastive_dataset() or get_dataset() directly.

    Args:
        data_path: Path to the raw credit card CSV.
        random_state: Random seed for reproducible splitting.

    Returns:
        A tuple of (preprocess_encoder, preprocess_decoder).
    """
    originaldata = pd.read_csv(data_path)

    normal_mask = originaldata["Class"] == 0
    fraud_mask = originaldata["Class"] == 1

    normaldata = originaldata[normal_mask]
    frauddata = originaldata[fraud_mask]

    # 50/50 stratified split within each class
    encoder_normal_df, decoder_normal_df = train_test_split(
        normaldata, test_size=0.5, random_state=random_state
    )
    encoder_fraud_df, decoder_fraud_df = train_test_split(
        frauddata, test_size=0.5, random_state=random_state
    )

    # Encoder half -> contrastive pre-training
    encoder_df = pd.concat([encoder_normal_df, encoder_fraud_df])
    logger.info("Encoder half: %d samples for contrastive pre-training", len(encoder_df))
    preprocess_encoder = Preprocessing(encoder_df, drop_time=True)

    # Decoder half -> reconstruction training
    decoder_df = pd.concat([decoder_normal_df, decoder_fraud_df])
    logger.info("Decoder half: %d samples for reconstruction", len(decoder_df))
    preprocess_decoder = Preprocessing(decoder_df, drop_time=True)

    return preprocess_encoder, preprocess_decoder


def main():
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Tuning for FFNN Autoencoder "
                    "(Contrastive Encoder + Reconstruction Decoder, AUPRC Metric)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to base configuration YAML file",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=15,
        help="Number of Optuna trial runs (default: 15)",
    )
    parser.add_argument(
        "--epochs_per_trial",
        type=int,
        default=30,
        help="Maximum reconstruction epochs per trial (default: 30)",
    )
    parser.add_argument(
        "--contrastive_epochs",
        type=int,
        default=20,
        help="Contrastive pre-training epochs per trial (default: 20)",
    )
    parser.add_argument(
        "--output_config",
        type=str,
        default="configs/config_best.yaml",
        help="Path to save best configuration YAML",
    )
    args = parser.parse_args()

    # Load Base Configuration
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    # Logging Setup
    log_dir = base_config.get("paths", {}).get("log_dir", "logs/")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "tune_autoencoder.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger.info("=" * 60)
    logger.info("  OPTUNA HYPERPARAMETER TUNING — FFNN AUTOENCODER")
    logger.info("  Contrastive Encoder + Reconstruction Decoder (AUPRC)")
    logger.info("=" * 60)
    logger.info(f"Logging initialized. Output saved to: {log_file_path}")

    # Data Split (once)
    # Split the raw data 50/50:
    #   - Encoder half -> contrastive pre-training
    #   - Decoder half -> reconstruction training
    logger.info("Splitting raw dataset 50/50 (encoder / decoder)...")
    preprocess_encoder, preprocess_decoder = prepare_data_split(
        data_path="data/creditcard.csv",
        random_state=base_config.get("seed", 42),
    )

    # Create Optuna Study
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = TPESampler(seed=base_config.get("seed", 42))
    study = optuna.create_study(
        study_name="ffnn_autoencoder_tuning_auprc",
        direction="maximize",
        sampler=sampler,
    )

    logger.info(
        f"Starting {args.n_trials} trials "
        f"({args.contrastive_epochs} contrastive + "
        f"max {args.epochs_per_trial} reconstruction epochs each)..."
    )
    study.optimize(
        lambda trial: objective(
            trial,
            preprocess_encoder,
            preprocess_decoder,
            base_config,
            contrastive_epochs=args.contrastive_epochs,
            num_epochs=args.epochs_per_trial,
        ),
        n_trials=args.n_trials,
    )

    # Report Results
    logger.info("=" * 60)
    logger.info("  TUNING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Best Trial AUPRC Score: {study.best_value:.4f}")
    logger.info("Best Hyperparameters:")
    for k, v in study.best_params.items():
        logger.info(f"  - {k}: {v}")

    # Build and Save Best Configuration
    best_config = copy.deepcopy(base_config)
    bp = study.best_params

    # Reconstruct hidden_dims list from individual layer suggestions
    best_hidden_dims = [bp["hidden_dim_1"], bp["hidden_dim_2"]]
    if bp["num_layers"] >= 3:
        best_hidden_dims.append(bp["hidden_dim_3"])
    if bp["num_layers"] >= 4:
        best_hidden_dims.append(bp["hidden_dim_4"])

    best_config["model"]["hidden_dims"] = best_hidden_dims
    best_config["batch_size"] = bp["batch_size"]
    best_config["training"]["learning_rate"] = bp["learning_rate"]
    best_config["training"]["weight_decay"] = bp["weight_decay"]
    best_config["training"]["optimizer"] = bp["optimizer"]
    best_config["training"]["loss"] = bp["loss"]

    with open(args.output_config, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Best configuration saved to: {args.output_config}")

    # Also update the base config file
    with open(args.config, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Updated base configuration file: {args.config}")


if __name__ == "__main__":
    main()
