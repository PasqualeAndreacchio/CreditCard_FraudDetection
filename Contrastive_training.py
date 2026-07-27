"""
Contrastive pre-training for the FraudAutoencoder backbone.

Pipeline:
    1. Load configuration (configs/config.yaml)
    2. Build a ContrastiveDataset via Preprocessing
    3. Train with ContrastiveTrainer (TripletMarginLoss)
    4. Save the backbone encoder weights (projection head discarded)
"""

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import load_config, setup_logging, set_seed, get_device
from src.models.Autoencoder import ContrastiveModel
from src.Datasets.preprocess import Preprocessing
from src.Train.contrastive_trainer import ContrastiveTrainer


def train_contrastive_model(config_path: str = "configs/config.yaml") -> None:
    """Contrastive pre-training pipeline.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)
    setup_logging(log_dir=config["paths"].get("log_dir"))
    set_seed(config.get("seed", 42))
    device = get_device(config)
    config["device"] = str(device)

    # Architecture parameters
    input_dim   = config["model"]["input_dim"]
    hidden_dims = config["model"]["hidden_dims"]
    batch_size  = config.get("batch_size", 256)

    # Read raw data and build the contrastive dataset via Preprocessing
    rawdata = pd.read_csv("data/creditcard.csv")
    preprocessed_data = Preprocessing(rawdata, drop_time=True)
    dataset = preprocessed_data.get_contrastive_dataset(
        test_size=config.get("test_size", 0.2),
        random_state=config.get("seed", 42),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialise the combined model (backbone + contrastive projection head)
    model = ContrastiveModel(input_dim=input_dim, hidden_dims=hidden_dims)

    # Delegate the entire training lifecycle to ContrastiveTrainer
    trainer = ContrastiveTrainer(model=model, config=config)
    trainer.fit(train_loader=loader)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contrastive pre-training for the FraudAutoencoder backbone"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file (default: configs/config.yaml)",
    )
    args = parser.parse_args()
    train_contrastive_model(config_path=args.config)


if __name__ == "__main__":
    main()