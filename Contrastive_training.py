import argparse
import torch
import pandas as pd
import yaml
from torch.utils.data import DataLoader

from src.models.Autoencoder import ContrastiveModel
from src.Datasets.preprocess import Preprocessing
from src.Train.contrastive_trainer import ContrastiveTrainer


# Training function exploiting ContrastiveTrainer
def train_contrastive_model(config_path: str = "configs/config.yaml") -> None:

    # Load configuration and get the device
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Architecture parameters
    input_dim   = config["model"]["input_dim"]
    hidden_dims = config["model"]["hidden_dims"]
    batch_size  = config.get("batch_size", 256)

    # Read raw data and build the contrastive dataset via Preprocessing.
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


# Main function to parse arguments and launch training
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