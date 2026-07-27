import argparse
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.models.Autoencoder import ContrastiveModel
from src.Datasets.preprocess import Preprocessing


# ─── TRAINING LOOP ───────────────────────────────────────────────────────

def train_contrastive_model(config_path: str = "configs/config.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the shared config so architecture is consistent with training_Autoencoder.py.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Hyperparameters
    batch_size = config.get("batch_size", 256)
    epochs = 20
    learning_rate = 1e-3

    # Architecture parameters — same list used by FraudAutoencoder.
    input_dim   = config["model"]["input_dim"]
    hidden_dims = config["model"]["hidden_dims"]

    # Read raw data and build the contrastive dataset via Preprocessing.
    # This applies RobustScaler (fitted on the training split only) and
    # produces (anchor, positive, negative) triplets without writing any
    # intermediate CSV files to disk.
    rawdata = pd.read_csv("data/creditcard.csv")
    preprocessed_data = Preprocessing(rawdata, drop_time=True)
    dataset = preprocessed_data.get_contrastive_dataset(test_size=0.2, random_state=42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize the combined model
    model = ContrastiveModel(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize Triplet Margin Loss (Replaces nt_xent_loss)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # Train
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        
        # Unpack the triplet: anchor, positive, negative
        for anchor, positive, negative in loader:
            anchor   = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Forward pass all three through backbone + head
            proj_anchor = model(anchor)
            proj_pos    = model(positive)
            proj_neg    = model(negative)

            # Compute Triplet Margin Loss
            loss = criterion(proj_anchor, proj_pos, proj_neg)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{epochs}] - Triplet Loss: {avg_loss:.8f}")

    # Isolate and save ONLY the backbone encoder (discarding the contrastive head)
    print("Training complete. Extracting and saving the trained backbone...")
    torch.save(model.backbone.state_dict(), "pretrained_tabular_encoder.pth")


def main():
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