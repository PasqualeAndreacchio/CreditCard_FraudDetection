import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.models.Autoencoder import ContrastiveModel
from src.Datasets.datasets import ContrastiveDataset

originaldata = pd.read_csv("data/creditcard.csv")

normalmask = originaldata["Class"] == 0
fraudmask = originaldata["Class"] == 1

normaldata = originaldata[normalmask]
fraudata = originaldata[fraudmask]

encoder_normal_df, decoder_normal_df = train_test_split(normaldata, test_size=0.5, random_state=42) 
encoder_fraud_df, decoder_fraud_df = train_test_split(fraudata, test_size=0.5, random_state=42) 

# FIX: Wrap the dataframes in a list []
encoder_df = pd.concat([encoder_normal_df, encoder_fraud_df])
decoder_df = pd.concat([decoder_normal_df, decoder_fraud_df])

# FIX: Add index=False
encoder_df.to_csv("data/contrastive.csv", index=False)
decoder_df.to_csv("data/reconstruction.csv", index=False)

# ─── TRAINING LOOP ───────────────────────────────────────────────────────

def train_contrastive_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 256
    epochs = 20
    learning_rate = 1e-3
    
    # NOTE: If your new dataset drops both 'Class' and 'Time', the input dimension 
    # will be 29. If it only drops 'Class', it will be 30. Adjust accordingly.
    input_dim = 29 

    # Dataset and Dataloader (Updated initialization)
    dataset = ContrastiveDataset(csv="data/contrastive.csv", drop_time=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize the combined model
    model = ContrastiveModel(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize Triplet Margin Loss (Replaces nt_xent_loss)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # Train
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        
        # Updated to unpack the triplet: anchor, positive, negative
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Forward pass all three through backbone + head
            proj_anchor = model(anchor)
            proj_pos = model(positive)
            proj_neg = model(negative)

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


if __name__ == "__main__":
    train_contrastive_model()