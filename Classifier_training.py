import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.Datasets.preprocess import Preprocessing
from src.models.Classifier import FraudDetectionMLP
from src.Datasets.preprocess import Preprocessing

# Load dataset and preprocess dropping 'Time' column and duplicates/null values
data = pd.read_csv("data/creditcard.csv")
preprocessor = Preprocessing(data, drop_time=True)

# Get train/test split with appropriate scaling and SMOTE only on the training set
X_train, X_test, y_train, y_test = preprocessor.get_smote_dataset(test_size=0.2, random_state=42)
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# Device, Model, Loss, Optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FraudDetectionMLP(input_dim=X_train.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Training Loop
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    for batch_idx, (samples, labels) in enumerate(pbar):
        # Move data to the same device as the model
        samples, labels = samples.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(samples)

        # Calculate loss
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # Update the bar with the running average loss
        pbar.set_postfix(loss=f"{total_loss / (batch_idx + 1):.4f}")

    avg_loss = total_loss / len(loader)
    pbar.set_postfix(loss=f"{avg_loss:.4f}")
    print(f"Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")