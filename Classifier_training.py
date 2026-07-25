import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.Classifier import FraudDetectionMLP
from src.Datasets.preprocess import Preprocessing

# Load and prepare data
rawdata = pd.read_csv("data/creditcard.csv")

# Preprocess data, dropping 'Time' column and applying SMOTE to the training set
preprocess = Preprocessing(rawdata, drop_time=True)
X_train, X_test, y_train, y_test = preprocess.get_smote_dataset(test_size=0.2, random_state=42)


# Create DataLoader
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# Device, Model, Loss, Optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FraudDetectionMLP(input_dim=X_train.shape[1]).to(device)

# BCEWithLogitsLoss is more numerically stable than adding a Sigmoid to the model
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Training Loop
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (samples, labels) in enumerate(loader):
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
            
    print(f"--- Epoch {epoch+1} Average Loss: {total_loss / len(loader):.4f} ---")