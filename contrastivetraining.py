import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from imblearn.over_sampling import SMOTE

from src.Datasets.datasets import ContrastiveDataset
from src.models.AttentionEncoder import ContrastiveModel

from src.utils import SupervisedContrastiveLoss


data = pd.read_csv("data/creditcard.csv")
smote = SMOTE(random_state=42)

X, y = smote.fit_resample(data.iloc[:,:-1], data.iloc[:, -1])
smotedata = pd.concat([X, y], axis=1)

dataset = ContrastiveDataset(df = smotedata)
loader = DataLoader(
    dataset=dataset, 
    batch_size=1024, 
    shuffle=True
)

model = ContrastiveModel()


criterion = SupervisedContrastiveLoss(temperature=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (sample, label) in enumerate(loader):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        embedded = model(sample)
        
        # Compute NT-Xent Loss
        loss = criterion(embedded,label)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
    print(f"--- Epoch {epoch+1} Average Loss: {total_loss / len(loader):.4f} ---")
