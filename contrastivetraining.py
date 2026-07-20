import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from src.Datasets.datasets import ContrastiveDataset
from src.models.AttentionEncoder import Encoder, ContrastiveHead, InputEmbedding
from src.Datasets.preprocess import Preprocessing

from src.utils import TripletNTXentLoss


data = pd.read_csv("data/creditcard.csv")

dataset = ContrastiveDataset(df = data)
loader = DataLoader(
    dataset=dataset, 
    batch_size=512, 
    shuffle=True
)

model = nn.Sequential(
    InputEmbedding(d_embed=64),
    Encoder(d_embed=64, d_ff=32, num_heads=2, dropout=0.1), 
    ContrastiveHead(dim_in=64, dim_out=64, p_drop=0.1)
)   

criterion = TripletNTXentLoss(temperature=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (sample, pos, neg) in enumerate(loader):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        out_sample = model(sample)
        out_pos = model(pos)
        out_neg = model(neg)
        
        # Compute NT-Xent Loss
        loss = criterion(out_sample, out_pos, out_neg)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
    print(f"--- Epoch {epoch+1} Average Loss: {total_loss / len(loader):.4f} ---")