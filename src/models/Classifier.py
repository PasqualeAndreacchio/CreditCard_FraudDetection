import torch
import torch.nn as nn

class FraudDetectionMLP(nn.Module):
    def __init__(self, input_dim=29, hidden_dim=128, dropout=0.3):
        super().__init__()
        
        # We use BatchNorm1d which is incredibly highly recommended for tabular data 
        # to keep the gradients stable across features of different scales.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            
            # Output layer: 1 node for binary classification
            nn.Linear(hidden_dim // 4, 1) 
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        return self.network(x)