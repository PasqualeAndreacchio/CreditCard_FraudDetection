import torch
import torch.nn as nn

from .Autoencoder import Encoder

class FraudDetectionMLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        input_dim = config["model"]["input_dim"]
        hidden_dims = config["model"]["hidden_dims"]
        latent_dim = hidden_dims[-1]
        dropout = config["model"]["dropout"]

        self.encoder = Encoder(input_dim, hidden_dims)
        self.classifier = nn.Sequential(

            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(), 
            nn.Dropout(dropout), 

            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.BatchNorm1d(latent_dim // 4),
            nn.ReLU(), 
            nn.Dropout(dropout),

            # Out layer
            nn.Linear(latent_dim // 4, 1),
        )
        

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        latent = self.encoder(x)
        prediction = self.classifier(latent)

        return prediction