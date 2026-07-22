import torch
import torch.nn as nn
import yaml

class FraudAutoencoder(nn.Module):
    def __init__(self,input_dim=30):
        super().__init__()
        
        # 1. The Encoder (Compresses data)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.Tanh(),
            nn.Linear(24, 16),
            nn.Tanh(),
            # The bottleneck (Latent space)
            nn.Linear(16, 8), 
            nn.ReLU()
        )
        
        # 2. The Decoder (Reconstructs data)
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 24),
            nn.Tanh(),
            # Output layer matches input dimensions
            nn.Linear(24, input_dim) 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded