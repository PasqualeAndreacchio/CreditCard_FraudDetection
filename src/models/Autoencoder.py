import torch
import torch.nn as nn
import yaml
import torch.nn.functional as F

class ContrastiveHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.network(x)



class Encoder(nn.Module):
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

    def forward(self, x):
        return self.encoder(x)


class ContrastiveModel(nn.Module):
    """
    Wraps the Encoder and the ContrastiveHead into a single training model.
    """
    def __init__(self, input_dim=30):
        super().__init__()
        
        # Instantiate your encoder
        self.backbone = Encoder(input_dim=input_dim)
        
        # Instantiate your contrastive head.
        # The encoder bottleneck outputs 8 features, so input_dim must be 8.
        self.head = ContrastiveHead(input_dim=8, hidden_dim=16, out_dim=8)

    def forward(self, x):
        # 1. Extract representation
        h = self.backbone(x)
        
        # 2. Map to contrastive space
        z = self.head(h)
        
        # 3. L2 Normalize for cosine similarity
        return F.normalize(z, dim=-1)



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


