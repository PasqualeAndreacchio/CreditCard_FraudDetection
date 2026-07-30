import torch
import torch.nn as nn

from .autoencoder import Encoder

class FraudDetectionMLP(nn.Module):
    """Supervised MLP classifier for direct fraud detection.

    Reuses the same Encoder backbone as FraudAutoencoder / ContrastiveModel,
    followed by a two-layer classification head with BatchNorm, ReLU and
    Dropout.  Outputs a single raw logit (use BCEWithLogitsLoss for training).

    Args:
        config: Configuration dictionary containing 'model.input_dim',
            'model.hidden_dims', and 'model.dropout'.
    """

    def __init__(self, config: dict) -> None:
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

            nn.Linear(latent_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.classifier(latent)