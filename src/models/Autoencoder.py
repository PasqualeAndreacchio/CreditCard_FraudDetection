import torch
import torch.nn as nn


class FraudAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 30,
        latent_dim: int = 8,
        dropout_rate: float = 0.1,
    ) -> None:
        """
        Args:
            input_dim: number of input features.
            latent_dim: size of the bottleneck latent space.
            dropout_rate: dropout probability used in hidden blocks.
        """
        super().__init__()

        # --- Encoder ---
        # Linear -> BN -> LeakyReLU -> Dropout for each hidden block.
        # No activation on the bottleneck output so the decoder gets raw latent values.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(24, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            # Bottleneck
            nn.Linear(16, latent_dim),
        )

        # --- Decoder ---
        # Mirrors the encoder: gradual expansion latent_dim -> 16 -> 24 -> input_dim.
        # No activation before the first linear (avoids zeroing negative latent values).
        # No activation on the final layer (features are standardized, so clipping hurts).
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(16, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(24, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded