import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function used by both encoder and decoder
# It creates a block of Linear -> BatchNorm -> ReLU
def _make_block(in_features: int, out_features: int) -> nn.Sequential:
    """
    Standard feedforward block used by both encoder and decoder.
    Order: Linear -> BatchNorm -> ReLU.
    BatchNorm is placed after the linear projection and before activation)
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
    )


class Encoder(nn.Module):
    """
    Compresses the input features down to a latent bottleneck.
    """

    def __init__(self, input_dim: int, hidden_dims: list) -> None:
        """
        Build the compression path as a sequence of Linear → BN → ReLU blocks,
        ending with a plain Linear bottleneck (no activation).

        Args:
            input_dim (int): Number of input features (e.g. 29 for the credit card dataset).
            hidden_dims (list[int]): Width of each layer, from first hidden to bottleneck.
                Example: [24, 16, 8] produces 29→24→16→8, where 8 is the latent dimension.
        """
        super().__init__()

        # Build the full list of layer widths: input first, then all hidden dims.
        dims = [input_dim] + hidden_dims

        blocks = []

        # Create hidden layers blocks using the helper function.
        for in_d, out_d in zip(dims[:-2], dims[1:-1]):
            blocks.append(_make_block(in_d, out_d))

        # Bottleneck projection: plain Linear, no activation.
        blocks.append(nn.Linear(dims[-2], dims[-1]))

        self.encoder = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Reconstructs the input from the latent bottleneck — symmetric mirror
    of the Encoder.
    """

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int) -> None:
        """
        Build the expansion path from the latent bottleneck back to the
        original feature dimension. It mirrors the encoder's architecture
        but in reverse order.

        Args:
            latent_dim (int): Dimension of the bottleneck (last encoder layer).
            hidden_dims (list[int]): Widths of the intermediate layers, used
                in reverse order (i.e., from narrowest to widest).
            output_dim (int): Dimension of the final reconstructed output
                (must match the original input_dim).
        """
        super().__init__()

        # Full sequence of widths: bottleneck -> intermediate steps -> output.
        dims = [latent_dim] + hidden_dims + [output_dim]

        blocks = []

        # Intermediate expansion steps
        for in_d, out_d in zip(dims[:-2], dims[1:-1]):
            blocks.append(_make_block(in_d, out_d))

        # Final projection back to input space: plain Linear, no activation
        blocks.append(nn.Linear(dims[-2], dims[-1]))

        self.decoder = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class ContrastiveHead(nn.Module):
    """
    Maps the encoder's bottleneck embedding to a contrastive projection space.
    Used only during contrastive pre-training; discarded afterwards.
    """

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContrastiveModel(nn.Module):
    """
    Backbone (Encoder) + ContrastiveHead wired together for triplet training.
    After training, only the backbone is saved and reused.
    """

    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()

        self.backbone = Encoder(input_dim=input_dim, hidden_dims=hidden_dims)

        # The head input matches the bottleneck (last value in hidden_dims).
        bottleneck_dim = hidden_dims[-1]
        self.head = ContrastiveHead(
            input_dim=bottleneck_dim,
            hidden_dim=bottleneck_dim * 2,
            out_dim=bottleneck_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract compact representation from the backbone.
        h = self.backbone(x)

        # Project to contrastive space and L2-normalise so that cosine
        # similarity equals the dot product (stable for triplet loss).
        z = self.head(h)
        return F.normalize(z, dim=-1)


class FraudAutoencoder(nn.Module):
    """
    Full encoder-decoder autoencoder for anomaly detection.

    Anomaly detection at inference time works by computing the reconstruction
    error for each sample — fraudulent transactions tend to reconstruct poorly
    because the network was trained only on normal data.

    Architecture is driven entirely by the 'model' section of config.yaml:
        model:
          input_dim: 29
          hidden_dims: [24, 16, 8]

    The decoder is automatically the mirror of the encoder, so only one list
    of dimensions is needed in the config.
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config["model"]
        input_dim: int = model_cfg["input_dim"]
        hidden_dims: list = model_cfg["hidden_dims"]

        # Encoder: input_dim → hidden_dims (last element is the bottleneck).
        self.encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims)

        # Decoder: mirror path.
        # Bottleneck is the last encoder dim; intermediate expansion steps are
        # hidden_dims reversed (minus the bottleneck itself); output is input_dim.
        bottleneck_dim = hidden_dims[-1]
        decoder_hidden = list(reversed(hidden_dims[:-1]))
        self.decoder = Decoder(
            latent_dim=bottleneck_dim,
            hidden_dims=decoder_hidden,
            output_dim=input_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
