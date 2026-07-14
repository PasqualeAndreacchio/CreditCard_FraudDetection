"""
Feed-Forward Neural Network Autoencoder for Anomaly Detection.

Architecture:
    Input (N features) → Encoder → Latent Space → Decoder → Reconstructed (N features)

The model is trained on *normal* transactions only. At inference time,
fraudulent transactions produce higher reconstruction errors than normal ones,
enabling anomaly-based fraud detection.

All architectural parameters (layer sizes, activation, dropout, batch norm)
are read from a YAML configuration file — no code changes needed to scale
the network.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ─── Activation Registry ────────────────────────────────────────────────────

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
}


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name.

    Args:
        name: One of ``relu``, ``leaky_relu``, ``gelu``, ``selu``.

    Raises:
        ValueError: If the name is not recognised.
    """
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS.keys())}."
        )
    return _ACTIVATIONS[key]()


# ─── Building Blocks ────────────────────────────────────────────────────────

class FFNNBlock(nn.Module):
    """Single feed-forward block: Linear → [BatchNorm] → Activation → [Dropout].

    A reusable building unit that composes the Encoder and Decoder stacks.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        activation: Name of the activation function.
        dropout: Dropout probability (0.0 disables dropout).
        batch_norm: Whether to apply Batch Normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]

        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))

        layers.append(_get_activation(activation))

        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Encoder ────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Encoder network that compresses input features into a latent space.

    Dynamically builds a stack of :class:`FFNNBlock` layers followed by a
    final linear projection to the latent dimension.

    Args:
        input_dim: Dimensionality of the input features.
        hidden_layers: List of hidden layer sizes (e.g. ``[64, 32, 16]``).
        latent_dim: Dimensionality of the latent (bottleneck) space.
        activation: Activation function name.
        dropout: Dropout rate.
        batch_norm: Whether to use Batch Normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        latent_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(
                FFNNBlock(prev_dim, h_dim, activation, dropout, batch_norm)
            )
            prev_dim = h_dim

        # Final projection to latent space (no activation — raw encoding)
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input ``x`` into latent representation."""
        return self.network(x)


# ─── Decoder ────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """Decoder network that reconstructs input features from a latent vector.

    Mirrors the encoder structure with an ascending sequence of layer sizes,
    terminating with a linear projection back to the original input dimension.

    Args:
        latent_dim: Dimensionality of the latent (bottleneck) space.
        hidden_layers: List of hidden layer sizes (e.g. ``[16, 32, 64]``).
        output_dim: Dimensionality of the reconstructed output (== input_dim).
        activation: Activation function name.
        dropout: Dropout rate.
        batch_norm: Whether to use Batch Normalization.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = latent_dim

        for h_dim in hidden_layers:
            layers.append(
                FFNNBlock(prev_dim, h_dim, activation, dropout, batch_norm)
            )
            prev_dim = h_dim

        # Final projection back to input space (no activation — linear output)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector ``z`` back to input space."""
        return self.network(z)


# ─── Autoencoder ─────────────────────────────────────────────────────────────

class FFNNAutoencoder(nn.Module):
    """Feed-Forward Neural Network Autoencoder for anomaly-based fraud detection.

    Composes an :class:`Encoder` and a :class:`Decoder` into a single module.
    Trained on *normal* (non-fraudulent) transactions, it learns to reconstruct
    them accurately. Fraudulent transactions incur higher reconstruction error,
    which is used as an anomaly score.

    Args:
        input_dim: Number of input features.
        encoder_layers: Hidden layer sizes for the encoder.
        latent_dim: Bottleneck dimensionality.
        decoder_layers: Hidden layer sizes for the decoder.
        activation: Activation function name.
        dropout: Dropout rate.
        batch_norm: Whether to use Batch Normalization.

    Example::

        model = FFNNAutoencoder.from_config(config["model"])
        reconstruction = model(x)
        latent = model.encode(x)
        errors = model.compute_reconstruction_error(x)
    """

    def __init__(
        self,
        input_dim: int,
        encoder_layers: list[int],
        latent_dim: int,
        decoder_layers: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_layers=encoder_layers,
            latent_dim=latent_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_layers=decoder_layers,
            output_dim=input_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        # Xavier initialisation for faster convergence
        self._init_weights()

    # ── Weight Initialisation ───────────────────────────────────────────

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Forward Methods ─────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input into the latent representation.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Latent tensor of shape ``(batch, latent_dim)``.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from a latent vector.

        Args:
            z: Latent tensor of shape ``(batch, latent_dim)``.

        Returns:
            Reconstructed tensor of shape ``(batch, input_dim)``.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode → decode.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Reconstructed tensor of shape ``(batch, input_dim)``.
        """
        z = self.encode(x)
        return self.decode(z)

    # ── Anomaly Scoring ─────────────────────────────────────────────────

    @torch.no_grad()
    def compute_reconstruction_error(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """Compute per-sample reconstruction error (anomaly score).

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            reduction: How to reduce across features. ``"mean"`` computes
                MSE per sample; ``"sum"`` computes SSE per sample;
                ``"none"`` returns full error tensor.

        Returns:
            Reconstruction error. Shape depends on *reduction*:
            - ``"mean"`` / ``"sum"``: ``(batch,)``
            - ``"none"``: ``(batch, input_dim)``
        """
        self.eval()
        x_hat = self.forward(x)
        error = (x - x_hat) ** 2

        if reduction == "mean":
            return error.mean(dim=1)
        elif reduction == "sum":
            return error.sum(dim=1)
        elif reduction == "none":
            return error
        else:
            raise ValueError(f"Unknown reduction '{reduction}'. Use 'mean', 'sum', or 'none'.")

    # ── Factory Method ──────────────────────────────────────────────────

    @classmethod
    def from_config(cls, model_config: dict[str, Any]) -> "FFNNAutoencoder":
        """Instantiate the autoencoder from a configuration dictionary.

        Args:
            model_config: The ``model`` section of the YAML configuration.

        Returns:
            An initialised :class:`FFNNAutoencoder` instance.
        """
        instance = cls(
            input_dim=model_config["input_dim"],
            encoder_layers=model_config["encoder_layers"],
            latent_dim=model_config["latent_dim"],
            decoder_layers=model_config["decoder_layers"],
            activation=model_config.get("activation", "relu"),
            dropout=model_config.get("dropout", 0.0),
            batch_norm=model_config.get("batch_norm", True),
        )
        logger.info(
            "FFNNAutoencoder created: input_dim=%d, latent_dim=%d, "
            "encoder=%s, decoder=%s",
            instance.input_dim,
            instance.latent_dim,
            model_config["encoder_layers"],
            model_config["decoder_layers"],
        )
        return instance

    # ── Summary ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the model architecture."""
        lines = [
            "=" * 60,
            "  FFNNAutoencoder Summary",
            "=" * 60,
            f"  Input dimension  : {self.input_dim}",
            f"  Latent dimension : {self.latent_dim}",
            "",
            "  Encoder:",
        ]
        for name, module in self.encoder.network.named_children():
            lines.append(f"    [{name}] {module}")
        lines.append("")
        lines.append("  Decoder:")
        for name, module in self.decoder.network.named_children():
            lines.append(f"    [{name}] {module}")
        lines.append("")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(f"  Total parameters     : {total_params:,}")
        lines.append(f"  Trainable parameters : {trainable_params:,}")
        lines.append("=" * 60)
        return "\n".join(lines)
