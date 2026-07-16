import torch
import torch.nn as nn
from pydantic import BaseModel
from AttentionEncoder import Encoder, InputEmbedding
from ffnn_autoencoder import Decoder

# Use BaseModel for better readability and "runtime sanity check"
# If an attribute is missing or of the wrong type, BaseModel will raise an error
class EncoderConfig(BaseModel):
    d_embed:   int   = 32
    d_ff:      int   = 64
    num_heads: int   = 4
    dropout:   float = 0.1

class DecoderConfig(BaseModel):
    latent_dim:    int       = 32
    hidden_layers: list[int] = [16, 32, 64]
    output_dim:    int       = 128
    activation:    str       = "relu"
    dropout:       float     = 0.1
    batch_norm:    bool      = True


class Complete_Autoencoder(nn.Module):
    """
    Hybrid autoencoder combining an Attention-based encoder and a FFNN decoder.

    The encoder maps a raw feature vector to a sequence of per-feature
    embeddings and processes them through an encoder block.
    The resulting 3-D tensor is collapsed to a 2-D latent vector via mean
    pooling before being passed to the feed-forward decoder.

    Args of the constructor:
        encoder_config (EncoderConfig): Configuration for the Attention encoder.
            - d_embed (int): Embedding dimension for each feature (default 32).
            - d_ff (int): Hidden size of the feed-forward sub-layer (default 64).
            - num_heads (int): Number of attention heads (default 4).
              Must evenly divide d_embed.
            - dropout (float): Dropout probability (default 0.1).

        decoder_config (DecoderConfig): Configuration for the FFNN decoder.
            - latent_dim (int): Input dimensionality; must match d_embed (default 32).
            - hidden_layers (list[int]): Intermediate layer sizes (default [16, 32, 64]).
            - output_dim (int): Number of reconstructed output features (default 128).
            - activation (str): One of relu, leaky_relu, gelu, selu (default relu).
            - dropout (float): Dropout probability (default 0.1).
            - batch_norm (bool): Whether to apply Batch Normalisation (default True).

    Important notes:
        The model will raise a ValueError in these cases:
        - If num_heads does not evenly divide d_embed
        - If latent_dim does not match d_embed
    """

    def __init__(
        self,
        encoder_config: EncoderConfig = None,
        decoder_config: DecoderConfig = None,
    ) -> None:
        super().__init__()

        # Use defaults if no config is provided
        if encoder_config is None:
            encoder_config = EncoderConfig()
        if decoder_config is None:
            decoder_config = DecoderConfig()

        # Sanity checks
        if encoder_config.d_embed % encoder_config.num_heads != 0:
            raise ValueError("num_heads must evenly divide d_embed")
        if decoder_config.latent_dim != encoder_config.d_embed:
            raise ValueError("latent_dim must match d_embed")

        # Per-feature linear embedding: (batch, N) -> (batch, N, d_embed)
        self.embedder = InputEmbedding(encoder_config.d_embed)

        # Transformer encoder block: (batch, N, d_embed) -> (batch, N, d_embed)
        self.encoder = Encoder(**encoder_config.model_dump())

        # FFNN decoder: (batch, d_embed) -> (batch, output_dim)
        self.decoder = Decoder(**decoder_config.model_dump())


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed and encode input features into a pooled latent vector.
        Applies the per-feature embedding, runs the encoder block, 
        then collapses the sequence dimension via mean pooling.
        Args:
            x: Raw input tensor of shape (batch, n_features).
        Returns:
            Latent tensor of shape (batch, d_embed).
        """
        x_emb = self.embedder(x)          # (batch, n_features, d_embed)
        enc_out = self.encoder(x_emb)     # (batch, n_features, d_embed)
        latent = enc_out.mean(dim=1)      # (batch, d_embed)
        return latent


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full autoencoder forward pass: embed -> encode -> pool -> decode.
        Args:
            x: Raw input tensor of shape (batch, n_features).
        Returns:
            Reconstructed tensor of shape (batch, output_dim).
        Notes:
            When output_dim == n_features, this matches the input shape
            and the reconstruction error can be used as an anomaly score.
        """
        latent = self.encode(x)           # (batch, d_embed)
        z = self.decoder(latent)          # (batch, output_dim)
        return z