import torch
import torch.nn as nn
from pydantic import BaseModel
from .AttentionEncoder import Encoder, InputEmbedding
from .FFNNDecoder import Decoder

class Complete_Autoencoder(nn.Module):
    """
    Hybrid autoencoder combining an Attention-based encoder and a FFNN decoder.

    The encoder maps a raw feature vector to a sequence of per-feature
    embeddings and processes them through an encoder block.
    The resulting 3-D tensor is collapsed to a 2-D latent vector via mean
    pooling before being passed to the feed-forward decoder.

    Args of the constructor:
        encoder_config: Configuration for the Attention encoder.
            - d_embed (int): Embedding dimension for each feature (default 32).
            - d_ff (int): Hidden size of the feed-forward sub-layer (default 64).
            - num_heads (int): Number of attention heads (default 4).
              Must evenly divide d_embed.
            - dropout (float): Dropout probability (default 0.1).

        decoder_config: Configuration for the FFNN decoder.
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
        model_cfg: dict = None,
    ) -> None:
        super().__init__()

        # Safely handle if model_cfg is None
        model_cfg = model_cfg or {}

        # Use .get() to avoid KeyErrors if the keys are missing
        encoder_config = model_cfg.get('encoder')
        decoder_config = model_cfg.get('decoder')

        # Streamlined validation
        if not isinstance(encoder_config, dict) or not isinstance(decoder_config, dict):
            raise ValueError("model_cfg must contain valid 'encoder' and 'decoder' dictionaries.")

        # Sanity checks 
        if encoder_config['d_embed'] % encoder_config['num_heads'] != 0:
            raise ValueError("num_heads must evenly divide d_embed")
        if decoder_config['latent_dim'] != encoder_config['d_embed']:
            raise ValueError("latent_dim must match d_embed")

        # Build the components
        self.embedder = InputEmbedding(encoder_config['d_embed'])

        # The Magic of Unpacking: Pass the whole dictionary at once!
        self.encoder = Encoder(**encoder_config)
        
        # Assuming your Decoder also matches the keys in decoder_config:
        self.decoder = Decoder(**decoder_config)


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