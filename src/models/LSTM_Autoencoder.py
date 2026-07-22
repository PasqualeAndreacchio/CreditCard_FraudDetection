import torch
import torch.nn as nn

class LSTM_Autoencoder(nn.Module):
    """
    LSTM-based Autoencoder architecture for anomaly detection in sequential data.
    This network compresses a 3D input sequence of shape (batch_size, seq_len, input_dim)
    into a fixed-size latent bottleneck vector, and then reconstructs the original
    sequence from that latent representation.
    """

    def __init__(
        self, 
        input_dim: int = 32, 
        hidden_dim: int = 64, 
        latent_dim: int = 16, 
        num_layers: int = 2, 
        dropout: float = 0.2
    ) -> None:
        """
        Initialize the LSTM Autoencoder architecture and validate parameter types.
        Args:
            input_dim: Number of input features per timestep. Must be a positive integer.
            hidden_dim: Number of hidden units in the LSTM layers. Must be a positive integer.
            latent_dim: Dimensionality of the latent bottleneck space. Must be a positive integer.
            num_layers: Number of stacked LSTM layers. Must be a positive integer.
            dropout: Dropout probability applied between LSTM layers. Must be a float in [0.0, 1.0].
        Raises:
            TypeError: If any parameter is not of the expected type.
            ValueError: If any parameter has an invalid numeric value.
        """
        # Sanity checks on parameter types and values
        if not isinstance(input_dim, int) or isinstance(input_dim, bool):
            raise TypeError(f"input_dim must be an integer, got {type(input_dim).__name__}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool):
            raise TypeError(f"hidden_dim must be an integer, got {type(hidden_dim).__name__}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        if not isinstance(latent_dim, int) or isinstance(latent_dim, bool):
            raise TypeError(f"latent_dim must be an integer, got {type(latent_dim).__name__}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        if not isinstance(num_layers, int) or isinstance(num_layers, bool):
            raise TypeError(f"num_layers must be an integer, got {type(num_layers).__name__}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        if not isinstance(dropout, (float, int)) or isinstance(dropout, bool):
            raise TypeError(f"dropout must be a float or int, got {type(dropout).__name__}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {dropout}")

        super(LSTM_Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = float(dropout)

        # Encoder layers: first LSTM, then a linear layer for the bottleneck
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers: linear projection, LSTM, and output feature projection
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.decoder_output = nn.Linear(hidden_dim, input_dim)


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of input sequences into latent bottleneck vectors.
        Args:
            x: Input sequence tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            Latent representation tensor of shape (batch_size, latent_dim).
        """
        # Pass through encoder LSTM: out shape (batch, seq_len, hidden_dim), hidden shape (num_layers, batch, hidden_dim)
        _, (hidden, _) = self.encoder_lstm(x)

        # Extract the hidden state of the last LSTM layer
        last_hidden = hidden[-1, :, :]

        # Project to latent dimension
        latent = self.encoder_fc(last_hidden)
        return latent


    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent bottleneck vectors back into reconstructed sequences.
        Args:
            z: Latent vector tensor of shape (batch_size, latent_dim).
            seq_len: Target sequence length to reconstruct.
        Returns:
            Reconstructed sequence tensor of shape (batch_size, seq_len, input_dim).
        """
        # Project from latent space to hidden dimension
        x = self.decoder_fc(z)

        # Repeat the projetaed hidden state across seq_len timesteps
        x = x.unsqueeze(1).repeat(1, seq_len, 1)

        # Pass through decoder LSTM
        out, _ = self.decoder_lstm(x)

        # Project output to original feature dimension
        reconstructed = self.decoder_output(out)
        return reconstructed


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a complete forward pass (encoding followed by decoding).
        Args:
            x: Input sequence tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            Reconstructed sequence tensor of shape (batch_size, seq_len, input_dim).
        """
        seq_len = x.size(1)
        z = self.encode(x)
        reconstructed = self.decode(z, seq_len)
        return reconstructed


    def reconstruct(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Reconstruct a sequence given an explicit target sequence length.
        Args:
            x: Input sequence tensor of shape (batch_size, seq_len, input_dim).
            seq_len: Target sequence length for the reconstruction.
        Returns:
            Reconstructed sequence tensor of shape (batch_size, seq_len, input_dim).
        """
        z = self.encode(x)
        return self.decode(z, seq_len)


    @torch.no_grad()
    def compute_reconstruction_error(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute the sample-wise reconstruction error (anomaly score).
        Args:
            x: Input sequence tensor of shape (batch_size, seq_len, input_dim).
            reduction: Error aggregation mode ('mean', 'sum', or 'none').
        Returns:
            Reconstruction error tensor per sample of shape (batch_size,) if reduction is 'mean' or 'sum',
            or (batch_size, seq_len, input_dim) if reduction is 'none'.
        Raises:
            ValueError: If reduction mode is not one of 'mean', 'sum', or 'none'.
        """

        self.eval()
        x_hat = self.forward(x)
        error = (x - x_hat) ** 2

        if reduction == "mean":
            return error.mean(dim=(1, 2))
        elif reduction == "sum":
            return error.sum(dim=(1, 2))
        elif reduction == "none":
            return error
        else:
            raise ValueError(f"Unsupported reduction mode: '{reduction}'. Choose from 'mean', 'sum', or 'none'.")


    def get_encoder(self) -> nn.Module:
        """
        Return the Encoder sub-module components.
        Returns:
            nn.Sequential containing the encoder LSTM and linear projection.
        """
        return nn.Sequential(
            self.encoder_lstm,
            self.encoder_fc
        )
        

    def get_decoder(self) -> nn.Module:
        """Return the Decoder sub-module components.

        Returns:
            nn.Sequential containing the decoder projection, LSTM, and output linear layer.
        """
        return nn.Sequential(
            self.decoder_fc,
            self.decoder_lstm,
            self.decoder_output
        )