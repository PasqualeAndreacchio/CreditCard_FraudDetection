import torch
import torch.nn as nn
import yaml

class FCEncoder(nn.Module):
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

    def forward (self, x):
        return self.encoder(x)

class FCDecoder(nn.Module):
    def __init__(self,latent_dim=8, input_dim=30):
        super().__init__()

        # 2. The Decoder (Reconstructs data)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 24),
            nn.Tanh(),
            # Output layer matches input dimensions
            nn.Linear(24, input_dim) 
        )

    def forward(self, x):
        return self.decoder(x)


class LSTM_Encoder(nn.Module):
    """
    LSTM-based Encoder architecture extracted from the LSTM_Autoencoder.
    This network compresses a 3D input sequence of shape (batch_size, seq_len, input_dim)
    into a fixed-size latent bottleneck vector.
    """

    def __init__(
        self, 
        input_dim: int = 32, 
        hidden_dim: int = 64, 
        latent_dim: int = 16, 
        num_layers: int = 2, 
        dropout: float = 0.2
    ) -> None:
        
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

        super(LSTM_Encoder, self).__init__()

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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class FraudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    