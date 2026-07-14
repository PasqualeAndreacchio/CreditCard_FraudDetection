"""
Models sub-package for Credit Card Fraud Detection.

Available models:
    - FFNNAutoencoder: Feed-Forward Neural Network Autoencoder for anomaly detection.
"""

from src.models.ffnn_autoencoder import FFNNAutoencoder

__all__ = ["FFNNAutoencoder"]
