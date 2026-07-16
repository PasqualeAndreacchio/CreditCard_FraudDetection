"""
Models sub-package for Credit Card Fraud Detection.

Available models:
    - FFNNAutoencoder: Feed-Forward Neural Network Autoencoder for anomaly detection.
"""

from src.models.FFNNDecoder import FFNNAutoencoder

__all__ = ["FFNNAutoencoder"]
