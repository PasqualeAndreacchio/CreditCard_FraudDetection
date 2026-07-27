"""
Models sub-package for Credit Card Fraud Detection.

Available models:
    - FraudAutoencoder: Feed-Forward Autoencoder for anomaly-based fraud detection.
    - FraudDetectionMLP: Supervised MLP classifier for direct fraud classification.
    - ContrastiveModel: Encoder backbone + projection head for contrastive pre-training.
"""

from src.models.Autoencoder import FraudAutoencoder, ContrastiveModel, Encoder, Decoder
from src.models.Classifier import FraudDetectionMLP

__all__ = [
    "FraudAutoencoder",
    "ContrastiveModel",
    "Encoder",
    "Decoder",
    "FraudDetectionMLP",
]
