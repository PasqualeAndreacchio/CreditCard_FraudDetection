"""
Train sub-package for Credit Card Fraud Detection.

Provides:
    - Trainer: Generic training pipeline for reconstruction and classification tasks.
    - ContrastiveTrainer: Contrastive pre-training with TripletMarginLoss.
"""

from src.Train.trainer import Trainer
from src.Train.contrastive_trainer import ContrastiveTrainer

__all__ = ["Trainer", "ContrastiveTrainer"]
