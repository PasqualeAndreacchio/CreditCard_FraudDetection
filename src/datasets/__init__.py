"""
Datasets sub-package for Credit Card Fraud Detection.

Provides:
    - Preprocessing: Data cleaning, splitting, scaling, SMOTE, and contrastive dataset creation.
    - ContrastiveDataset: PyTorch Dataset yielding anchor/positive/negative triplets.
"""

from src.datasets.preprocess import Preprocessing
from src.datasets.datasets import ContrastiveDataset

__all__ = ["Preprocessing", "ContrastiveDataset"]
