import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    """
    Dataset for tabular contrastive learning.
    
    Returns:
        anchor:   A normal transaction.
        positive: An augmented version (added noise) of the anchor.
        negative: A randomly sampled fraudulent transaction.
    """
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "Don't instantiate ContrastiveDataset directly. \n "
            "Use Preprocessing.get_contrastive_dataset() instead to ensure a proper train/val/test split and proper scaling."
        )

    def __len__(self) -> int:
        # Base the dataset length on normal transactions
        return len(self.normal_data)

    def __getitem__(self, idx: int):
        # 1. Anchor: Normal transaction at current index
        anchor = self.normal_data[idx]

        # 2. Positive: Augmented view of the anchor (Gaussian noise)
        noise = torch.randn_like(anchor) * self.noise_std
        positive = anchor + noise

        # 3. Negative: Randomly selected fraud transaction
        random_fraud_idx = torch.randint(0, len(self.fraud_data), (1,)).item()
        negative = self.fraud_data[random_fraud_idx]

        return anchor, positive, negative

    @classmethod
    def from_dataframe(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        noise_std: float = 0.05,
    ) -> "ContrastiveDataset":
        """
        Alternative constructor that builds a ContrastiveDataset from
        already-scaled DataFrames produced by Preprocessing, bypassing
        the CSV-based __init__.

        Args:
            X (pd.DataFrame): Feature matrix (already scaled, 'Time' and
                              'Class' columns must have been removed beforehand).
            y (pd.Series): Binary label series (0 = normal, 1 = fraud),
                           aligned with X.
            noise_std (float): Standard deviation of the Gaussian noise used
                               to generate positive pairs (default: 0.05).
        Returns:
            ContrastiveDataset: A ready-to-use dataset instance.
        """
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        normal_mask = (y_np == 0)
        fraud_mask  = (y_np == 1)

        if normal_mask.sum() == 0:
            raise ValueError("No normal (class=0) samples found in X/y.")
        if fraud_mask.sum() == 0:
            raise ValueError("No fraud (class=1) samples found in X/y.")

        # Bypass __init__ to avoid re-reading a CSV
        instance = cls.__new__(cls)
        instance.normal_data = torch.tensor(X_np[normal_mask], dtype=torch.float32)
        instance.fraud_data  = torch.tensor(X_np[fraud_mask],  dtype=torch.float32)
        instance.noise_std   = noise_std
        return instance