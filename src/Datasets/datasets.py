import pandas as pd
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
    def __init__(self, csv: str = "data/creditcard.csv", drop_time: bool = True, noise_std: float = 0.05):
        df = pd.read_csv(csv)

        # Optionally drop 'Time' column as per standard preprocessing
        if drop_time and "Time" in df.columns:
            df = df.drop(columns=["Time"])

        # Separate target ('Class') from features
        X = df.drop(columns=["Class"]).to_numpy()
        y = df["Class"].to_numpy()

        # Mask normal (0) and fraud (1) samples
        normal_mask = (y == 0)
        fraud_mask = (y == 1)

        # Convert to PyTorch FloatTensors
        self.normal_data = torch.tensor(X[normal_mask], dtype=torch.float32)
        self.fraud_data = torch.tensor(X[fraud_mask], dtype=torch.float32)
        
        self.noise_std = noise_std

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