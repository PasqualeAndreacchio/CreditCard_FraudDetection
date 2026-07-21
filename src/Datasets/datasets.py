import torch
from torch.utils.data import Dataset

class ContrastiveDataset(Dataset):
    def __init__(self, df):
        self.labels = torch.tensor(df["Class"].values, dtype=torch.long)
        features = df.drop(columns=["Class"]).values
        self.data = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        # Simply return the anchor and its actual class label
        return self.data[idx], self.labels[idx]