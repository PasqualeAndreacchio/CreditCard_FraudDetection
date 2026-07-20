import torch
from torch.utils.data import Dataset

class ContrastiveDataset(Dataset):
    def __init__(self, df):
        # Convert everything to PyTorch Tensors for performance
        # Assuming "Class" is 0 for Normal, 1 for Fraud
        self.labels = torch.tensor(df["Class"].values, dtype=torch.long)
        
        # Drop the label column to keep only features
        features = df.drop(columns=["Class"]).values
        self.data = torch.tensor(features, dtype=torch.float32)
        
        # Pre-calculate the indices for each class to make sampling instant
        self.normal_idx = torch.where(self.labels == 0)[0]
        self.fraud_idx = torch.where(self.labels != 0)[0]

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = self.labels[idx]

        # Correct Logic: Positive shares the class, Negative is the opposite class
        if anchor_label == 0:
            # Anchor is Normal -> Positive is Normal, Negative is Fraud
            pos_idx = self.normal_idx[torch.randint(len(self.normal_idx), (1,))]
            neg_idx = self.fraud_idx[torch.randint(len(self.fraud_idx), (1,))]
        else:
            # Anchor is Fraud -> Positive is Fraud, Negative is Normal
            pos_idx = self.fraud_idx[torch.randint(len(self.fraud_idx), (1,))]
            neg_idx = self.normal_idx[torch.randint(len(self.normal_idx), (1,))]

        # .squeeze(0) removes the extra dimension added by randint
        positive = self.data[pos_idx].squeeze(0)
        negative = self.data[neg_idx].squeeze(0)

        return anchor, positive, negative