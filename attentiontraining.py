import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.AttentionEncoder import Encoder, Classifier
from src.preprocess import Preprocessing

# Read raw data
rawdata = pd.read_csv("data/creditcard.csv")

# Apply SMOTE to the dataset
preprocess = Preprocessing(rawdata)
Xtrain_smote, Xtest_smote, Ytrain_smote, Ytest_smote = preprocess.get_smote_dataset(test_size=0.2)

# Transform the dataset into the right data type and format
Xtrain = torch.tensor(Xtrain_smote.values, torch.float32)
Ytrain = torch.tensor(Ytrain_smote.values, torch.float32)
train_dataset = TensorDataset(Xtrain, Ytrain)

# Define dataloader
batch_size = 32
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True 
)

# Define the model
encoder = Encoder(d_embed=128, d_ff=64, num_heads=4)
classifier = Classifier(d_embed=128, dropout=0.1)

