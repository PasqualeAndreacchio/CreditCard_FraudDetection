import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.Datasets.datasets import ContrastiveDataset
from src.models.AttentionEncoder import Encoder, ContrastiveHead, InputEmbedding
from src.Datasets.preprocess import Preprocessing

from src.utils import TripletNTXentLoss


data = pd.read_csv("data/creditcard.csv")

dataset = ContrastiveDataset(df = data)
loader = DataLoader(
    dataset=dataset, 
    batch_size=512, 
    shuffle=True
)

model = nn.Sequential(
    InputEmbedding(d_embed=64),
    Encoder(d_embed=64, d_ff=32, num_heads=2, dropout=0.1), 
    ContrastiveHead(dim_in=64, dim_out=64, p_drop=0.1)
)   

criterion = TripletNTXentLoss(temperature=0.1)
