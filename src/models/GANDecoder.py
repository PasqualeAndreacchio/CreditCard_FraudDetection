import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, dim_in, nfeatures, p_drop):
        super().__init__()

        self.nfeatures = nfeatures

        self.downsample = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.LayerNorm(dim_in // 2),
            nn.ReLU(), 
            nn.Dropout(p=p_drop),
            nn.Linear(dim_in // 2, 1)
        )

    def forward(self, x):           # x shape (batch, nfeatures, dim_in)  
        
        out1 = self.downsample(x)   # shape (batch, nfeatures, 1)
        out = out1.squeeze(-1)      # shape (batch, nfeatures) 

        return out
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()


