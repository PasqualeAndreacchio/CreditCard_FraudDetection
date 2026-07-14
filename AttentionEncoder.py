import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEmbedding(nn.Module):
    def __init__(self, d_embed):
        super().__init__()

        self.d_embed = d_embed
        self.embedding = nn.Linear(1, d_embed)

    def forward(self, x):

        x_embed = x.unsqueeze(-1)
        x_embed = self.embedding(x_embed)

        return x_embed

class Attention(nn.Module):
    def __init__(self, d_embed=64, p_drop = 0.1):
        super().__init__()

        # Latent space dimention
        self.d_embed = d_embed

        # Defines the matrices for the attention mechanism
        self.queryw = nn.Linear(d_embed, d_embed)
        self.keyw = nn.Linear(d_embed, d_embed)
        self.valuesw = nn.Linear(d_embed, d_embed)

        # Dropout for stability
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):

        # Calculates the self attention matrices
        queries = self.queryw(x)
        keys = self.keyw(x)
        values = self.valuesw(x)

        # Calculates scores and weights. 
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.d_embed)
        weights = F.softmax(input=scores, dim=-1)

        # Apply dropout 
        weights = self.dropout(weights)

        representation = torch.matmul(weights, values)

        return representation, weights
    

class Encoder(nn.Module):
    def __init__(self, d_embed=64, d_ff=32, dropout=0.1):
        super().__init__()

        # Attention head
        self.attention = Attention(d_embed=d_embed, p_drop=dropout)
        
        # First layernorm 
        self.norm1 =  nn.LayerNorm(d_embed)
        
        # FFNN 
        self.feed_forward = nn.Sequential(
            nn.Linear(d_embed, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_embed)
        )

        # Second layernorm 
        self.norm2 = nn.LayerNorm(d_embed)
    
    def forward(self, x):

        attout, att_weights = self.attention(x)
        norm1out = self.norm1(attout + x)
        ffnnout = self.feed_forward(norm1out)
        norm2out = self.norm2(ffnnout + norm1out)

        return norm2out, att_weights
    
