import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model, in_features):
        super().__init__()

        # Latent space dimention
        self.d_model = d_model

        # Defines the matrices for the attention mechanism
        self.query = nn.Linear(in_features, self.d_model)
        self.key = nn.Linear(in_features, self.d_model)
        self.values = nn.Linear(in_features, self.d_model)

    def forward(self, x):
        
        """
        input size: (batch, in_features)
        linear transformations size: (batch, d_model, in_features)
        query, key and values matrices size: (batch, d_model)
        scores / weights size: (batch, batch)
        representation size: (batch, d_model)
        """

        # Calculates the self attention matrices
        query_matrix = self.query(x)
        key_matrix = self.key(x)
        values_matrix = self.values(x)

        # Calculates scores and weights. 
        scores = torch.matmul(query_matrix, key_matrix.transpose(-2, -1)) / np.sqrt(self.d_model)
        weights = F.softmax(input=scores, dim=-1)

        representation = torch.matmul(weights, values_matrix)

        return representation, weights