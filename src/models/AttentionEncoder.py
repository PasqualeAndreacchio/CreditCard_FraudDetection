import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEmbedding(nn.Module):

    """ 
    This class takes the data coming from the dataset, already preprocessed and transformes
    their dimentions. The incoming data have shape (batch, features), but the attention process
    needs the shape (batch, seq_length, embedding_dim). The forward pass adds the third dimention
    (d_embed) and makes the seq_length correspond to features. 

    Input shape (batch, features) --> output shape (batch, features, d_embed)
    """

    def __init__(self, d_embed):
        super().__init__()

        self.d_embed = d_embed
        self.embedding = nn.Linear(1, d_embed)

    def forward(self, x):

        x_embed = x.unsqueeze(-1)
        x_embed = self.embedding(x_embed)

        return x_embed

def _scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len) --> (batch, heads, seq_len, seq_len)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    attention = F.softmax(scaled, dim=-1)
    # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim) --> (batch, heads, seq_len, head_dim)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim      # Input embedding size
        self.d_model = d_model          # Model embedding size (output of self-attention)
        self.num_heads = num_heads      # Number of parallel attention heads
        self.head_dim = d_model // num_heads  # Dimensionality per head

        # For efficiency, compute Q, K, V for all heads at once with a single linear layer
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        # Final projection, combines all heads' outputs
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()

        # Project x into concatenated q, k, v for ALL heads at once
        qkv = self.qkv_layer(x)

        # reshape into (batch, seq_len, num_heads, 3 * head_dim)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)

        # Rearrange to (batch, num_heads, seq_len, 3 * head_dim)
        qkv = qkv.permute(0, 2, 1, 3)

        # Split the last dimension into q, k, v (each get last dimension of head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (batch, num_heads, seq_len, head_dim)

        # Apply scaled dot product attention to get outputs (contextualized values) and attention weights
        values, attention = _scaled_dot_product(q, k, v, mask)

        # Merge the heads (permute before reshape)
        values = values.permute(0, 2, 1, 3)   # (batch, seq_len, heads, head_dim)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

        # Final linear projection to match d_model
        out = self.linear_layer(values)

        return out

class Encoder(nn.Module):
    def __init__(self, d_embed=32, d_ff=16, num_heads=4, dropout=0.1):
        super().__init__()

        # Attention head
        self.attention = MultiheadAttention(d_embed, d_embed, num_heads)
        
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

        attout = self.attention(x)
        norm1out = self.norm1(attout + x)
        ffnnout = self.feed_forward(norm1out)
        norm2out = self.norm2(ffnnout + norm1out)

        return norm2out #, att_weights (Batch, nfeatutes, d_model)
