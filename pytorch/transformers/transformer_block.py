import torch
from torch import nn
import torch.nn.functional as F
from self_attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, k, n_heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(k, n_heads=heads, mask=mask)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        
        # We’ve made the relatively arbitrary choice of making the hidden layer 
        # of the feedforward 4 times as big as the input and output. 
        # Smaller values may work as well, and save memory, 
        # but it should be bigger than the input/output layers.
        self.ff = nn.Sequential(
          nn.Linear(k, ff_hidden_mult * k),
          nn.ReLU(),
          nn.Linear(ff_hidden_mult * k, k))
        
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x