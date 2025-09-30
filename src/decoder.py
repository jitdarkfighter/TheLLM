import torch
import torch.nn as nn
from src.mha import MultiHeadSelfAttention
from src.ffn import FeedForwardNetwork

class DecoderBlock(nn.Attention):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, dropout = dropout)
    
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x