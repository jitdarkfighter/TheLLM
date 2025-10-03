import torch
import torch.nn as nn

from src.Model.rmsnorm import RMSNorm
from src.Model.swiglu import SwiGLU
from src.Model.attention import SlidingWindowAttention


## OG Decoder Block (with self-attention and FFN)
# from old.mha import MultiHeadSelfAttention
# from old.ffn import FeedForwardNetwork
# class DecoderBlock(nn.Attention):
#     def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(d_model)
#         self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
#         self.ln2 = nn.LayerNorm(d_model)
#         self.ffn = FeedForwardNetwork(d_model, dropout = dropout)
    
#     def forward(self, x: torch.Tensor):
#         x = x + self.attn(self.ln1(x))[0]
#         x = x + self.ffn(self.ln2(x))
#         return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0,
                 max_pos: int = 4096, sliding_window: int | None = None, attention_sink: int = 0,
                 n_kv_head: int | None = None,device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        """
        Transformer block with sliding window attention and either SwiGLU or FFN.

        Args:
            d_model: embedding dimension
            n_head: number of attention heads
            dropout: dropout rate
            max_pos: maximum position for RoPE
            sliding_window: size of the sliding window (None for full attention) [set to None when training]
            attention_sink: number of tokens to always attend to at the start
            n_kv_head: number of key/value heads (for GQA, None means same as n_head) [set to None when training]
            use_swiglu: whether to use SwiGLU or standard FFN
            device: device to run the model on
        """
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attention = SlidingWindowAttention(d_model = d_model, n_head = n_head, dropout = dropout, max_pos = max_pos, 
                                                sliding_window = sliding_window, attention_sink = attention_sink, n_kv_head = n_kv_head, 
                                                device = device)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_dim_factor = 4,  dropout = dropout)

    def forward(self, x: torch.Tensor, kv_cache = None, start_pos: int = 0):
        a, kv_cache = self.attention(self.ln1(x), kv_cache, start_pos)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, kv_cache
