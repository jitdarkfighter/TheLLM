import math, torch
import torch.nn as nn
import torch.nn.functional as F

from src.Model.rope import RoPECache, apply_rope_single
from src.Model.kv_cache import KVCache  # your existing class

class SlidingWindowAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None, 
                 device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        """
        Sliding window attention mechanism with RoPE and kv cache with attention sink.

        Args:
            n_embd: embedding dimension
            n_head: number of attention heads
            dropout: dropout rate
            max_pos: maximum position for RoPE
            sliding_window: size of the sliding window (None for full attention) [set to None when training]
            attention_sink: number of tokens to always attend to at the start
            n_kv_head: number of key/value heads (for GQA, None means same as n_head) [set to None when training]
            device: device to run the model on
        """
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        self.group_size = self.n_head // self.n_kv_head
        self.d_head = n_embd // n_head

        # Separate projections for Q vs K/V (sizes differ under GQA)
        self.wq  = nn.Linear(n_embd, self.n_head   * self.d_head, bias=False)
        self.wk  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rope_cache: RoPECache | None = None
        self.max_pos = max_pos
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink
        
        self.rope_cache = RoPECache(self.d_head, self.max_pos, device = device)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """x: (B,T,C). If kv_cache given, we assume generation (T small, often 1)."""
        B, T, C = x.shape
        
        # Projections
        q = self.wq(x).view(B, T, self.n_head,   self.d_head).transpose(1, 2)    # (B,H, T,D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)

        # RoPE on *current* tokens (cached keys are already rotated)
        # Relative positional encoding
        pos = torch.arange(start_pos, start_pos + T, device=x.device)
        cos, sin = self.rope_cache.get(pos)
        q = apply_rope_single(q, cos, sin)   # (B,H, T,D)
        k = apply_rope_single(k, cos, sin)   # (B,Hk,T,D)

        # Concatenate past cache (cache is stored in Hk heads)
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk, Tpast+T, D)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # Sliding-window + attention-sink (crop along seq length)
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            s = self.attention_sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)

        # GQA expand: repeat K/V heads to match Q heads before attention
        if self.n_kv_head != self.n_head:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
        else:
            k_attn, v_attn = k_all, v_all

        # Scaled dot-product attention (PyTorch scales internally)
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(q, k_attn, v_attn,
                                           attn_mask=None,
                                           dropout_p=self.dropout.p if self.training else 0.0,
                                           is_causal=is_causal)          # (B,H,T,D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        # Return the processed cache (after sliding window application)
        new_cache = KVCache(k_all, v_all)
        return y, new_cache
    
