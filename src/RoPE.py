# Attention pattern depends only on the relative positions of the tokens, not their absolute positions.
# k = Rm_theta * W_q * x -> key at pos. m
# q = Rn_theta * W_k * x -> query at pos. n
# attn = q @ k^T = (W_q * x) @ Rm_theta^T @ Rn_theta @ (W_k * x)^T  = (W_q * x) @ (W_k * x)^T @ R_(n-m)_theta
# R_(n-m)_theta can be computed optimally using cos and sin matrices.

# RoPECache -> Precompute the cos and sin matrices for a max sequence length and store them. Otherwise compute and memory will explode.

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class RoPECache:
    def __init__(self, d_head: int, max_pos: int, base: int = 10000, device=None):
        assert d_head % 2 == 0, "d_head must be even for RoPE"
        self.d_head = d_head
        self.max_pos = max_pos
        self.base = base
        self.device = device
        self.build = _build(max_pos)

    
    def _build(self, max_pos: int):
        self.max_pos = max_pos
        inv_freq = 1.0/ (self.base ** (torch.arange(0, self.d_head, 2, device = self.device).float() / self.d_head))
        t = torch.arange(max_pos, device = self.device).float()
        freqs = torch.outer(t, inv_freq) # (max_pos, d_head/2)
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

    def get(self, positions: torch.Tensor):

        # positions: (T,) or (B, T)       
        if positions.dim() == 2:
            positions = positions[0]

        req = int(positions.max().item()) + 1 if positions.numel() > 0 else 1

        if req > self.max_pos:
            self._build = (max(req, int(self.max_pos * 2)))

        cos = self.cos[positions] # (T, d_head/2)
        sin = self.sin[positions] # (T, d_head/2)   

        return cos, sin


