import torch
from dataclasses import dataclass

# no. of entries in cache = 2 * n_tokens * d_head * n_head * n_layers * 4 bytes (float32) * Batch size(optional)
# for 1B params model with 2048 tokens context, 8 heads, 64 d_head, 16 layers = 4 * 2048 * 64 * 8 * 16 * 4 = 2GB
@dataclass
class KVCache:
    k: torch.Tensor
    v: torch.Tensor

    @property
    def T(self):
        return self.k.size(2)

# Attention sinks.
class SlidingKV:
    def __init__(self, window, sink: int = 0):
        self.window = window
        self.sink = sink
        self.k = None
        self.v = None
    
    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim = 2)
            self.v = torch.cat([self.v, v_new], dim = 2)

        # (B, n_heads, seq_len, d_head)
        # Clipping
        if self.k.size(2) > self.window:
            sink_k = self.k[:, :, :self.sink, :]
            sink_v = self.v[:, :, :self.sink, :]
            recent_k = self.k[:, :, -self.window:, :]
            recent_v = self.v[:, :, -self.window:, :]
            self.k = torch.cat([sink_k, recent_k], dim = 2)
            self.v = torch.cat([sink_v, recent_v], dim = 2)

        return self.k, self.v



