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
    pass