import torch
import torch.nn as nn

"""
Slighly more efficient than LayerNorm as these operations can be fused into a single operation in the GPU.
We don't need to keep track of mean and variance, just the root mean square (RMS) of the elements.
"""
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = (x / rms) * self.weight
        return x_norm