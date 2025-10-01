import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from flash_attn import flash_attn_func
from attn_mask import casual_mask

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.n_head = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias = False)
        self.proj = nn.Linear(d_model, d_model, bias = False)
        
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape # Input has B batches each with T tokens with Each token having C dims (d_model)
        qkv = self.qkv(x) # Compute q, k, v with a single linear layer
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head) # Split the embedding into self.n_head different heads, each with self.d_head dims
        q, k, v = qkv.unbind(dim = 2) # Separate q, k, v for each head
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) # Now each has shape (B, n_head, T, d_head)

        scale = 1.0 / math.sqrt(self.d_head)
        attn_pattern = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = casual_mask(T, device = x.device)
        attn_pattern = attn_pattern.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_pattern, dim = -1) # -1 means along the row
        attn_output = torch.matmul(attn_weights, v) # (B, n_head, T, d_head)

        # Merge the heads back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) # (B, T, d_model)

        """
        Concatenation just stacks, it doesn't allow learned combinations.
        the proj layer W^O will learn to combine the information from different heads.
        Each output element is now a weighted sum of information from all heads. This will help capture complex dependencies.
        """
        out = self.proj(attn_output) # (B, T, d_model)

        return out, attn_weights