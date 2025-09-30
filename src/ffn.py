import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mult)
        self.fc2 = nn.Linear(d_model * mult, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x