import torch.nn as nn

# SwiGLU(x)=(xW1​)⊙SiLU(xW2​), Combines GLU and SiLU (Swish) activations
# SwiGLU is found to perform better than ReLU and GELU in some transformer architectures
# SiLU acts as the gate. 0 for negative inputs, linear for positive inputs. So this introduces learnable gating. (Only imp features are passed through)
# Can express complex patterns without extra layers, eg if feat. A is strong supress B or if feat. C is strong enhance D. This would require more layers in GELU.
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = dim * hidden_dim_factor
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()  # Swish activation, x * sigmoid(x)

    def forward(self, x):
        a = self.fc1(x)
        b = self.act(self.fc2(x))
        x = a * b  # Element-wise multiplication
        x = self.dropout(self.fc3(x))
        return x