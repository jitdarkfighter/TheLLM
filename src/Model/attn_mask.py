import torch

def causal_mask(T: int, device=None):
    """
    So using a boolean matrix, it sets the values in the actual matrix as -inf (which will become 0 because of softmax) wherever the boolean is True.
    """
    m = torch.triu(torch.ones((T,T), dtype = torch.bool, device = device), diagonal = 1)
    return m.view(1, 1, T, T)