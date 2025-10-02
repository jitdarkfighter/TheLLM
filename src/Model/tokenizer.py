import torch

class ByteTokenizer:
    def encode(self, str) -> torch.Tensor:
        return torch.tensor(list(str.encode('utf-8')), dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return bytes(tokens).decode('utf-8', errors='ignore')
    
    def vocab_size(self) -> int:
        return 256