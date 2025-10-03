import torch
# use tokenizer library. They are implemented in rust and is very fast.
from tokenizers import ByteLevelBPETokenizer, Tokenizer

class ByteTokenizer:
    def __init__(self):
        self.eos_token_id = 1
    
    def encode(self, str) -> torch.Tensor:
        return torch.tensor(list(str.encode('utf-8')), dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return bytes(tokens).decode('utf-8', errors='ignore')
    
    def vocab_size(self) -> int:
        return 256