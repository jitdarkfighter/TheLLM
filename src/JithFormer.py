import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Model.transformer_block import TransformerBlock
from src.Model.tokenizer import ByteTokenizer

class JithFormer(nn.Module):
    def __init__(self, vocab_size: int = 512, block_size: int = 256, n_head: int = 4, d_model: int = 512, dropout: float = 0.0,
                 max_pos: int = 4096, sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None,
                 n_layer = 6):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ TransformerBlock(d_model, n_head, dropout, max_pos, sliding_window, attention_sink, n_kv_head) for _ in range(n_layer)])
        self.ln_f = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, kv_cache_list = None, start_pos: int = 0):
        B, T = idx.shape
        assert T <= self.block_size, "Cannot forward, model block size is exhausted."
        
        # Token and positional embeddings
        tok_emb = self.token_embed(idx)  # (B,T,C)
        x = self.dropout(tok_emb)

        new_kv_cache_list = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_cache_list[i] if kv_cache_list is not None else None
            x, new_kv_cache = layer(x, kv_cache, start_pos)
            new_kv_cache_list.append(new_kv_cache)

        x = self.ln_f(x)  # (B,T,C)
        logits = self.head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            # reshape for loss computation
            logits_flat = logits.view(-1, logits.size(-1))  # (B*T, vocab_size)
            targets_flat = targets.view(-1)  # (B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, new_kv_cache_list

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0, top_k: int = 10,
                 top_p: float | None = None, eos_token: int = 1, sliding_window: int | None = None, attention_sink: int = 0):
        

        self.eval()
        idx = prompt
        kv_cache_list = [None] * len(self.layers)  # Initialize kv_cache for each layer
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  if kv_cache_list[0] is None else idx[:, -1:]  # crop context if needed

            # absolute start position from cache length(0 on the first step)
            start_pos = 0 if kv_cache_list[0] is None else kv_cache_list[0].shape[2]

            logits, _, kv_cache_list = self(idx = idx_cond, targets = None, kv_cache_list = kv_cache_list, start_pos = start_pos)

            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)  # (B, vocab_size)
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_id), dim=1)  # append to the sequence

            if eos_token is not None and next_id.item() == eos_token:
                break
        
        return idx