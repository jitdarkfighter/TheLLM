"""
Quick test script to verify model initialization and basic functionality
"""

import torch
import json

from src.JithFormer import JithFormer
from src.Model.tokenizer import ByteTokenizer
    
with open('config.json', 'r') as f:
    config = json.load(f)

model_config = config['model_config']

print("Initializing model")
model = JithFormer(**model_config)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Test forward pass
print("Testing forward pass")
batch_size = 2
seq_len = 32

# Create dummy input
dummy_input = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
dummy_target = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))

# Forward pass
with torch.no_grad():
    logits, loss, kv_cache = model(dummy_input, dummy_target)

print(f"Input shape: {dummy_input.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item():.4f}")

# Test generation
print("Testing generation")
tokenizer = ByteTokenizer()
prompt = "Hello"
prompt_tokens = tokenizer.encode(prompt).unsqueeze(0)

with torch.no_grad():
    generated = model.generate(
        prompt=prompt_tokens,
        max_new_tokens=10,
        temperature=1.0
    )

generated_text = tokenizer.decode(generated[0])
print(f"Prompt: '{prompt}'")
print(f"Generated: '{generated_text}'", flush=True)

