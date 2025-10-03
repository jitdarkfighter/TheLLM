# TheLLM

Aim
- Flash Attention Inference
- RoPE encodings
- KV cache and Rolling buffer KV cache for streaming
- Sliding window attention and attention sink
- RMSNorm
- MoE
- SFT
- Reward Modeling
- RLHF with PPO or GRPO
- Have good evaluation benchmarks
- Inference Efficency (For later, QloRA and stuff)
- Automatic Mixed Precision - https://docs.pytorch.org/docs/stable/amp.html

Todo
- Implement a better tokenizer
- Check configs
- Get more data and train on kaggle. it is better to have x20 number of parameters of tokens.
- Check if inference is working properly
- Check training script

Something is wrong with 0 temperature
Generation config: {'max_new_tokens': 100, 'temperature': 0.0, 'top_k': 10, 'top_p': None, 'eos_token': 1}

Prompt: 'movement limited by their Action Gauge . Up to nine'
Generating...

==================================================
Prompt: 'movement limited by their Action Gauge . Up to nine'
Generated: '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'


Papers 
- Flash Attention
- Rope Encodings (RoFormer)
- Sliding window attention (LongFormer, Sparse Transformer)
    - Attention sink
- Mixture of Experts (Switch Transformer)
- Modern normalization like RMSNorm (RMSNorm, PreNorm vs PostNorm - On layer norm in the trans. arch)
- Gated FFN, SwiGLU (Switch Transformer)
- KV cache (Transformer-XL, ARTICLES) - Inference only. (Sliding KV cache.)
- Reference: https://arxiv.org/abs/2002.05202 (GLU paper)
- Reference: https://arxiv.org/abs/1710.05941 (SiLU paper)

For later Papers
- LLaMA / LLaMA 2 → KV cache + RMSNorm
- Mistral → Sliding window + FlashAttention


Before going public
- Support checkpointing / resume training.
- Support mixed-precision / AMP for GPU efficiency.
- Add config-driven architecture → easily switch #layers, #heads, d_model, etc.