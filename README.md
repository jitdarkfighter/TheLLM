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


Papers 
- Flash Attention
- Rope Encodings (RoFormer)
- Sliding window attention (LongFormer, Sparse Transformer)
    - Attention sink
- Mixture of Experts (Switch Transformer)
- Modern normalization like RMSNorm (RMSNorm, PreNorm vs PostNorm - On layer norm in the trans. arch)
- Gated FFN, SwiGLU (Switch Transformer)
- KV cache (Transformer-XL, ARTICLES)
- Reference: https://arxiv.org/abs/2002.05202 (GLU paper)
- Reference: https://arxiv.org/abs/1710.05941 (SiLU paper)

For later Papers
- LLaMA / LLaMA 2 → KV cache + RMSNorm
- Mistral → Sliding window + FlashAttention