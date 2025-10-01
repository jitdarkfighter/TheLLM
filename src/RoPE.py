# Attention pattern depends only on the relative positions of the tokens, not their absolute positions.
# k = Rm_theta * W_q * x -> key at pos. m
# q = Rn_theta * W_k * x -> query at pos. n
# attn = q @ k^T = (W_q * x) @ Rm_theta^T @ Rn_theta @ (W_k * x)^T  = (W_q * x) @ (W_k * x)^T @ R_(n-m)_theta
# R_(n-m)_theta can be computed optimally using cos and sin matrices.

# RoPECache -> Precompute the cos and sin matrices for a max sequence length and store them. Otherwise compute and memory will explode.