# TheLLM - JithFormer

A modern transformer language model implementation featuring advanced attention mechanisms and optimization techniques.

**Current Status:**<br>
Total parameters: 590,336<br>
Trained on 1.1M tokens from WikiText-2

- Model was trained for a block size of 256. So the code isn't adapted for taking input or generating long sequence. Expect degradation after the 256th token.

## Features
- RoPE (Rotary Position Embeddings)
- Sliding Window Attention with Attention Sink
- KV Cache for efficient inference
- RMSNorm & SwiGLU activation
- Grouped Query Attention (GQA) support
- Automatic Mixed Precision training
- Config-driven architecture

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file with your HuggingFace API key:
```
HUGGINGFACE_API_KEY=your_key_here
```

## Usage

### Training
Set number of epochs and dataset. (Preferably large datasets and low epochs)
```bash
python train.py
```

### Inference
Add your prompt to the variable `PROMPT` in the begining of the script and run the script.
```bash
python inference.py
```

### Test Model
To see model configs.
```bash
python test_model.py
```

##### Model config
```json
"model_config": {
    "vocab_size": 256,
    "block_size": 256,
    "n_head": 4,
    "d_model": 128,
    "dropout": 0.1,
    "max_pos": 4096,
    "sliding_window": null, //Inference only. Set to None automatically in training script. Only useful for long-context. ToDo 
    "attention_sink": 0,    //Inference only. Set to 0 automatically in training script. Only useful for long-context. ToDo
    "n_kv_head": null,      //If using a large model with many attention heads, use GQA
    "n_layer": 2
  }
```


#### Papers 
- https://arxiv.org/abs/2205.14135 - Flash Attention
- https://arxiv.org/abs/2104.09864 - Rope Encodings (RoFormer)
- Sliding window attention
    - https://arxiv.org/abs/2004.05150 - LongFormer
    - https://arxiv.org/abs/1904.10509 - Sparse Transformer
- https://arxiv.org/abs/2309.17453 - Attention sink
- https://arxiv.org/abs/1910.07467 - Modern normalization like RMSNorm (RMSNorm, PreNorm vs PostNorm - On layer norm in the trans. arch)
- https://arxiv.org/abs/2101.03961 - Gated FFN, SwiGLU (Switch Transformer)
- https://arxiv.org/abs/1901.02860 - KV cache (Transformer-XL, ARTICLES) - Inference only. (Sliding KV cache.)
- https://arxiv.org/abs/2002.05202 - GLU paper
- https://arxiv.org/abs/1710.05941 - SiLU paper
- https://docs.pytorch.org/docs/stable/amp.html - Automatic Mixed Precision

<br><br>

To-Do:
- When saving the model save the configs as well, so no errors incase configs had been tweaked.
- top_p and top_k probabilistic sampling.
- Benchmarks
- New model arch
    - Hyperparameter tuning.
    - Implement Modern Tokenizer.
    - Train model with a larger block_size. Model needs to handle larger sequences to make use of sliding window attention and attention sink. 
    - Increase model parameters 5M or 25M params on a 25M or 100M token dataset. with Batch_size 32 in P100 GPU for 1 Epoch. (10hrs - 15hrs)
- RL

<br>

Do feel free to reach out if you find any mistakes 😊