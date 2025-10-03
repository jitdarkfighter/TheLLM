import json
import torch

from src.JithFormer import JithFormer
from src.Model.tokenizer import ByteTokenizer

PROMPT = "In a village of La Mancha, the name of which I have no desire to call to mind, "

def load_model(checkpoint_path, config_path, device):
    print(f"Loading model from {checkpoint_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = JithFormer(**config['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    return model, config



def generate_text(model, tokenizer, prompt, generation_config, device):
    # Encode prompt
    if isinstance(prompt, str):
        prompt_tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
    else:
        prompt_tokens = prompt.to(device)
    
    # Check if prompt length exceeds block size
    prompt_length = prompt_tokens.shape[1]
    if prompt_length > model.block_size:
        raise ValueError(
            f"Prompt length ({prompt_length} tokens) exceeds model's block size ({model.block_size}). Use shorter prompt"
        )
    
    # Generate
    # Note: sliding_window and attention_sink are configured in model_config at model initialization
    with torch.no_grad():
        generated = model.generate(
            prompt=prompt_tokens,
            max_new_tokens=generation_config['max_new_tokens'],
            temperature=generation_config['temperature'],
            top_k=generation_config.get('top_k', 50),
            top_p=generation_config.get('top_p', 0.9),
            eos_token=generation_config.get('eos_token', None)
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0])
    prompt_text = tokenizer.decode(prompt_tokens[0])
    
    # Extract only the newly generated part
    new_text = generated_text[len(prompt_text):]

    return generated_text, new_text


def main():
    checkpoint_path = "models/jithformer_checkpoint.pt"
    config_path = "config.json"
    prompt = PROMPT 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    model, config = load_model(checkpoint_path, config_path, device)

    # Initialize tokenizer
    tokenizer = ByteTokenizer()
    
    generation_config = config['generation_config']
    
    full_text, new_text = generate_text(model, tokenizer, prompt, generation_config, device)
    print(f"Prompt: {repr(prompt)}")
    print(f"Generated: {repr(new_text)}")
    print(f"\nFull text:\n{full_text}")
   
if __name__ == "__main__":
    main()