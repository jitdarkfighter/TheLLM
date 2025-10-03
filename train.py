import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from huggingface_hub import login

from src.JithFormer import JithFormer
from src.Model.tokenizer import ByteTokenizer
from src.Model.amp import ampGrad
from src.Model.lr_scheduler import WarmupCosineScheduler

login(os.getenv("HUGGINGFACE_API_KEY"))

os.makedirs('models', exist_ok=True)

# Get configs
with open('config.json', 'r') as f:
    config = json.load(f)
model_config = config['model_config']
training_config = config['training_config']
paths_config = config['paths']
EPOCHS = 5 
BATCH_SIZE = training_config['batch_size']
LEARNING_RATE = training_config['learning_rate']
WEIGHT_DECAY = training_config['weight_decay']
WARMUP_STEPS = training_config['warmup_iters']
TOTAL_STEPS = training_config['max_iters']
BASE_LR = training_config['learning_rate']

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.tokens = []
        for text in texts:
            encoded_tokens = self.tokenizer.encode(text).tolist()  # Convert tensor to list
            self.tokens.extend(encoded_tokens + [tokenizer.eos_token_id])
        print(f"Dataset has {len(self.tokens)} tokens")
        
    def __len__(self):
        return max(1, len(self.tokens) - self.block_size)
    
    def __getitem__(self, idx):
        if len(self.tokens) <= self.block_size:
            # If text is too short, pad with zeros
            tokens = self.tokens + [0] * (self.block_size + 1 - len(self.tokens))
        else:
            tokens = self.tokens[idx:idx + self.block_size + 1]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y
    
def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys

tokenizer = ByteTokenizer()

# Use only a quarter of the dataset
train_texts = [text for text in dataset['train']['text'] if text.strip()]
val_texts = [text for text in dataset['validation']['text'] if text.strip()]

# Take only 1/100 of the training and validation data
quarter_train_size = len(train_texts) // 50
quarter_val_size = len(val_texts) // 50
train_texts = train_texts[:quarter_train_size]
val_texts = val_texts[:quarter_val_size]

print(f"Using {len(train_texts)} training texts and {len(val_texts)} validation texts (1/100 of original dataset)")

train_dataset = WikiTextDataset(train_texts, tokenizer, model_config['block_size'])
val_dataset = WikiTextDataset(val_texts, tokenizer, model_config['block_size'])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True ,num_workers=2, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

model = JithFormer(**model_config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = WarmupCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS, 
                                  base_lr=BASE_LR)
amp = ampGrad(optimizer=optimizer, accumulation_steps=1)


model.train()
best_val_loss = float('inf')
iter_num = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0
    num_batches = 0

    model.train() 
    for input, targets in train_dataloader:
        input, targets = input.to(device), targets.to(device)

        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            logits, loss, _ = model(input, targets)
        amp.backward(loss)
        
        if amp.should_step():
            amp.step()
            amp.zero_grad()
            lr = scheduler.step()

        iter_num += 1
        epoch_loss += loss.item()
        num_batches += 1

        if iter_num % 100 == 0:
            print(f"Iter {iter_num}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
        
    
    # End of epoch validation
    avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
    print(f"End of Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")
    
    # # Validation
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for input, targets in val_dataloader:
            input, targets = input.to(device), targets.to(device)
            logits, loss, _ = model(input, targets)
            val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Save best model

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': best_val_loss,
    }, paths_config['model_save_path'])
    print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
    

print("Training completed!")