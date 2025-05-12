import os
import torch
from torch.utils.data import Dataset, DataLoader
from mingpt.model import GPT
import numpy as np
import wandb
from tqdm import tqdm
from datasets import load_dataset

# --- Tokenizer class ---
from mingpt.bpe_new import BPETokenizerReasoning

# --- Dataset class with packing instead of padding ---
class PackedReasoningDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        # We'll use block_size - 1 for packing to ensure we have exact lengths after shifting
        self.pack_size = block_size - 1
        
        print("Tokenizing and packing data...")
        # First concatenate all texts with a separator token (using EOS token if available)
        separator = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 0
        
        # Tokenize all texts
        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing texts"):
            tokens = tokenizer(text)
            all_tokens.append(tokens)
            # Add separator token
            all_tokens.append(torch.tensor([separator]))
        
        # Concatenate all tokens
        all_tokens = torch.cat(all_tokens)
        
        # Pack into fixed-size chunks
        self.packed_data = []
        for i in range(0, len(all_tokens) - self.pack_size, self.pack_size): # Changed the upper bound
            chunk = all_tokens[i:i + self.pack_size + 1]
            if len(chunk) == self.pack_size + 1: # Ensure exact length
                self.packed_data.append(chunk)

        print(f"Created {len(self.packed_data)} packed examples of size {self.pack_size}")

    def __len__(self):
        return len(self.packed_data)

    def __getitem__(self, idx):
        chunk = self.packed_data[idx]
        x = chunk[:-1]  # Input: all but last token
        y = chunk[1:]   # Target: all but first token
        
        return {'x': x, 'y': y}

def collate_batch_packed(batch):
    # All sequences should be exactly the same length now
    x = torch.stack([item['x'] for item in batch])
    y = torch.stack([item['y'] for item in batch])
    return {'x': x, 'y': y}

# --- WandB Init ---
wandb.init(project='reasoning-gpt', name='bpe-reasoning-packed-run')

# --- Config ---
train_config = {
    'learning_rate': 3e-4,
    'batch_size': 256,
    'max_iters': 50,
    'num_workers': 4
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Tokenizer and Dataset ---
tokenizer = BPETokenizerReasoning()

# load train, val, test from Pavankalyan/synreasoning huggingface dataset
dataset = load_dataset("Pavankalyan/syn-reasoning")
# Pick all texts which have value except 4 for the column 'length'
pretrain_dataset = dataset['pretrain']#.filter(lambda x: x['length'] != 4)
train_texts = pretrain_dataset['text']
val_texts = dataset['val']['text']
test_texts = dataset['test']['text']

block_size = 300

# --- GPT Model ---
model_config = GPT.get_default_config()
model_config.model_type = 'gpt-mini'
model_config.vocab_size = tokenizer.vocab_size
model_config.block_size = block_size
model = GPT(model_config).to(device)

train_dataset = PackedReasoningDataset(train_texts, tokenizer, block_size)
val_dataset = PackedReasoningDataset(val_texts, tokenizer, block_size)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=train_config['batch_size'],
    shuffle=True,
    num_workers=train_config['num_workers'],
    collate_fn=collate_batch_packed
)

val_loader = DataLoader(
    val_dataset,
    batch_size=train_config['batch_size'],
    shuffle=False,
    num_workers=train_config['num_workers'],
    collate_fn=collate_batch_packed
)

# --- Optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'])

# --- Loss ---
loss_fn = torch.nn.CrossEntropyLoss()

print("Starting training...")

# --- Training Loop ---
for epoch in range(train_config['max_iters']):
    model.train()
    total_train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_config['max_iters']} (Training)"):
        x = batch['x'].to(device)  # (B, T)
        y = batch['y'].to(device)  # (B, T)

        logits, _ = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        wandb.log({'train_loss': loss.item(), 'epoch': epoch + 1})

    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{train_config['max_iters']} (Validation)"):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            logits, _ = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}: train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}")
    wandb.log({'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'epoch': epoch + 1})

    # --- Save model checkpoint ---
    if (epoch + 1) % 10 == 0:
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/packed_non4_epoch_{epoch + 1}.pth')
        wandb.save(f'checkpoints/packed_non4_epoch_{epoch + 1}.pth')

print("Training complete!")