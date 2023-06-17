import logging
import torch
import torch.optim as optim
import torch.nn as nn

from model import RotationallyInvariantGPT, RotationallyInvariantGPTConfig
from utils import *
from nanoGPT.model import GPTConfig, GPT, MLP
from datasets import load_from_disk
from torch.utils.data import DataLoader

class TokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        ids = torch.tensor(item["ids"], dtype=torch.long)
        return ids[:-1], ids[1:]

# Training loop
def train(model: nn.Module, optimizer: optim.Optimizer, train_loader) -> float:
    model.train()
    running_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            logging.info(f"Batch {i}: Loss={loss.item()}")
    return running_loss / len(train_loader)

def evaluate(model, valid_loader) -> float:
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, loss = model(inputs, targets)
            running_loss += loss.item()
            if i % 100 == 0:
                logging.info(f"Batch {i}: Validation Loss={loss.item()}")
    return running_loss / len(valid_loader)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Load the tokenized dataset from disk
    tokenized_train = load_from_disk("train_dataset")
    tokenized_val = load_from_disk("val_dataset")

    # Create train/val dataset objects
    train_dataset = TokenizedTextDataset(tokenized_train)
    valid_dataset = TokenizedTextDataset(tokenized_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False)

    vocab_size = 50257  # GPT-2 tokenizer vocab size
    logging.info(f"Vocab size: {vocab_size}")
    
    # Configs
    d_model = 512
    num_heads = 8
    num_layers = 6
    block_size = 512
    dropout = 0.2
    bias = True
    batch_size = 128
    eval_batch_size = 256
    epochs = 10
    lr = 0.001

    logging.info(f"Config: d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, block_size={block_size}, dropout={dropout}, bias={bias}")
    logging.info(f"Training for {epochs} epochs with a learning rate of {lr}...")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Eval batch size: {eval_batch_size}")
    
    # Calculate the number of batches
    num_train_batches = len(train_dataset) // batch_size
    num_eval_batches = len(valid_dataset) // eval_batch_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    gpt_config = GPTConfig(vocab_size=vocab_size, n_embd=d_model, n_head=num_heads, n_layer=num_layers, block_size=block_size, dropout=dropout, bias=bias)
    rigpt_config = RotationallyInvariantGPTConfig(vocab_size=vocab_size, n_embd=d_model, n_head=num_heads, n_layer=num_layers, block_size=block_size, dropout=dropout, bias=bias)
    gpt = GPT(gpt_config).to(device)
    rigpt = RotationallyInvariantGPT(rigpt_config).to(device)

    optimizer_gpt = optim.Adam(gpt.parameters(), lr=lr)
    optimizer_rigpt = optim.Adam(rigpt.parameters(), lr=lr)

    for model, optimizer, model_name in [(gpt, optimizer_gpt, 'GPT'), (rigpt, optimizer_rigpt, 'RotationallyInvariantGPT')]:
        print(f"Training {model_name}")
        for epoch in range(1, epochs + 1):
            print(f"Training epoch {epoch}")
            train_loss = train(model, optimizer, num_train_batches)
            print(f"Validating epoch {epoch}")
            valid_loss = evaluate(model, num_eval_batches)
            print(f'{model_name} - Epoch: {epoch}, Train loss: {train_loss:.3f}, Validation loss: {valid_loss:.3f}')

    torch.save(gpt.state_dict(), 'gpt.pt')
    torch.save(rigpt.state_dict(), 'rigpt.pt')
