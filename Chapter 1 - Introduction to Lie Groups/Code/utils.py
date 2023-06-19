import os
import pickle
import math
import torch
import numpy as np
from tqdm import tqdm
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader
import multiprocessing as mp

def process(example, tokenizer):
    ids = tokenizer.encode(example['text'], max_length=1024, truncation=True)
    ids.append(tokenizer.eos_token_id)  # Appending the End of Document token
    return {'ids': ids, 'len': len(ids)}


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device.type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def new_rielu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))