import os
import math
import torch
import numpy as np
from tqdm import tqdm
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader

num_proc = 8
data_dir = os.path.join('data', 'openwebtext')
batch_size = 32  # Adjust this as needed
block_size = 128  # Adjust this as needed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizer function
enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

# Download and split the dataset
dataset = load_dataset("openwebtext")
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

# Tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="Tokenizing the dataset",
    num_proc=num_proc,
)

# Store tokenized data into binary files
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(data_dir, f'{split}.bin')
    dtype = np.uint16 
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

# Load data
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

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
