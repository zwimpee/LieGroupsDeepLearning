import os
import multiprocessing
import pickle
import torch
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from utils import process

def main():
    multiprocessing.freeze_support()

    data_dir = os.path.join('data', 'openwebtext')
    cache_dir = 'C:/Users/User/.cache/huggingface/datasets/openwebtext/plain_text/1.0.0/6f68e85c16ccc770c0dd489f4008852ea9633604995addd0cd76e293aed9e521'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Download the OpenWebText dataset
    dataset = load_dataset("openwebtext", cache_dir=cache_dir)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)
    split_dataset['val'] = split_dataset['test']

    # Tokenize the dataset
    def process_with_tokenizer(example):
        return process(example, tokenizer)

    tokenized = split_dataset.map(
        process_with_tokenizer,
        remove_columns=['text'],
        desc="Tokenizing the dataset",
        num_proc=8,
    )

    # Move the tokenized dataset tensors to the GPU
    tokenized["input_ids"] = tokenized["input_ids"].to(device)
    tokenized["attention_mask"] = tokenized["attention_mask"].to(device)

    # Save the tokenized dataset to file
    with open("tokenized_dataset.pickle", "wb") as f:
        pickle.dump(tokenized, f)

    # Save train and val datasets to disk
    tokenized['train'].save_to_disk('train_dataset')
    tokenized['val'].save_to_disk('val_dataset')

if __name__ == '__main__':
    main()
