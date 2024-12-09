import torch
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
import os

class TokenizedDataset(Dataset):
    def __init__(self, input_dir):
        # Dataset type: (input_ids, attention_mask)
        for file_path in input_dir:
            data = torch.load(file_path)
        self.input_ids = data[0]          # [num_samples, seq_len]
        self.attention_mask = data[1]     # [num_samples, seq_len]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

if __name__ == "__main__":

    output_path = './fedtokens'

    print(f"Tokenized dataset will be saved at {output_path}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('bookcorpus/bookcorpus', split='train')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    batch_size = 5_000_000
    input_ids_list = []
    attention_mask_list = []

    for i, example in enumerate(tqdm(tokenized_dataset)):
        input_ids_list.append(example['input_ids'])
        attention_mask_list.append(example['attention_mask'])

        if (i + 1) % batch_size == 0 or (i + 1) == len(tokenized_dataset):
            batch_idx = i // batch_size
            input_ids = torch.tensor(input_ids_list)
            attention_mask = torch.tensor(attention_mask_list)
            save_path = os.path.join(output_path, f'tokenized_dataset_{batch_idx}.pt')
            torch.save((input_ids, attention_mask), save_path)
            print(f"Tokenized dataset saved in batch to {output_path}")

            input_ids_list = []
            attention_mask_list = []

