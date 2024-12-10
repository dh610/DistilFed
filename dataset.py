import torch
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
from tqdm import tqdm
import os

class TokenizedDataset(Dataset):
    def __init__(self, file_path):
        # Dataset type: (input_ids, attention_mask)
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

def make_dataset_list_by_iter(file_num, batch_size=16, world_size=None, rank=None):
    file_path = f"./tokens/tokenized_dataset_{file_num}.pt"
    dataset = TokenizedDataset(file_path)

    # DistributedSampler로 데이터셋을 GPU별로 나누기
    if world_size is not None:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

        # DataLoader 생성
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # DistributedSampler 적용
            pin_memory=True,
            num_workers=4
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4
        )
    return dataloader

if __name__ == "__main__":

    output_path = './tokens'

    print(f"Tokenized dataset will be saved at {output_path}")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('allenai/c4', 'realnewslike', split='validation')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    batch_size = 700000
    input_ids_list = []
    attention_mask_list = []

    for i, example in enumerate(tqdm(tokenized_dataset)):
        input_ids_list.append(example['input_ids'])
        attention_mask_list.append(example['attention_mask'])

        if (i + 1) % batch_size == 0 or (i + 1) == len(tokenized_dataset):
            batch_idx = i // batch_size
            input_ids = torch.tensor(input_ids_list)
            attention_mask = torch.tensor(attention_mask_list)
            save_path = os.path.join(output_path, f'validation_dataset.pt')
            torch.save((input_ids, attention_mask), save_path)
            print(f"Tokenized dataset saved in batch to {save_path}")

            input_ids_list = []
            attention_mask_list = []

