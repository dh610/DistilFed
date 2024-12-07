import torch
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

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

if __name__ == "__main__":
    output_path = 'tokens'
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = load_dataset("bookcorpus/bookcorpus", split='train')

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    # Use batch
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    input_ids_list = []
    attention_mask_list = []

    for example in tqdm(tokenized_dataset):
        input_ids_list.append(example['input_ids'])
        attention_mask_list.append(example['attention_mask'])

    input_ids = torch.tensor(input_ids_list)
    attention_mask = torch.tensor(attention_mask_list)

    torch.save((input_ids, attention_mask), output_path)
    print(f"Tokenized dataset saved to {output_path}")