import torch
import math

def calculate_perplexity(data_loader, model, device):
    total_loss = 0.0
    total_tokens = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        batch_size, seq_len = input_ids.size()
        total_loss += loss.item() * batch_size * seq_len
        total_tokens += batch_size * seq_len

    average_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(average_loss) if average_loss < float('inf') else float('inf')

    return perplexity

def save_checkpoint(checkpoint_path, model):
    torch.save(model.state_dict(), checkpoint_path)

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)