import torch

def calculate_perplexity(data_loader, model, device):
    total_loss = 0.0
    total_tokens = 0

    for batch in data_loader:
        input_ids = batch.to(device)  # input_ids

        for i in range(1, input_ids.size(1)):
            input_context = input_ids[:, :i]
            target_token = input_ids[:, i]

            with torch.no_grad():
                outputs = model(input_context, target_token)
                loss = outputs.loss

            total_loss += loss.sum().item()
            total_tokens += loss.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def save_checkpoint(checkpoint_path, model):
    torch.save(model.state_dict(), checkpoint_path)

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)