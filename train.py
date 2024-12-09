import sys, os, torch, model
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from dataset import TokenizedDataset
import torch.nn.functional as F
import torch.nn as nn

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

def main():
    # Model setting
    llm, fedlm = model.call_gpt2_model(device)
    fedlm.load_state_dict(torch.load('./ckpts/fedlm_distilled_epoch1.pt'))
    fedlm = nn.parallel.DistributedDataParallel(
        fedlm,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    # Freezing embedding and lmhead params
    for param in fedlm.module.transformer.wte.parameters():
        param.requires_grad = False

    for param in fedlm.module.transformer.wpe.parameters():
        param.requires_grad = False

    for param in fedlm.module.lm_head.parameters():
        param.requires_grad = False

    # Dataset setting
    data_dir = './tokens'
    dataset = TokenizedDataset(data_dir)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=False)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=32, pin_memory=True, num_workers=4)

    llm.eval()
    fedlm.train()

    optimizer = torch.optim.AdamW(fedlm.parameters(), lr=1e-4)

    T = 1.0
    alpha = 0.5
    num_epochs = 2

    for epoch in range(num_epochs):
        data_iter = tqdm(data_loader, unit='batch', desc=f"Epoch [{epoch + 1}/{num_epochs}]", disable=(local_rank != 0))

        for batch_idx, batch in enumerate(data_iter):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Teacher model inference (no_grad)
            with torch.no_grad():
                teacher_outputs = llm(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            # Student model inference
            student_outputs = fedlm(input_ids, attention_mask=attention_mask, labels=input_ids)
            student_loss = student_outputs.loss  # CE Loss
            student_logits = student_outputs.logits

            # KD Loss
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)

            total_loss = alpha * student_loss + (1 - alpha) * kd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if local_rank == 0:
                data_iter.set_postfix(ce_loss=student_loss.item(), kd_loss=kd_loss.item(), total_loss=total_loss.item())

        if local_rank == 0:
            torch.save(fedlm.module.state_dict(), f"ckpts/fedlm_distilled_epoch{epoch + 1}.pt")
            print(f"Model saved to ckpts/fedlm_distilled_epoch{epoch + 1}.pt")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    if device.type != 'cuda':
        print('Cannot use CUDA')
        sys.exit(1)
    main()
