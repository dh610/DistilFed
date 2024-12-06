import sys
from model import call_gpt2_model
import os, torch
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from utils import calculate_perplexity
from dataset import TokenizedDataset

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

def main():
    llm, fedlm = call_gpt2_model(device)
    data_dir = "./tokens"
    dataset = TokenizedDataset(data_dir)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=False)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=16, pin_memory=True, num_workers=4)

    llm.eval()
    fedlm.eval()

    data_iter = tqdm(data_loader, desc='Processing dataset', unit='example') if local_rank == 0 else data_loader

    llm_perplexity = calculate_perplexity(data_iter, llm, device)
    fedlm_perplexity = calculate_perplexity(data_iter, fedlm, device)

    # Convert perplexity to tensor for distributed reduction
    llm_tensor = torch.tensor(llm_perplexity, device=device)
    fedlm_tensor = torch.tensor(fedlm_perplexity, device=device)

    # Reduce perplexity values to GPU 0
    dist.reduce(llm_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(fedlm_tensor, dst=0, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        # Calculate average perplexity
        num_gpus = dist.get_world_size()
        llm_avg_ppl = llm_tensor.item() / num_gpus
        fedlm_avg_ppl = fedlm_tensor.item() / num_gpus

        # Print final perplexity
        print(f"\nFinal Perplexity Results:")
        print(f"LLM Average Perplexity: {llm_avg_ppl:.4f}")
        print(f"FedLM Average Perplexity: {fedlm_avg_ppl:.4f}")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    if device.type != 'cuda':
        print('Cannot use CUDA')
        sys.exit(1)

    main()