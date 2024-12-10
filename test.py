import sys
from model import call_gpt2_model
import os, torch
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from utils import calculate_perplexity
from dataset import TokenizedDataset
import pandas as pd

'''
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
'''

def main(device):
    llm, fedlm = call_gpt2_model()
    data_dir = "./tokens/validation_dataset.pt"
    dataset = TokenizedDataset(data_dir)
    data_loader = DataLoader(dataset, batch_size=64, pin_memory=True, num_workers=4)

    llm.eval()
    fedlm.eval()

    fedlm.to(device)

    fedlm.transformer.h.load_state_dict(torch.load('./ckpts/params_round_59_60.pt'))
    perplexity = calculate_perplexity(data_loader, fedlm, device)
    print(perplexity)
    sys.exit(0)

    num_clients = 128
    num_rounds = 60
    df = pd.DataFrame(index=[f"Round {i + 1}" for i in range(num_rounds)],
                      columns=[f"Client {j + 1}" for j in range(num_clients)] + ["FedAvg"])

    for round in range(num_rounds):
        for client in range(num_clients):
            ckpt_path = f'./ckpts/params_round_{round}_{client}.pt'
            params = torch.load(ckpt_path)
            fedlm.transformer.h.load_state_dict(params)

            with torch.no_grad():
                data_iter = tqdm(data_loader, desc=f'Round: {round}, Client: {client}', unit='example')
                perplexity = calculate_perplexity(data_iter, fedlm, device)

            print(f'round {round}, client {client}, perplexity: {perplexity}')
            df.iloc[round, client] = perplexity

        ckpt_path = f'./ckpts/params_round_{round}.pt'
        params = torch.load(ckpt_path)
        fedlm.transformer.h.load_state_dict(params)

        with torch.no_grad():
            data_iter = tqdm(data_loader, desc=f'Round: {round}', unit='example')
            perplexity = calculate_perplexity(data_iter, fedlm, device)

        print(f"round {round}, perplexity: {perplexity}")
        df.iloc[round, 128] = perplexity

    output_file = "perplexity_results_with_fedavg.xlsx"
    df.to_excel(output_file)
    print(f"Perplexity results saved to {output_file}")


'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
if __name__ == '__main__':
    '''
    if device.type != 'cuda':
        print('Cannot use CUDA')
        sys.exit(1)
    '''
    device = torch.device("cuda:0")

    main(device)