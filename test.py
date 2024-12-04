import sys
from model import call_gpt2_model
import os, torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

def calculate_accuracy(model, input_ids, is_huggingface=False):
    total_tokens = input_ids.size(1) - 1
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for i in range(total_tokens):
            input_sequence = input_ids[:,:i+1]
            target_token = input_ids[:,i+1].item()

            logits = model(input_sequence).logits if is_huggingface else model(input_sequence)

            last_token_logits = logits[:,-1, :]
            probs = F.softmax(last_token_logits, dim=-1)
            top5_probs, top5_indices = torch.topk(probs, 5)

            if top5_indices[0, 0].item() == target_token:
                correct_top1 += 1

            if target_token in top5_indices[0].tolist():
                correct_top5 += 1

    return correct_top1 / total_tokens, correct_top5 / total_tokens

def main():
    tokenizer, llm, fedlm = call_gpt2_model(device)
    dataset = load_dataset('cam-cst/cbt', 'CN', split='test')
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=True)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=16, pin_memory=True, num_workers=4)

    llm.eval()
    fedlm.eval()

    fed_result = {"top1_acc": 0, "top5_acc": 0, "file_count": 0}
    llm_result = {"top1_acc": 0, "top5_acc": 0, "file_count": 0}

    data_iter = tqdm(data_loader, desc='Processing dataset', unit='example') if local_rank == 0 else data_loader

    for batch_idx, example in enumerate(data_iter):
        sentences = example['sentences']
        question = example['question']

        flat_sentences = [" ".join(inner) if isinstance(inner, list) else inner for inner in sentences]
        flat_question = " ".join(question) if isinstance(question, list) else question
        text = " ".join(flat_sentences) + " " + flat_question  # Add corresponding question

        start_idx = 0
        chunk_size = 1024

        while start_idx < len(text):
            tokens = tokenizer(
                text[start_idx:],
                return_tensors='pt',
                truncation=True,
                max_length=chunk_size,
                return_offsets_mapping=True
            )
            input_ids = tokens['input_ids'].to(device)

            offset_mapping = tokens['offset_mapping'].squeeze(0)
            slice = -2 if offset_mapping.size(0) > 1 else -1
            last_token_end = offset_mapping[slice, 1].item()  # 마지막 유효 토큰의 끝 위치
            pre_idx = start_idx
            start_idx += last_token_end  # 슬라이싱 위치 갱신

            if input_ids.size(1) < 2:
                continue  # 너무 짧은 입력은 스킵

            fed_top1_acc, fed_top5_acc = calculate_accuracy(fedlm, input_ids)
            fed_result["top1_acc"] += fed_top1_acc
            fed_result["top5_acc"] += fed_top5_acc

            llm_top1_acc, llm_top5_acc = calculate_accuracy(llm, input_ids, is_huggingface=True)
            llm_result["top1_acc"] += llm_top1_acc
            llm_result["top5_acc"] += llm_top5_acc

    # Convert results to tensors for distributed reduction
    fed_tensor = torch.tensor([fed_result["top1_acc"], fed_result["top5_acc"], fed_result["file_count"]], device=device)
    llm_tensor = torch.tensor([llm_result["top1_acc"], llm_result["top5_acc"], llm_result["file_count"]], device=device)

    # Reduce results to rank 0
    dist.reduce(fed_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(llm_tensor, dst=0, op=dist.ReduceOp.SUM)

    # Rank 0 computes and prints the final results
    if local_rank == 0:
        # Calculate average accuracy
        fed_top1_acc_avg = fed_tensor[0].item() / fed_tensor[2].item()
        fed_top5_acc_avg = fed_tensor[1].item() / fed_tensor[2].item()
        llm_top1_acc_avg = llm_tensor[0].item() / llm_tensor[2].item()
        llm_top5_acc_avg = llm_tensor[1].item() / llm_tensor[2].item()

        # Print final results
        print("\nFinal Results:")
        print(f"FedLM - Top-1 Accuracy: {fed_top1_acc_avg:.4f}, Top-5 Accuracy: {fed_top5_acc_avg:.4f}")
        print(f"LLM   - Top-1 Accuracy: {llm_top1_acc_avg:.4f}, Top-5 Accuracy: {llm_top5_acc_avg:.4f}")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    if device.type != 'cuda':
        print('Cannot use CUDA')
        sys.exit(1)
    main()