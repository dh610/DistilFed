import os, torch, model
from tqdm import tqdm
from collections import OrderedDict

from dataset import make_dataset_list_by_iter
import torch.nn.functional as F

'''
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
'''

def main(device):
    # Model setting
    gpt2 = model.get_gpt2_wo_embedder()
    fedlm, embedder = model.get_minigpt_and_embedder()
    fedlm.transformer.h.load_state_dict(torch.load('./ckpts/before_fed.pt'))

    gpt2.to(device)
    fedlm.to(device)
    embedder.to(device)

    gpt2.eval()
    fedlm.train()
    embedder.eval()

    for param in fedlm.lm_head.parameters():
        param.requires_grad = False

    # Dataset setting

    optimizer = torch.optim.AdamW(fedlm.parameters(), lr=1e-4)

    file_num = 15
    client_num = 128
    end_data = 524288
    batch_size = 64
    round_num = 128

    round = 0
    client = 0

    T = 1.0
    alpha = 0.5

    for i in range(file_num):
        dataloader = make_dataset_list_by_iter(i)
        length = end_data // batch_size
        bar = tqdm(enumerate(dataloader), total=length)

        parameters = [None] * client_num

        for batch_idx, batch in bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                embeds = embedder(input_ids).last_hidden_state
                # 원래는 이 과정이 서버를 거치고 돌아와야 하는 과정임.
                teacher_logits = model.embed_to_transformer(gpt2, embeds, attention_mask=attention_mask).logits

            student_outputs = model.embed_to_transformer(fedlm, embeds, attention_mask=attention_mask, labels=input_ids)
            student_loss = student_outputs.loss
            student_logits = student_outputs.logits

            # KD Loss
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)

            total_loss = alpha * student_loss + (1 - alpha) * kd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            bar.set_postfix(
                state=f'{round}-{client}',
                ce_loss=student_loss.item(),
                kd_loss=kd_loss.item(),
                total_loss=total_loss.item()
            )
            if (batch_idx + 1) % 16 == 0:
                parameters[client] = fedlm.transformer.h.state_dict()
                torch.save(parameters[client], f'./ckpts/params_round_{round}_{client}.pt')
                client += 1
                if client == client_num:
                    client = 0
                    #fedavg calculate
                    fedparam = average_state_dicts(parameters)
                    torch.save(fedparam, f'./ckpts/params_round_{round}.pt')
                    parameters = [fedparam] * client_num

                    round += 1

                next_param = parameters[client]
                if next_param is not None:
                    fedlm.transformer.h.load_state_dict(parameters[client])

            if batch_idx * batch_size >= end_data:
                break

        if round == round_num:
            break


def average_state_dicts(state_dicts):
    avg_state_dict = OrderedDict()
    for key in state_dicts[0].keys():
        avg_state_dict[key] = torch.stack([sd[key] for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state_dict


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    device = torch.device("cuda:0")
    main(device)
