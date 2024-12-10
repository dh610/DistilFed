import socket
import socket_utils
import torch
from model import get_minigpt_and_embedder, embed_to_transformer
from dataset import make_dataset_list_by_iter
from tqdm import tqdm
import torch.nn.functional as F


# TODO: When send tensor to server, must send attention_mask together!!!!
# params = torch.load('./ckpts/params_before_fed.pt')

def main(device, host='127.0.0.1', port=65432):
    model, embedder = get_minigpt_and_embedder()
    model.to(device)
    model.train()
    for param in model.lm_head.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    T = 2.0
    alpha = 0.25

    embedder.to(device)
    embedder.eval()

    file_num = 15
    client_num = 64
    end_data = 524288
    batch_size = 16
    round_num = 128

    round = 0
    client = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connecting to server {host}:{port}")
        print(f"========== Round {round+1} ==========")

        for i in range(file_num):
            dataloader = make_dataset_list_by_iter(i)
            bar = tqdm(enumerate(dataloader), total=len(dataloader))

            for batch_idx, batch in bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                embeds = embedder(input_ids).last_hidden_state
                output = embed_to_transformer(model, embeds, attention_mask=attention_mask, labels=input_ids)

                socket_utils.send_tensor(s, 100, embeds)
                socket_utils.send_tensor(s, 200, attention_mask)
                _, header2 = socket_utils.recv_header(s)
                data_length = header2
                # Receive LLM logits from server.
                teacher_logits = socket_utils.recv_tensor(s, data_length)

                student_loss = output.loss
                student_logits = output.logits

                # KD Loss
                teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                student_log_probs = F.log_softmax(student_logits / T, dim=-1)
                kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)

                total_loss = alpha * student_loss + (1 - alpha) * kd_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                bar.set_postfix(
                    round=round,
                    client=client,
                    ce_loss=student_loss.item(),
                    kd_loss=kd_loss.item(),
                    total_loss=total_loss.item()
                )

                if (batch_idx + 1) % 64 == 0:
                    # save dict and send to server
                    params = model.transformer.h.state_dict()
                    torch.save(params, f'./ckpts/params_round_{round}_{client}.pt')
                    socket_utils.send_tensor(s, client, params)
                    client += 1
                    if client == client_num:
                        client = 0
                        # Receive params
                        _, data_length = socket_utils.recv_header(s)
                        params = socket_utils.recv_tensor(s, data_length)

                        round += 1

                    model.transformer.h.load_state_dict(params)

                if batch_idx * batch_size >= end_data:
                    # Bring next dataset file
                    break

            if round == round_num:
                break

        while True:
            message = input("To server: ")
            if message.lower() == 'exit':
                print("Connection closed.")
                break
            s.sendall(message.encode())

            data = s.recv(1024)
            if not data:
                print("Server Closed.")
                break
            print(f"From server: {data.decode()}")


if __name__ == "__main__":
    device = torch.device("cuda:1")
    main(device)