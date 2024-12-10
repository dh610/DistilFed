import socket
import socket_utils
from model import get_gpt2_wo_embedder, embed_to_transformer
import torch


token_len = 128
client_cnt = 15

def main(device, host='127.0.0.1', port=65432):
    model = get_gpt2_wo_embedder()
    model.to(device)
    model.eval()

    params = [] * client_cnt
    round = 0
    print(f'Round {round}')

    # Socket connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"server waiting at {host}:{port}...")

        conn, addr = s.accept()
        with conn:
            print(f"Connection Success: {addr}")
            while round < client_cnt:
                header1, header2 = socket_utils.recv_header(s)

                if header1 < 100:
                    # Send parameters from client.
                    client_no = header1
                    data_length = header2
                    params[client_no] = socket_utils.recv_tensor(conn, data_length)
                    if client_no == client_cnt - 1:
                        # Last client of this round
                        round += 1
                        print(f'Round {round}')
                        params = fedavg(params)
                        # save parameter
                        torch.save(params, f'./ckpts/params_round_{round}.pt')
                        head = 0
                        tensor = params[0]
                    else:
                        continue

                elif header1 == 100:
                    # header2 is tensor length.
                    data_length = header2
                    embeds = socket_utils.recv_tensor(conn, data_length).to(device)
                    attention_mask = socket_utils.recv_tensor(conn, token_len).to(device)

                    with torch.no_grad():
                        logits = embed_to_transformer(model, embeds, attention_mask=attention_mask).logits

                    head = 100
                    tensor = logits

                socket_utils.send_tensor(conn, head, tensor)

                data = conn.recv(1024)
                if not data:
                    break
                print(f"from client: {data.decode()}")
                response = input("to client: ")
                conn.sendall(response.encode())
    print("Shutdown Server...")


def fedavg(tensors):
    stacked_tensors = torch.stack(tensors)
    return [torch.mean(stacked_tensors, dim=0)] * client_cnt

if __name__ == "__main__":
    device = torch.device("cuda:0")
    main(device)
