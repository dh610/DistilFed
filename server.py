import socket
from model import get_gpt2_wo_embedder
import torch


def main(device, host='127.0.0.1', port=65432):
    model = get_gpt2_wo_embedder()
    model.to(device)

    # Socket connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"server waiting at {host}:{port}...")

        conn, addr = s.accept()
        with conn:
            print(f"Connection Success: {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print(f"from client: {data.decode()}")
                response = input("to client: ")
                conn.sendall(response.encode())
    print("Shutdown Server...")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    main(device)
