import socket
import torch


# TODO: When send tensor to server, must send attention_mask together!!!!

def main(device, host='127.0.0.1', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connecting to server {host}:{port}")

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