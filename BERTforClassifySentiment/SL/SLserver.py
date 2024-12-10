import socket
import pickle
import torch
from model_split import b_model

HOST = '127.0.0.1'
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

b_model.eval()

while True:
    conn, addr = server.accept()
    data = b''
    while True:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet
    # data에는 직렬화된 텐서 정보가 들어있음
    a_output = pickle.loads(data)
    with torch.no_grad():
        b_output = b_model(embedding_output=a_output)
    # 출력 텐서를 직렬화 후 전송
    serialized = pickle.dumps(b_output)
    conn.sendall(serialized)
    conn.close()
