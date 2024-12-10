import socket
import pickle
import torch
from transformers import RobertaTokenizer
from model_split import a_model, c_model

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
texts = "This is a bad day!"
inputs = tokenizer(texts, return_tensors="pt")
a_model.eval()
c_model.eval()

with torch.no_grad():
    a_output = a_model(**inputs)

# 소켓으로 a_output 전송
HOST = '127.0.0.1'
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))
serialized = pickle.dumps(a_output)
client.sendall(serialized)
client.shutdown(socket.SHUT_WR)

data = b''
while True:
    packet = client.recv(4096)
    if not packet:
        break
    data += packet
client.close()

b_output = pickle.loads(data)

with torch.no_grad():
    c_output = c_model(outputs=b_output)
    logits = c_output.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)
print(probabilities)
