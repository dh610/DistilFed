import socket
import pickle

from typing import Any
from collections import OrderedDict

import torch
from evaluate import load as load_metric
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import torch.nn.functional as F

from transformers import RobertaTokenizer
from SL.model_split import a_model, c_model

disable_progress_bar()
fds = None  # Cache FederatedDataset


def load_data(
    partition_id: int, num_partitions: int, model_name: str
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load IMDB data (training and eval)"""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Partition the IMDB dataset into N partitions
        partitioner = IidPartitioner(num_partitions=num_partitions*20)
        fds = FederatedDataset(
            dataset="stanfordnlp/imdb", partitioners={"train": partitioner}
        )
    partition = fds.load_partition(partition_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, add_special_tokens=True)

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def get_model(model_name):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def get_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model, parameters) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# Loss 계산 (교사 모델의 Soft Label 사용)
def distillation_loss(student_logits, teacher_logits, hard_labels, temperature=2.0, alpha=0.5):
    # Soft Label Loss
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    # Hard Label Loss
    hard_loss = F.cross_entropy(student_logits, hard_labels)

    # Combined Loss
    return alpha * soft_loss + (1 - alpha) * hard_loss

teacher_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
HOST = '127.0.0.1'
PORT = 5000

def train(net, trainloader, epochs, device) -> None:
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            teacher_inputs = batch#teacher_tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=128)
            with torch.no_grad():
                a_output = a_model(**teacher_inputs)
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
                teacher_logits = c_output.logits

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = net(**batch)
            student_logits = outputs.logits
            loss = distillation_loss(student_logits, teacher_logits, batch["labels"], temperature=2.0, alpha=0.5)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader, device) -> tuple[Any | float, Any]:
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy