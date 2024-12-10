import socket
import struct
import torch
import io


def recv_all(sock, n):
    """
    Receive n bytes from a socket.

    Args:
        sock (socket_utils.socket): Socket instance.
        n (int): Bytes.

    Returns:
        bytes: Received data.
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_tensor(sock, data_length):
    """
    Recieve tensor that send by socket

    Args:
        sock (socket_utils.socket): Socket instance.

    Returns:
        torch.Tensor: Received tensor.
    """
    data = recv_all(sock, data_length)
    if not data:
        return None

    # Unserialize binary data(to tensor)
    buffer = io.BytesIO(data)
    tensor = torch.load(buffer)
    return tensor

def recv_header(sock):
    # Receive header
    header = recv_all(sock, 8)
    if not header:
        return None
    header1, header2= struct.unpack('!II', header)

    return header1, header2


def send_tensor(sock, head, tensor):
    """
    Send tensor by socket.

    Args:
        sock (socket.socket): Socket instance.
        tensor (torch.Tensor): Tensor to send.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Serialize tensor(to binary data)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    data = buffer.getvalue()

    if head < 200:
        # Send header
        data_length = len(data)
        header = (head, data_length)
        sock.sendall(struct.pack('!II', *header))

    # Send data
    sock.sendall(data)