import io
import time
import torch
import socket
import threading
import numpy as np
import matplotlib.pyplot as plt
from parameters import *


class ServerUDP:
    def __init__(
        self, SERVER=socket.gethostbyname(socket.gethostname()), PORT=5050, MSG_SIZE=100000, CHUNK=100, DELAY=5e-4
    ):
        self.SERVER = SERVER
        self.PORT = PORT
        self.MSG_SIZE = MSG_SIZE
        self.CHUNK = CHUNK
        self.DELAY = DELAY

        self.ADDR = (SERVER, PORT)
        self.END_OF_MESSAGE = torch.tensor(float("inf"))

        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # self.SEND_BUF_SIZE = 4096
        # self.RECV_BUF_SIZE = 4096
        #
        # self.server.setsockopt(
        #     socket.SOL_SOCKET,
        #     socket.SO_SNDBUF,
        #     self.SEND_BUF_SIZE)
        # self.server.setsockopt(
        #     socket.SOL_SOCKET,
        #     socket.SO_RCVBUF,
        #     self.RECV_BUF_SIZE)

        self.server.bind(self.ADDR)
        buffer_size = self.server.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print("Buffer size [After]:%d" % buffer_size)

    def encode(self, tensor):
        file = io.BytesIO()
        torch.save(tensor, file)

        file.seek(0)
        encoded = file.read()

        return encoded

    def decode(self, buffer):
        tensor = torch.load(io.BytesIO(buffer))

        return tensor

    def send(self, tensor, addr):
        messages = tensor.split(self.CHUNK)

        for message in messages:
            encoded_message = self.encode(message.clone())
            self.server.sendto(encoded_message, addr)

            time.sleep(self.DELAY)

        self.send_EOT(addr)

    def send_EOT(self, addr):
        encoded_message = self.encode(self.END_OF_MESSAGE)
        self.server.sendto(encoded_message, addr)

    def send_TCP_EOT(self):
        from TCPSocket import ClientTCP

        clientTCP = ClientTCP(SERVER="10.216.18.179", MSG_SIZE=MSG_SIZE, DELAY=DELAY)
        clientTCP.send(self.END_OF_MESSAGE)

    def receive(self):
        buffer = []
        readnext = True
        while readnext:
            msg, addr = self.server.recvfrom(BUFFER)

            try:
                decoded_msg = self.decode(msg)
            except:
                continue

            if not len(decoded_msg.shape) and torch.isinf(decoded_msg):
                print(decoded_msg)
                break

            buffer.append(decoded_msg)

        # TODO Handle buffer [] case
        if len(buffer) > 1:
            msg = torch.cat(buffer)
        else:
            try:
                msg = buffer[0]
            except:
                print(buffer, len(buffer))

        print(f"[{addr}] {msg}")
        print(f"Length of message received: {msg.shape[0]}")

        # Uncomment
        # time.sleep(self.DELAY)
        # self.send(msg, addr)

        self.send_TCP_EOT()

    def start(self):
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        try:
            while True:
                self.receive()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.server.close()


class ClientUDP:
    def __init__(
        self, SERVER=socket.gethostbyname(socket.gethostname()), MSG_SIZE=100000, PORT=5050, CHUNK=100, DELAY=5e-4
    ):
        self.SERVER = SERVER
        self.PORT = PORT
        self.MSG_SIZE = MSG_SIZE
        self.CHUNK = CHUNK
        self.DELAY = DELAY

        self.ADDR = (SERVER, PORT)
        self.END_OF_MESSAGE = torch.tensor(float("inf"))

        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        buffer_size = self.client.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print("Buffer size [After]:%d" % buffer_size)

    def encode(self, tensor):
        file = io.BytesIO()
        torch.save(tensor, file)

        file.seek(0)
        encoded = file.read()

        return encoded

    def decode(self, buffer):
        tensor = torch.load(io.BytesIO(buffer))

        return tensor

    def send(self, tensor):
        messages = tensor.split(self.CHUNK)

        for message in messages:
            encoded_message = self.encode(message.clone())
            self.client.sendto(encoded_message, self.ADDR)

            time.sleep(self.DELAY)

        self.send_EOT()

    def send_EOT(self):
        encoded_message = self.encode(self.END_OF_MESSAGE)
        self.client.sendto(encoded_message, self.ADDR)

    def receive(self):
        buffer = []
        readnext = True
        while readnext:
            msg, addr = self.client.recvfrom(BUFFER)

            try:
                decoded_msg = self.decode(msg)
            except:
                continue

            if not len(decoded_msg.shape) and torch.isinf(decoded_msg):
                break

            buffer.append(decoded_msg)

        # TODO Handle buffer [] case
        if len(buffer) > 1:
            msg = torch.cat(buffer)
        else:
            msg = buffer[0]

        print(f"[{addr}] {msg}")
        print(f"Length of message received: {msg.shape[0]}")

    def receive_TCP_EOT(self):
        from TCPSocket import ServerTCP

        server = ServerTCP(SERVER="10.216.18.179", MSG_SIZE=MSG_SIZE, DELAY=DELAY)
        server.start()
