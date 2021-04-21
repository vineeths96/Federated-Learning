import io
import time
import torch
import socket
import threading
import numpy as np
import matplotlib.pyplot as plt


class ServerTCP:
    def __init__(self, SERVER=socket.gethostbyname(socket.gethostname()), PORT=5050, MSG_SIZE=100000, DELAY=5e-2):
        self.SERVER = SERVER
        self.PORT = PORT
        self.MSG_SIZE = MSG_SIZE
        self.DELAY = DELAY

        self.ADDR = (SERVER, PORT)
        self.END_OF_MESSAGE = torch.tensor(float("inf"))

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

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

        packet_size = len(file.getvalue())
        header = "{0}:".format(packet_size)
        header = bytes(header.encode())

        encoded = bytearray()
        encoded += header

        file.seek(0)
        encoded += file.read()

        return encoded

    def decode(self, buffer):
        tensor = torch.load(io.BytesIO(buffer))

        return tensor

    def send_EOT(self, conn):
        encoded_message = self.encode(self.END_OF_MESSAGE)
        conn.send(encoded_message)

    def send(self, tensor, conn):
        encoded_message = self.encode(tensor)
        conn.send(encoded_message)

        time.sleep(self.DELAY)
        self.send_EOT(conn)

    def receive(self, conn, addr):
        print(f"[NEW CONNECTION] {addr} connected.")

        connected = True
        while connected:
            length = None
            buffer = bytearray()

            readnext = True
            while readnext:
                msg = conn.recv(2048 * 8)
                buffer += msg

                if len(buffer) == length:
                    readnext = False

                if length is None:
                    if b":" not in buffer:
                        break

                    length_str, ignored, buffer = buffer.partition(b":")
                    length = int(length_str)

                    if len(buffer) == length:
                        readnext = False

            buffer = buffer[:length]

            msg = self.decode(buffer)
            print(f"[{addr}] {msg}")

            if not len(msg.shape) and torch.isinf(msg):
                connected = False
                print(f"[DROP CONNECTION] {addr} closed")
                conn.shutdown(1)
                conn.close()
                continue

            self.send(msg, conn)

    def start(self):
        self.server.listen()
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        try:
            while True:
                conn, addr = self.server.accept()
                thread = threading.Thread(target=self.receive, args=(conn, addr))
                thread.start()
                print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.server.shutdown(1)
        self.server.close()


class ClientTCP:
    def __init__(self, HEADER=64, PORT=5050, SERVER=socket.gethostbyname(socket.gethostname())):
        self.HEADER = HEADER
        self.PORT = PORT
        self.SERVER = SERVER

        self.ADDR = (SERVER, PORT)
        self.FORMAT = "utf-8"
        self.DISCONNECT_MESSAGE = torch.tensor(float("inf"))

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(self.ADDR)

    def encode(self, tensor):
        file = io.BytesIO()
        torch.save(tensor, file)

        packet_size = len(file.getvalue())
        header = "{0}:".format(packet_size)
        header = bytes(header.encode())

        encoded = bytearray()
        encoded += header

        file.seek(0)
        encoded += file.read()

        return encoded

    def decode(self, buffer):
        tensor = torch.load(io.BytesIO(buffer))

        return tensor

    def send(self, tensor):
        message = self.encode(tensor)

        self.client.send(message)
        self.receive()

    def receive(self):
        connected = True
        while connected:
            length = None
            buffer = bytearray()

            readnext = True
            while readnext:
                msg = self.client.recv(2048 * 8)
                buffer += msg

                if length and len(buffer) >= length:
                    readnext = False

                if length is None:
                    if b":" not in buffer:
                        break

                    length_str, ignored, buffer = buffer.partition(b":")
                    length = int(length_str)

                    if len(buffer) == length:
                        readnext = False

            buffer = buffer[:length]
            msg = self.decode(buffer)
            print(f"[{0000}] {msg}")

            if not len(msg.shape) and torch.isinf(msg):
                connected = False
                print(f"[DROP CONNECTION] closed")
                self.client.send(self.encode(self.DISCONNECT_MESSAGE))
                # conn.shutdown(1)
                # conn.close()
                continue


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

        # self.server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

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

    def receive(self):
        buffer = []
        readnext = True
        while readnext:
            msg, addr = self.server.recvfrom(2048 * 8)

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

        # time.sleep(self.DELAY)
        self.send(msg, addr)

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

    def receive(
        self,
    ):
        buffer = []
        readnext = True
        while readnext:
            msg, addr = self.client.recvfrom(2048 * 4)

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
