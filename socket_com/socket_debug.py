import io
import time
import torch
import socket
import threading
import numpy as np
import matplotlib.pyplot as plt

from seed import set_seed


UDP_DEBUG = False


class TCPServer:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        PORT=5050,
        NUM_CLIENTS=1,
        GRADIENT_SIZE=14728266,
        DELAY=5e-1,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.NUM_CLIENTS = NUM_CLIENTS
        self.GRADIENT_SIZE = GRADIENT_SIZE
        self.DELAY = DELAY
        self.SEED = SEED

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

        self.accumulated_gradient = torch.zeros(self.GRADIENT_SIZE)

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

    def send(self, tensor, conn):
        encoded_message = self.encode(tensor)
        conn.send(encoded_message)

        # time.sleep(self.DELAY)
        # self.send_EOT(conn)

    def send_EOT(self, conn):
        encoded_message = self.encode(self.END_OF_MESSAGE)
        conn.send(encoded_message)

    def receive(self, conn, addr):
        length = None
        buffer = bytearray()

        readnext = True
        while readnext:
            msg = conn.recv(BUFFER)
            buffer += msg

            if len(buffer) == length:
                break

            while True:
                if length is None:
                    if b":" not in buffer:
                        break

                    length_str, ignored, buffer = buffer.partition(b":")
                    length = int(length_str)

                if len(buffer) < length:
                    break

                buffer = buffer[length:]

                length = None
                break

        msg = self.decode(buffer)
        print(f"[{addr}] {msg}")

        if not UDP_DEBUG:
            gradient = msg
            self.accumulated_gradient += gradient

            # self.send(msg, conn)
            # conn.shutdown(1)
            # conn.close()
            return
        else:
            self.stop()
            return

    def start(self):
        self.server.listen()
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        try:
            if not UDP_DEBUG:
                clients = []
                client_count = 0

                while True:
                    conn, addr = self.server.accept()
                    clients.append(conn)
                    client_count += 1

                    thread = threading.Thread(target=self.receive, args=(conn, addr))
                    thread.start()
                    thread.join()
                    # print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")

                    if threading.activeCount() == 1 and client_count == self.NUM_CLIENTS:
                        for client in clients:
                            self.send(self.accumulated_gradient, client)
                            client.shutdown(1)
                            client.close()

                        clients = []
                        client_count = 0
                        self.accumulated_gradient.zero_()
            else:
                conn, addr = self.server.accept()
                thread = threading.Thread(target=self.receive, args=(conn, addr))
                thread.start()
                print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.server.shutdown(1)
        self.server.close()


class TCPClient:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        PORT=5050,
        GRADIENT_SIZE=14728266,
        DELAY=5e-1,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.GRADIENT_SIZE = GRADIENT_SIZE
        self.DELAY = DELAY
        self.SEED = SEED

        self.ADDR = (SERVER, PORT)
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

        # time.sleep(self.DELAY)
        # self.send_EOT()

    def send_EOT(self):
        encoded_message = self.encode(self.DISCONNECT_MESSAGE)
        self.client.sendto(encoded_message, self.ADDR)

    def receive(self):
        length = None
        buffer = bytearray()

        readnext = True
        while readnext:
            msg = self.client.recv(BUFFER)
            buffer += msg

            if len(buffer) == length:
                break

            while True:
                if length is None:
                    if b":" not in buffer:
                        break

                    length_str, ignored, buffer = buffer.partition(b":")
                    length = int(length_str)

                if len(buffer) < length:
                    break

                buffer = buffer[length:]

                length = None
                break

        msg = self.decode(buffer)

        return msg


class UDPServer:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        PORT=5050,
        NUM_CLIENTS=1,
        TIMEOUT=5,
        GRADIENT_SIZE=14728266,
        MSG_SIZE=100000,
        CHUNK=100,
        DELAY=5e-4,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.NUM_CLIENTS = NUM_CLIENTS
        self.MSG_SIZE = MSG_SIZE
        self.GRADIENT_SIZE = GRADIENT_SIZE
        self.CHUNK = CHUNK
        self.DELAY = DELAY
        self.SEED = SEED

        self.ADDR = (SERVER, PORT)
        self.END_OF_MESSAGE = torch.tensor(float("inf"))

        self.DEVICES = []
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
        self.server.settimeout(TIMEOUT)

        buffer_size = self.server.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print("Buffer size [After]:%d" % buffer_size)

        self._indices_queue = []
        self.accumulated_gradient = torch.zeros(self.GRADIENT_SIZE)

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
        from TCPSocket import TCPClient

        clientTCP = TCPClient(SERVER=SERVER_COMP, MSG_SIZE=MSG_SIZE, DELAY=DELAY)
        clientTCP.send(self.END_OF_MESSAGE)

    def receive(self):
        buffer = []
        readnext = True
        while readnext:
            msg, addr = self.server.recvfrom(BUFFER)

            if addr not in self.DEVICES:
                self.DEVICES.append(addr)

            try:
                decoded_msg = self.decode(msg)
            except:
                continue

            if not len(decoded_msg.shape) and torch.isinf(decoded_msg):
                break

            buffer.append(decoded_msg)

        # TODO Handle buffer [] case
        if len(buffer) > 1:
            # print(buffer)
            msg = torch.cat(buffer)
        else:
            try:
                msg = buffer[0]
            except:
                print(buffer, len(buffer))

        print(f"[{addr}] {msg}")
        print(f"Length of message received: {msg.shape[0]}")

        if not UDP_DEBUG:
            indices = msg[:, 0].long()
            gradient = msg[:, 1]
            self.accumulated_gradient[indices] += gradient

            # time.sleep(self.DELAY)
            # self.send(msg, addr)
        else:
            self.send_TCP_EOT()

    def start(self):
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        try:
            while True:
                self.receive()

                if len(self.DEVICES) < self.NUM_CLIENTS:
                    continue

                if not self._indices_queue:
                    set_seed(self.SEED)
                    self._indices_queue = torch.randperm(self.GRADIENT_SIZE).split(self.MSG_SIZE)
                    self._indices_queue = list(self._indices_queue)

                RandK_indices = self._indices_queue.pop().long()
                print(self.GRADIENT_SIZE, self.MSG_SIZE)
                print(RandK_indices)
                print(torch.randperm(100).split(10))
                exit(44)
                RandK_flat_grad = self.accumulated_gradient[RandK_indices]
                accumulated_grad_indices = torch.vstack([RandK_indices, RandK_flat_grad]).T

                for client in self.DEVICES:
                    self.send(accumulated_grad_indices, client)

                self.DEVICES = []
                self.accumulated_gradient.zero_()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.server.close()


class UDPClient:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        PORT=5050,
        TIMEOUT=5,
        GRADIENT_SIZE=14728266,
        MSG_SIZE=100000,
        CHUNK=100,
        DELAY=5e-4,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.TIMEOUT = TIMEOUT
        self.GRADIENT_SIZE = GRADIENT_SIZE
        self.MSG_SIZE = MSG_SIZE

        self.CHUNK = CHUNK
        self.DELAY = DELAY
        self.SEED = SEED

        self.ADDR = (SERVER, PORT)
        self.END_OF_MESSAGE = torch.tensor(float("inf"))

        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # buffer_size = self.client.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        # print("Buffer size [After]:%d" % buffer_size)

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

        # print(f"[{addr}] {msg}")
        # print(f"Length of message received: {msg.shape[0]}")

        return msg

    def receive_TCP_EOT(self):
        from TCPSocket import TCPServer

        server = TCPServer(SERVER=SERVER_COMP, MSG_SIZE=MSG_SIZE, DELAY=DELAY)
        server.start()
