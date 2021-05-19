import io
import time
import torch
import socket
import threading

from seed import set_seed

BUFFER = 1024 * 64


class TCPServer:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        PORT=5050,
        NUM_CLIENTS=1,
        GRADIENT_SIZE=14728266,
        DELAY=5e-3,
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

        gradient = msg
        self.accumulated_gradient += gradient

        return

    def start(self):
        self.server.listen()
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        try:
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
        DELAY=5e-3,
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


class TCPKServer:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        PORT=5050,
        NUM_CLIENTS=1,
        GRADIENT_SIZE=14728266,
        K=10000,
        DELAY=5e-3,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.NUM_CLIENTS = NUM_CLIENTS
        self.K = K
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

        self._indices_queue = []
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

        indices = msg[:, 0].long()
        gradient = msg[:, 1]
        self.accumulated_gradient[indices] += gradient

        return

    def start(self):
        self.server.listen()
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        try:
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

                    if not self._indices_queue:
                        set_seed(self.SEED)
                        self._indices_queue = torch.randperm(self.GRADIENT_SIZE).split(self.K)
                        self._indices_queue = list(self._indices_queue)

                    RandK_indices = self._indices_queue.pop().long()
                    RandK_flat_grad = self.accumulated_gradient[RandK_indices]
                    accumulated_grad_indices = torch.vstack([RandK_indices, RandK_flat_grad]).T

                    for client in clients:
                        self.send(accumulated_grad_indices, client)
                        client.shutdown(1)
                        client.close()

                    clients = []
                    client_count = 0
                    self.accumulated_gradient.zero_()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.server.shutdown(1)
        self.server.close()


class TCPKClient:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        PORT=5050,
        GRADIENT_SIZE=14728266,
        K=10000,
        DELAY=5e-3,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.GRADIENT_SIZE = GRADIENT_SIZE
        self.K = K
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
