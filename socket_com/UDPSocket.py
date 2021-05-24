import io
import time
import torch
import socket

from seed import set_seed

UDP_DEBUG = False
BUFFER = 1024 * 64


class UDPServer:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        PORT=5050,
        NUM_CLIENTS=1,
        TIMEOUT=5,
        GRADIENT_SIZE=14728266,
        CHUNK=100,
        DELAY=5e-3,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.NUM_CLIENTS = NUM_CLIENTS
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
        msg, addr = None, None

        try:
            while readnext:
                msg, addr = self.server.recvfrom(BUFFER)

                # if addr not in self.DEVICES:
                #     self.DEVICES.append(addr)

                try:
                    decoded_msg = self.decode(msg)
                except:
                    continue

                if not len(decoded_msg.shape) and torch.isinf(decoded_msg):
                    if addr not in self.DEVICES:
                        self.DEVICES.append(addr)

                    break

                buffer.append(decoded_msg)
        except:
            if addr and addr not in self.DEVICES:
                    self.DEVICES.append(addr)

        if len(buffer) > 1:
            msg = torch.cat(buffer)
        else:
            msg = buffer[0]

        print(f"[{addr}] {msg}")
        print(f"Length of message received: {msg.shape[0]}")

        indices = msg[:, 0].long()
        gradient = msg[:, 1]
        self.accumulated_gradient[indices] += gradient

    def start(self):
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        try:
            while True:
                self.receive()

                if len(self.DEVICES) < self.NUM_CLIENTS:
                    continue

                accumulated_grad_indices = torch.vstack(
                    [torch.arange(self.GRADIENT_SIZE), self.accumulated_gradient]
                ).T

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
        CHUNK=100,
        DELAY=5e-3,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.TIMEOUT = TIMEOUT
        self.GRADIENT_SIZE = GRADIENT_SIZE

        self.CHUNK = CHUNK
        self.DELAY = DELAY
        self.SEED = SEED

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

    def receive(self):
        buffer = []
        readnext = True

        try:
            while readnext:
                msg, addr = self.client.recvfrom(BUFFER)

                try:
                    decoded_msg = self.decode(msg)
                except:
                    continue

                if not len(decoded_msg.shape) and torch.isinf(decoded_msg):
                    break

                buffer.append(decoded_msg)
        except socket.error:
            pass

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


class UDPKServer:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        PORT=5050,
        NUM_CLIENTS=1,
        TIMEOUT=5,
        GRADIENT_SIZE=14728266,
        K=10000,
        CHUNK=100,
        DELAY=5e-3,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.NUM_CLIENTS = NUM_CLIENTS
        self.K = K
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
        msg, addr = None, None

        try:
            while readnext:
                msg, addr = self.server.recvfrom(BUFFER)

                # if addr not in self.DEVICES:
                #     self.DEVICES.append(addr)

                try:
                    decoded_msg = self.decode(msg)
                except:
                    continue

                if not len(decoded_msg.shape) and torch.isinf(decoded_msg):
                    if addr not in self.DEVICES:
                        self.DEVICES.append(addr)

                    break

                buffer.append(decoded_msg)
        except socket.error:
            if addr and addr not in self.DEVICES:
                    self.DEVICES.append(addr)

        if len(buffer) > 1:
            msg = torch.cat(buffer)
        else:
            msg = buffer[0]

        print(f"[{addr}] {msg}")
        print(f"Length of message received: {msg.shape[0]}")

        indices = msg[:, 0].long()
        gradient = msg[:, 1]
        self.accumulated_gradient[indices] += gradient

        # time.sleep(self.DELAY)
        # self.send(msg, addr)

    def start(self):
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        try:
            while True:
                self.receive()

                if len(self.DEVICES) < self.NUM_CLIENTS:
                    continue

                if not self._indices_queue:
                    set_seed(self.SEED)
                    self._indices_queue = torch.randperm(self.GRADIENT_SIZE).split(self.K)
                    self._indices_queue = list(self._indices_queue)

                RandK_indices = self._indices_queue.pop().long()
                RandK_flat_grad = self.accumulated_gradient[RandK_indices]
                accumulated_grad_indices = torch.vstack([RandK_indices, RandK_flat_grad]).T

                print(RandK_indices)

                for client in self.DEVICES:
                    self.send(accumulated_grad_indices, client)

                self.DEVICES = []
                self.accumulated_gradient.zero_()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.server.close()


class UDPKClient:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        PORT=5050,
        TIMEOUT=5,
        GRADIENT_SIZE=14728266,
        K=10000,
        CHUNK=100,
        DELAY=5e-3,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.TIMEOUT = TIMEOUT
        self.GRADIENT_SIZE = GRADIENT_SIZE
        self.K = K
        self.CHUNK = CHUNK
        self.DELAY = DELAY
        self.SEED = SEED

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

    def receive(self):
        buffer = []
        readnext = True

        try:
            while readnext:
                msg, addr = self.client.recvfrom(BUFFER)

                try:
                    decoded_msg = self.decode(msg)
                except:
                    continue

                if not len(decoded_msg.shape) and torch.isinf(decoded_msg):
                    break

                buffer.append(decoded_msg)
        except socket.error:
            pass

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
