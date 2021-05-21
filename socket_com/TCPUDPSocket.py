import io
import time
import torch
import socket
import threading

from seed import set_seed

BUFFER = 1024 * 64


class TCPUDPKServer:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        TCP_PORT=5050,
        UDP_PORT=5060,
        TIMEOUT=5,
        NUM_CLIENTS=1,
        GRADIENT_SIZE=14728266,
        K=10000,
        CHUNK=100,
        DELAY=5e-3,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.TCP_PORT = TCP_PORT
        self.UDP_PORT = UDP_PORT

        self.TIMEOUT = TIMEOUT
        self.NUM_CLIENTS = NUM_CLIENTS
        self.K = K
        self.GRADIENT_SIZE = GRADIENT_SIZE
        self.CHUNK = CHUNK
        self.DELAY = DELAY
        self.SEED = SEED

        self.TCP_ADDR = (SERVER, TCP_PORT)
        self.UDP_ADDR = (SERVER, UDP_PORT)

        self.START_OF_MESSAGE = torch.tensor(-float("inf"))
        self.END_OF_MESSAGE = torch.tensor(float("inf"))

        self.DEVICES = []

        self.serverTCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverTCP.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serverTCP.bind(self.TCP_ADDR)
        # buffer_size = self.serverTCP.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        # print("TCP Buffer size [After]:%d" % buffer_size)

        # self.serverUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.serverUDP.bind(self.UDP_ADDR)
        # self.serverUDP.settimeout(self.TIMEOUT)
        # buffer_size = self.serverUDP.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        # print("Buffer size [After]:%d" % buffer_size)

        self.UDP_PORT_LIST = [UDP_PORT + i for i in range(NUM_CLIENTS)]
        self.UDP_ADDR_LIST = [(SERVER, UDP_PORT_ELEMENT) for UDP_PORT_ELEMENT in self.UDP_PORT_LIST]
        self.serverUDP = [None] * NUM_CLIENTS

        for ind in range(NUM_CLIENTS):
            self.serverUDP[ind] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.serverUDP[ind].bind(self.UDP_ADDR_LIST[ind])
            self.serverUDP[ind].settimeout(self.TIMEOUT)
            # buffer_size = self.serverUDP[ind].getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            # print("Buffer size [After]:%d" % buffer_size)

        print(self.serverUDP)

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

    def sendTCP(self, tensor, conn):
        encoded_message = self.encode(tensor)
        conn.send(encoded_message)

        # time.sleep(self.DELAY)
        # self.send_EOT(conn)

    def sendTCP_SOT(self, conn):
        encoded_message = self.encode(self.START_OF_MESSAGE)
        conn.send(encoded_message)

    def sendTCP_EOT(self, conn):
        encoded_message = self.encode(self.END_OF_MESSAGE)
        conn.send(encoded_message)

    def sendUDP(self, tensor, addr):
        messages = tensor.split(self.CHUNK)

        for message in messages:
            encoded_message = self.encode(message.clone())
            self.serverUDP.sendto(encoded_message, addr)

            time.sleep(self.DELAY)

    def receive(self, conn, addr):
        buffer = []
        readnext = True
        while readnext:
            try:
                msg = conn.recv(BUFFER)
                decoded_msg = self.decode(msg)
                flag = decoded_msg[0]
                local_rank = int(decoded_msg[1].item())
                conn.setblocking(False)
            except socket.error as e:
                print(e)
                pass

            print(local_rank, flag)
            if torch.isinf(flag) and torch.sign(flag) > 0:
                readnext = False

            if torch.isinf(flag) and torch.sign(flag) < 0:
                while True:
                    try:
                        msg, addr = self.serverUDP[local_rank].recvfrom(BUFFER)
                        print(self.serverUDP[local_rank], local_rank)
                    except socket.error as e:
                        print(self.serverUDP[local_rank], local_rank)
                        print(e)
                        break

                    if addr not in self.DEVICES:
                        self.DEVICES.append(addr)

                    try:
                        decoded_msg = self.decode(msg)
                    except:
                        continue

                    buffer.append(decoded_msg)

        # TODO Handle buffer [] case
        if len(buffer) > 1:
            msg = torch.cat(buffer)
        else:
            msg = buffer[0]

        print(f"[{addr}] {msg}")
        print(f"Length of message received: {msg.shape[0]}")

        indices = msg[:, 0].long()
        gradient = msg[:, 1]
        self.accumulated_gradient[indices] += gradient

        return

    def start(self):
        self.serverTCP.listen()
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        try:
            clients = []
            client_count = 0

            while True:
                conn, addr = self.serverTCP.accept()
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

                    print(RandK_indices)
                    self.DEVICES.sort(key=lambda device: device[0])
                    clients.sort(key=lambda client: client.getpeername())

                    for client, device in zip(clients, self.DEVICES):
                        self.sendTCP_SOT(client)
                        self.sendUDP(accumulated_grad_indices, device)
                        self.sendTCP_EOT(client)
                        client.shutdown(1)
                        client.close()

                    clients = []
                    client_count = 0
                    self.DEVICES = []
                    self.accumulated_gradient.zero_()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.serverTCP.shutdown(1)
        self.serverTCP.close()
        self.serverUDP.close()


class TCPUDPKClient:
    def __init__(
        self,
        SERVER=socket.gethostbyname(socket.gethostname()),
        TCP_PORT=5050,
        UDP_PORT=5060,
        TIMEOUT=5,
        GRADIENT_SIZE=14728266,
        K=10000,
        CHUNK=100,
        DELAY=5e-3,
        SEED=42,
        LOCAL_RANK=0,
    ):
        self.SERVER = SERVER
        self.TCP_PORT = TCP_PORT
        self.UDP_PORT = UDP_PORT

        self.TIMEOUT = TIMEOUT
        self.K = K
        self.GRADIENT_SIZE = GRADIENT_SIZE
        self.CHUNK = CHUNK
        self.DELAY = DELAY
        self.SEED = SEED
        self.LOCAL_RANK = LOCAL_RANK

        self.TCP_ADDR = (SERVER, TCP_PORT)
        self.UDP_ADDR = (SERVER, UDP_PORT)

        self.START_OF_MESSAGE = torch.tensor([-float("inf"), LOCAL_RANK])
        self.END_OF_MESSAGE = torch.tensor([float("inf"), LOCAL_RANK])

        self.clientTCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientTCP.connect(self.TCP_ADDR)

        self.clientUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.clientUDP.settimeout(self.TIMEOUT)

    def encode(self, tensor):
        file = io.BytesIO()
        torch.save(tensor, file)

        file.seek(0)
        encoded = file.read()

        return encoded

    def decode(self, buffer):
        tensor = torch.load(io.BytesIO(buffer))

        return tensor

    def sendTCP(self, tensor, conn):
        encoded_message = self.encode(tensor)
        conn.send(encoded_message)

        # time.sleep(self.DELAY)
        # self.send_EOT(conn)

    def sendTCP_SOT(self):
        encoded_message = self.encode(self.START_OF_MESSAGE)
        self.clientTCP.send(encoded_message)

    def sendTCP_EOT(self):
        encoded_message = self.encode(self.END_OF_MESSAGE)
        self.clientTCP.send(encoded_message)

    def sendUDP(self, tensor):
        messages = tensor.split(self.CHUNK)

        for message in messages:
            encoded_message = self.encode(message.clone())
            self.clientUDP.sendto(encoded_message, self.UDP_ADDR)

            time.sleep(self.DELAY)

    def receive(self):
        buffer = []
        readnext = True
        while readnext:
            try:
                msg = self.clientTCP.recv(BUFFER)
                flag = self.decode(msg)
                self.clientTCP.setblocking(False)
            except socket.error:
                pass

            if torch.isinf(flag) and torch.sign(flag) > 0:
                readnext = False

            if torch.isinf(flag) and torch.sign(flag) < 0:
                while True:
                    try:
                        msg, addr = self.clientUDP.recvfrom(BUFFER)
                    except socket.error:
                        break

                    try:
                        decoded_msg = self.decode(msg)
                    except:
                        continue

                    buffer.append(decoded_msg)

        # TODO Handle buffer [] case
        if len(buffer) > 1:
            msg = torch.cat(buffer)
        else:
            msg = buffer[0]

        # print(f"[{addr}] {msg}")
        # print(f"Length of message received: {msg.shape[0]}")

        return msg
