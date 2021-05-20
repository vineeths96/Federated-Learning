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
        PORT=5050,
        TIMEOUT=5,
        NUM_CLIENTS=1,
        GRADIENT_SIZE=14728266,
        K=10000,
        CHUNK=100,
        DELAY=5e-3,
        SEED=42,
    ):
        self.SERVER = SERVER
        self.PORT = PORT

        self.TIMEOUT = TIMEOUT
        self.NUM_CLIENTS = NUM_CLIENTS
        self.K = K
        self.GRADIENT_SIZE = GRADIENT_SIZE
        self.CHUNK = CHUNK
        self.DELAY = DELAY
        self.SEED = SEED

        self.ADDR = (SERVER, PORT)
        self.ADDR1 = (SERVER, PORT+10)

        self.START_OF_MESSAGE = torch.tensor(-float("inf"))
        self.END_OF_MESSAGE = torch.tensor(float("inf"))

        self.DEVICES = []

        self.serverTCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverTCP.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serverTCP.bind(self.ADDR)
        buffer_size = self.serverTCP.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print("TCP Buffer size [After]:%d" % buffer_size)

        self.serverUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.serverUDP.bind(self.ADDR1)
        self.serverUDP.settimeout(TIMEOUT)
        buffer_size = self.serverUDP.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
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
                flag = self.decode(msg)
                conn.setblocking(False)
            except:
                pass

            print(flag)
            if torch.isinf(flag) and torch.sign(flag) > 0:
                readnext = False

            if torch.isinf(flag) and torch.sign(flag) < 0:
                while True:
                    msg, addr = None, None
                    try:
                        msg, addr = self.serverUDP.recvfrom(BUFFER)
                    except:
                        pass

                    if not msg:
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
        print(f"Length of message received: {msg[:,0].unique().shape[0]}")

        indices = msg[:, 0].long()
        gradient = msg[:, 1]
        self.accumulated_gradient[indices] += gradient










        # msg = self.decode(buffer)
        # print(f"[{addr}] {msg}")
        #
        # gradient = msg
        # self.accumulated_gradient += gradient

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
                    for client in clients:
                        self.sendTCP(self.accumulated_gradient, client)
                        client.shutdown(1)
                        client.close()

                    clients = []
                    client_count = 0
                    self.accumulated_gradient.zero_()
        except StopIteration:
            self.stop()

    def stop(self):
        self.serverTCP.shutdown(1)
        self.serverTCP.close()


class TCPUDPKClient:
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
        self.K = K
        self.GRADIENT_SIZE = GRADIENT_SIZE
        self.CHUNK = CHUNK
        self.DELAY = DELAY
        self.SEED = SEED

        self.ADDR = (SERVER, PORT)
        self.ADDR1 = (SERVER, PORT+10)

        self.START_OF_MESSAGE = torch.tensor(-float("inf"))
        self.END_OF_MESSAGE = torch.tensor(float("inf"))

        self.clientTCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientTCP.connect(self.ADDR)

        self.clientUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
            self.clientUDP.sendto(encoded_message, self.ADDR1)

            time.sleep(self.DELAY)

    def receive(self):
        length = None
        buffer = bytearray()

        readnext = True
        while readnext:
            msg = self.clientUDP.recv(BUFFER)
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
