import io
import torch
import socket
import threading
import numpy as np
import matplotlib.pyplot as plt


class ServerTCP:
    def __init__(self, HEADER=64, PORT=5050, MSG_SIZE=100000, SERVER=socket.gethostbyname(socket.gethostname())):
        self.HEADER = HEADER
        self.PORT = PORT
        self.SERVER = SERVER
        self.ADDR = (SERVER, PORT)
        self.FORMAT = "utf-8"
        self.DISCONNECT_MESSAGE = torch.tensor(float("inf"))

        self.SEND_BUF_SIZE = 4096
        self.RECV_BUF_SIZE = 4096

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            self.SEND_BUF_SIZE)
        self.server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            self.RECV_BUF_SIZE)

        self.server.bind(self.ADDR)

        self.vector = torch.zeros(MSG_SIZE)

        # buffsize = self.server.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        # print("Buffer size [After]:%d" % buffsize)

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

    def handle_client(self, conn, addr):
        print(f"[NEW CONNECTION] {addr} connected.")

        connected = True
        while connected:
            length = None
            buffer = bytearray()

            readnext = True
            while readnext:
                msg = conn.recv(8192)
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
            conn.send("Message received".encode(self.FORMAT))

            if not len(msg.shape) and torch.isinf(msg):
                connected = False
                print(f"[DROP CONNECTION] {addr} closed")
                conn.shutdown(1)
                conn.close()
                continue

            self.vector.index_add_(0, msg[:, 0].to(torch.int64), msg[:, 1])
            print("Vector", self.vector)

    def start(self):
        self.server.listen()
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        try:
            while True:
                conn, addr = self.server.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
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
        print(self.client.recv(2048).decode(self.FORMAT))








class ServerUDP:
    def __init__(self, HEADER=64, PORT=5050, MSG_SIZE=100000, CHUNK=100, SERVER=socket.gethostbyname(socket.gethostname())):
        self.HEADER = HEADER
        self.PORT = PORT
        self.MSG_SIZE = MSG_SIZE
        self.CHUNK = CHUNK
        self.SERVER = SERVER
        self.ADDR = (SERVER, PORT)
        self.FORMAT = "utf-8"
        self.DISCONNECT_MESSAGE = torch.tensor(float("inf"))

        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # self.SEND_BUF_SIZE = 4096
        # self.RECV_BUF_SIZE = 4096

        # self.server.setsockopt(
        #     socket.SOL_SOCKET,
        #     socket.SO_SNDBUF,
        #     self.SEND_BUF_SIZE)
        # self.server.setsockopt(
        #     socket.SOL_SOCKET,
        #     socket.SO_RCVBUF,
        #     self.RECV_BUF_SIZE)

        self.server.bind(self.ADDR)

        self.vector = None
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

    def handle_client(self):
        print(f"[NEW CONNECTION] {0} connected.")

        buffer = []
        readnext = True
        while readnext:
            msg, addr = self.server.recvfrom(1024 * 2)
            # print(msg)
            # self.server.sendto("Message received".encode(self.FORMAT), addr)

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

        self.vector = msg

        # reference_indices = torch.arange(self.MSG_SIZE)
        # lost_indices = torch.from_numpy(np.setdiff1d(reference_indices.numpy(), msg.numpy())).to(dtype=torch.int64)
        # lost_indices_locations = torch.zeros_like(reference_indices).index_fill_(0, lost_indices, 1)
        #
        # import datetime
        #
        # plt.plot(reference_indices.numpy(), lost_indices_locations.numpy(), label=f'{(lost_indices.shape[0] / self.MSG_SIZE)}')
        # plt.xlabel('Coordinate index')
        # plt.ylabel('Lost Packet')
        # plt.legend()
        # plt.savefig(f'./{datetime.datetime.now()}.jpg')
        # plt.show(block=False)
        # plt.pause(2)
        # plt.close()

        message = self.encode(torch.arange(25))
        # print(message)
        self.server.sendto(message, addr)

        self.server.sendto(self.encode(torch.tensor(float("inf"))), addr)

        # self.server.sendto("Message received".encode(self.FORMAT), addr)

    def start(self):
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        # try:
        #     while True:
        #         self.handle_client()
        # except KeyboardInterrupt:
        #     self.stop()
        while True:
            self.handle_client()

    def stop(self):
        self.server.close()


class ClientUDP:
    def __init__(self, HEADER=64, PORT=5050, CHUNK=100, SERVER=socket.gethostbyname(socket.gethostname())):
        self.HEADER = HEADER
        self.PORT = PORT
        self.CHUNK = CHUNK
        self.SERVER = SERVER
        self.ADDR = (SERVER, PORT)
        self.FORMAT = "utf-8"
        self.DISCONNECT_MESSAGE = torch.tensor(float("inf"))

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
        message = self.encode(tensor)

        self.client.sendto(message, self.ADDR)

        # data, server = self.client.recvfrom(1024)
        # print(data.decode(self.FORMAT))

    def receive(self,):
        buffer = []
        readnext = True
        while readnext:
            msg, addr = self.client.recvfrom(1024 * 2)
            # print(msg)
            # self.server.sendto("Message received".encode(self.FORMAT), addr)

            try:
                decoded_msg = self.decode(msg)
            except:
                continue

            if not len(decoded_msg.shape) and torch.isinf(decoded_msg):
                break

            buffer.append(decoded_msg)

        if len(buffer) > 1:
            msg = torch.cat(buffer)
        else:
            print(buffer)
            msg = buffer[0]

        print(f"[{addr}] {msg}")
        print(f"Length of message received: {msg.shape[0]}")

