import time
import torch
from TCPSocket import TCPClient
from UDPSocket import UDPClient
from TCPUDPSocket import TCPUDPKClient
from parameters import *

torch.manual_seed(42)
GRADIENT_SIZE = 14728266


if use_TCPUDP:
    for msg in MSG_SIZES:
        time_sum = 0
        for i in range(REPS):
            client = TCPUDPKClient(SERVER=SERVER, DELAY=DELAY, CHUNK=CHUNK)

            message = torch.cat([torch.arange(msg).unsqueeze(1), 1 * torch.arange(msg).unsqueeze(1)], dim=-1).to(
                torch.float32
            )

            start = time.time()
            client.sendTCP_SOT()
            client.sendUDP(message)

            # time.sleep(15)
            client.sendTCP_EOT()

            print(client.receive())
            time_elapsed = time.time() - start
            time_sum += time_elapsed

        print("Total: ", time_sum)
        print(f"MSG: [{msg}] Avg Time: {time_sum / REPS}")
elif use_TCP:
    for msg in MSG_SIZES:
        time_sum = 0
        for i in range(REPS):
            client = TCPClient(SERVER=SERVER, DELAY=DELAY)

            message = torch.cat([torch.arange(msg).unsqueeze(1), 1 * torch.arange(msg).unsqueeze(1)], dim=-1).to(
                torch.float32
            )

            start = time.time()
            client.send(message)
            print(client.receive())
            time_elapsed = time.time() - start
            time_sum += time_elapsed

        print("Total: ", time_sum)
        print(f"MSG: [{msg}] Avg Time: {time_sum / REPS}")
else:
    for msg in MSG_SIZES:
        time_sum = 0
        for i in range(REPS):
            client = UDPClient(SERVER=SERVER, CHUNK=CHUNK, DELAY=DELAY)

            message = torch.vstack([torch.arange(GRADIENT_SIZE), 1 * torch.arange(GRADIENT_SIZE)]).to(torch.float32).T

            indices_queue = torch.randperm(GRADIENT_SIZE).split(msg)
            indices_queue = list(indices_queue)

            RandK_indices = indices_queue.pop().long()
            message = message[RandK_indices, :]

            start = time.time()
            client.send(message.clone())
            if UDP_DEBUG:
                print(client.receive_TCP_EOT())
            else:
                print(client.receive())
            time_elapsed = time.time() - start
            time_sum += time_elapsed

        print("Total: ", time_sum)
        print(f"MSG: [{msg}] Avg Time: {time_sum / REPS}")
