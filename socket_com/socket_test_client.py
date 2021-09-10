import time
import torch
import argparse
from TCPSocket import TCPClient, TCPKClient
from UDPSocket import UDPClient, UDPKClient
from TCPUDPSocket import TCPUDPClient, TCPUDPKClient
from parameters import *

torch.manual_seed(42)
GRADIENT_SIZE = 14728266


if use_TCPUDP:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    local_rank = args.local_rank
    world_size = args.world_size

    for msg in MSG_SIZES:
        time_sum = 0
        for i in range(REPS):
            client = TCPUDPClient(
                SERVER=SERVER, TIMEOUT=TIMEOUT, DELAY=DELAY, CHUNK=CHUNK, LOCAL_RANK=local_rank, GRADIENT_SIZE=MSG_SIZE
            )

            message = torch.arange(msg).to(torch.float32)

            start = time.time()
            client.sendTCP_SOT()
            client.sendUDP(message)
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
            client = TCPKClient(SERVER=SERVER, DELAY=DELAY)

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
