import time
import torch
from TCPSocket import TCPClient
from UDPSocket import UDPClient
from parameters import *


if use_TCP:
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

            message = torch.cat([torch.arange(msg).unsqueeze(1), 1 * torch.arange(msg).unsqueeze(1)], dim=-1).to(
                torch.float32
            )

            start = time.time()
            client.send(message.clone())
            if UDP_DEBUG:
                client.receive_TCP_EOT()
            else:
                client.receive()
            time_elapsed = time.time() - start
            time_sum += time_elapsed

        print("Total: ", time_sum)
        print(f"MSG: [{msg}] Avg Time: {time_sum / REPS}")
