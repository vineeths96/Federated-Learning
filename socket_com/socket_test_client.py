import time
import torch
from TCPSocket import ClientTCP
from UDPSocket import ClientUDP
from parameters import *


if use_TCP:
    client = ClientTCP(SERVER=SERVER, DELAY=DELAY)

    for msg in MSG_SIZES:
        time_sum = 0
        for i in range(REPS):
            message = torch.cat([torch.arange(msg).unsqueeze(1), 1 * torch.arange(msg).unsqueeze(1)], dim=-1).to(
                torch.float32
            )

            start = time.time()
            client.send(message)
            time_elapsed = time.time() - start
            time_sum += time_elapsed

        print("Total: ", time_sum)
        print(f"MSG: [{msg}] Avg Time: {time_sum / REPS}")

        client.receive()
else:
    client = ClientUDP(SERVER=SERVER, CHUNK=CHUNK, DELAY=DELAY)

    for msg in MSG_SIZES:
        time_sum = 0
        for i in range(REPS):
            message = torch.cat([torch.arange(msg).unsqueeze(1), 1 * torch.arange(msg).unsqueeze(1)], dim=-1).to(
                torch.float32
            )

            start = time.time()
            client.send(message.clone())
            # client.receive()
            client.receive_TCP_EOT()
            time_elapsed = time.time() - start
            time_sum += time_elapsed

        print("Total: ", time_sum)
        print(f"MSG: [{msg}] Avg Time: {time_sum / REPS}")

        client.receive()
