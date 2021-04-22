import time
import torch
from socket_com import ClientTCP, ClientUDP


use_TCP = True
# use_TCP = False

if use_TCP:
    REPS = 1
    MSG_SIZES = [25e5]
    # MSG_SIZES = [1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 25e6]
    client = ClientTCP(SERVER="10.217.22.85")

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

        time.sleep(1)
        # client.send_EOT()
else:
    REPS = 1
    MSG_SIZES = [25e5]
    # MSG_SIZES = [1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 25e6]
    client = ClientUDP(SERVER="10.217.22.85", CHUNK=1000, DELAY=1000e-6)

    for msg in MSG_SIZES:
        time_sum = 0
        for i in range(REPS):
            message = torch.cat([torch.arange(msg).unsqueeze(1), 1 * torch.arange(msg).unsqueeze(1)], dim=-1).to(
                torch.float32
            )

            start = time.time()
            client.send(message.clone())
            time_elapsed = time.time() - start
            time_sum += time_elapsed

        print("Total: ", time_sum)
        print(f"MSG: [{msg}] Avg Time: {time_sum / REPS}")

        client.receive()
