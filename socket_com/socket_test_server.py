from TCPSocket import TCPServer
from UDPSocket import UDPServer
from parameters import *

if use_TCP:
    server = TCPServer(SERVER=SERVER, NUM_CLIENTS=NUM_CLIENTS, MSG_SIZE=MSG_SIZE, DELAY=DELAY)
else:
    server = UDPServer(
        SERVER=SERVER,
        NUM_CLIENTS=NUM_CLIENTS,
        TIMEOUT=TIMEOUT,
        GRADIENT_SIZE=14728266,
        MSG_SIZE=MSG_SIZE,
        CHUNK=CHUNK,
        DELAY=DELAY,
    )

server.start()
