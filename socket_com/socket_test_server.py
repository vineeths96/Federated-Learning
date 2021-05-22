from TCPSocket import TCPServer, TCPKServer
from UDPSocket import UDPServer, UDPKServer
from TCPUDPSocket import TCPUDPKServer
from parameters import *


if use_TCPUDP:
    server = TCPUDPKServer(
        SERVER=SERVER,
        NUM_CLIENTS=NUM_CLIENTS,
        TIMEOUT=TIMEOUT,
        GRADIENT_SIZE=10000,
        CHUNK=CHUNK,
        DELAY=DELAY,)
elif use_TCP:
    server = TCPKServer(SERVER=SERVER, NUM_CLIENTS=NUM_CLIENTS, DELAY=DELAY)
else:
    server = UDPServer(
        SERVER=SERVER,
        NUM_CLIENTS=NUM_CLIENTS,
        TIMEOUT=TIMEOUT,
        GRADIENT_SIZE=14728266,
        CHUNK=CHUNK,
        DELAY=DELAY,
    )

server.start()
