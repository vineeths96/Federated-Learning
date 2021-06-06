from TCPSocket import TCPServer, TCPKServer
from UDPSocket import UDPServer, UDPKServer
from TCPUDPSocket import TCPUDPServer, TCPUDPKServer
from parameters import *


if use_TCPUDP:
    server = TCPUDPServer(
        SERVER=SERVER,
        NUM_CLIENTS=NUM_CLIENTS,
        TIMEOUT=TIMEOUT,
        GRADIENT_SIZE=MSG_SIZE,
        CHUNK=CHUNK,
        DELAY=DELAY,
    )
elif use_TCP:
    server = TCPKServer(SERVER=SERVER, NUM_CLIENTS=NUM_CLIENTS, DELAY=DELAY)
else:
    server = UDPServer(
        SERVER=SERVER,
        NUM_CLIENTS=NUM_CLIENTS,
        TIMEOUT=TIMEOUT,
        GRADIENT_SIZE=MSG_SIZE,
        CHUNK=CHUNK,
        DELAY=DELAY,
    )

server.start()
