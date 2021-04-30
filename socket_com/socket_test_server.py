from TCPSocket import TCPServer
from UDPSocket import UDPServer
from parameters import *

if use_TCP:
    server = TCPServer(SERVER=SERVER, MSG_SIZE=MSG_SIZE, DELAY=DELAY)
else:
    server = UDPServer(SERVER=SERVER, MSG_SIZE=MSG_SIZE, CHUNK=CHUNK, DELAY=DELAY)

server.start()
